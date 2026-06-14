#include <Arduino.h>
#include <SPI.h>
#include <esp_timer.h>
#include <driver/gpio.h>

#include "ads1299_regs.h"
#include "config.h"

namespace {

SPIClass eeg_spi(FSPI);
SPIClass emg_spi(HSPI);
portMUX_TYPE drdy_mux = portMUX_INITIALIZER_UNLOCKED;

constexpr uint8_t EmgCsA() { return cfg::kEmgSwapABPins ? cfg::kEmgCsBPin : cfg::kEmgCsAPin; }
constexpr uint8_t EmgCsB() { return cfg::kEmgSwapABPins ? cfg::kEmgCsAPin : cfg::kEmgCsBPin; }
constexpr bool EmgUseA() { return cfg::kEmgActiveMode != 2; }
constexpr bool EmgUseB() { return cfg::kEmgActiveMode != 1; }
constexpr uint8_t kExpectedEegConfig1 = ads1299::kConfig1Base | ads1299::kRate1000Sps;
constexpr uint8_t kExpectedEegConfig2 = ads1299::kConfig2InternalRef;
constexpr uint8_t kExpectedEegConfig3 = ads1299::kConfig3InternalRefBuffer;
constexpr uint8_t kExpectedEmgConfig1 = ads1299::kConfig1Base | ads1299::kRate1000Sps;
constexpr uint8_t kExpectedEmgConfig2 = ads1299::kConfig2InternalRef;
constexpr uint8_t kExpectedEmgConfig3 = ads1299::kConfig3NoBiasDrive;
// External-clocked ADS chains need extra settle after RESET deassert.
constexpr uint16_t kPostResetClockSettleMs = 100;
constexpr uint8_t EmgDrdyA() {
  return cfg::kEmgSwapABPins ? cfg::kEmgDrdyBPin : cfg::kEmgDrdyAPin;
}
constexpr uint8_t EmgDrdyB() {
  return cfg::kEmgSwapABPins ? cfg::kEmgDrdyAPin : cfg::kEmgDrdyBPin;
}

volatile bool eeg_drdy = false;
volatile uint32_t eeg_drdy_edges = 0;
volatile bool emg_a_drdy = false;
volatile bool emg_b_drdy = false;
volatile uint32_t emg_a_drdy_edges = 0;
volatile uint32_t emg_b_drdy_edges = 0;

struct __attribute__((packed)) SampleFrameV1 {
  uint32_t magic;
  uint16_t version;
  uint16_t frame_bytes;
  uint64_t mcu_time_us;
  uint64_t sample_index;
  int32_t channels[cfg::kChannelCount];
  uint32_t checksum;
};

uint32_t XorChecksum(const uint8_t *data, size_t length) {
  uint32_t checksum = 0xA5A55A5Au;
  for (size_t i = 0; i < length; ++i) {
    checksum ^= static_cast<uint32_t>(data[i]) << ((i % 4) * 8);
  }
  return checksum;
}

void EmitFrame(SampleFrameV1 &frame) {
  frame.magic = cfg::kMagic;
  frame.version = cfg::kFrameVersion;
  frame.frame_bytes = sizeof(SampleFrameV1);
  frame.mcu_time_us = static_cast<uint64_t>(esp_timer_get_time());
  frame.checksum = XorChecksum(reinterpret_cast<const uint8_t *>(&frame),
                               sizeof(SampleFrameV1) - sizeof(frame.checksum));
  Serial.write(reinterpret_cast<const uint8_t *>(&frame), sizeof(SampleFrameV1));
}

uint8_t AdsGainCodeFromValue(uint8_t gain) {
  switch (gain) {
    case 1:
      return 0b000;
    case 2:
      return 0b001;
    case 4:
      return ads1299::kGain4;
    case 6:
      return 0b011;
    case 8:
      return 0b100;
    case 12:
      return ads1299::kGain12;
    case 24:
    default:
      return ads1299::kGain24;
  }
}

int32_t SignExtend24(const uint8_t *ptr) {
  int32_t value = (static_cast<int32_t>(ptr[0]) << 16) |
                  (static_cast<int32_t>(ptr[1]) << 8) |
                  static_cast<int32_t>(ptr[2]);
  if (value & 0x00800000) {
    value |= 0xFF000000;
  }
  return value;
}

void IRAM_ATTR OnEegDrdy() {
  portENTER_CRITICAL_ISR(&drdy_mux);
  eeg_drdy = true;
  ++eeg_drdy_edges;
  portEXIT_CRITICAL_ISR(&drdy_mux);
}

void IRAM_ATTR OnEmgADrdy() {
  portENTER_CRITICAL_ISR(&drdy_mux);
  emg_a_drdy = true;
  ++emg_a_drdy_edges;
  portEXIT_CRITICAL_ISR(&drdy_mux);
}

void IRAM_ATTR OnEmgBDrdy() {
  portENTER_CRITICAL_ISR(&drdy_mux);
  emg_b_drdy = true;
  ++emg_b_drdy_edges;
  portEXIT_CRITICAL_ISR(&drdy_mux);
}

bool IsEmgChipSelect(uint8_t cs_pin) { return cs_pin == EmgCsA() || cs_pin == EmgCsB(); }

void SelectChipForSpi(uint8_t cs_pin) {
  if (IsEmgChipSelect(cs_pin)) {
    // Keep the non-selected EMG chip deselected to avoid bus contention.
    digitalWrite(EmgCsA(), HIGH);
    digitalWrite(EmgCsB(), HIGH);
    delayMicroseconds(1);
  }
  digitalWrite(cs_pin, LOW);
}

void DeselectChipForSpi(uint8_t cs_pin) {
  digitalWrite(cs_pin, HIGH);
  if (IsEmgChipSelect(cs_pin)) {
    digitalWrite(EmgCsA(), HIGH);
    digitalWrite(EmgCsB(), HIGH);
  }
}

void SpiSendCommand(SPIClass &spi, uint8_t cs_pin, uint8_t command) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  SelectChipForSpi(cs_pin);
  spi.transfer(command);
  DeselectChipForSpi(cs_pin);
  spi.endTransaction();
  delayMicroseconds(3);
}

void SpiSendCommandDaisy(SPIClass &spi, uint8_t cs_pin, uint8_t command, uint8_t chip_count) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_pin, LOW);
  for (uint8_t i = 0; i < chip_count; ++i) {
    spi.transfer(command);
  }
  digitalWrite(cs_pin, HIGH);
  spi.endTransaction();
  delayMicroseconds(3);
}

void SpiWriteRegisters(SPIClass &spi, uint8_t cs_pin, uint8_t start_reg, const uint8_t *values,
                       size_t count) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  SelectChipForSpi(cs_pin);
  spi.transfer(static_cast<uint8_t>(0x40 | start_reg));
  spi.transfer(static_cast<uint8_t>(count - 1));
  for (size_t i = 0; i < count; ++i) {
    spi.transfer(values[i]);
  }
  DeselectChipForSpi(cs_pin);
  spi.endTransaction();
  delayMicroseconds(3);
}

void SpiReadRegisters(SPIClass &spi, uint8_t cs_pin, uint8_t start_reg, uint8_t *out_values,
                      size_t count) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  SelectChipForSpi(cs_pin);
  spi.transfer(static_cast<uint8_t>(0x20 | start_reg));
  spi.transfer(static_cast<uint8_t>(count - 1));
  for (size_t i = 0; i < count; ++i) {
    out_values[i] = spi.transfer(0x00);
  }
  DeselectChipForSpi(cs_pin);
  spi.endTransaction();
  delayMicroseconds(3);
}

void SpiWriteRegistersDaisySame(SPIClass &spi, uint8_t cs_pin, uint8_t start_reg,
                                const uint8_t *values, size_t count, uint8_t chip_count) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_pin, LOW);
  for (uint8_t chip = 0; chip < chip_count; ++chip) {
    spi.transfer(static_cast<uint8_t>(0x40 | start_reg));
    spi.transfer(static_cast<uint8_t>(count - 1));
    for (size_t i = 0; i < count; ++i) {
      spi.transfer(values[i]);
    }
  }
  digitalWrite(cs_pin, HIGH);
  spi.endTransaction();
  delayMicroseconds(3);
}

void SpiWriteRegistersDaisyDual(SPIClass &spi, uint8_t cs_pin, uint8_t start_reg,
                                const uint8_t *chip0_values, const uint8_t *chip1_values,
                                size_t count) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_pin, LOW);
  spi.transfer(static_cast<uint8_t>(0x40 | start_reg));
  spi.transfer(static_cast<uint8_t>(count - 1));
  for (size_t i = 0; i < count; ++i) {
    spi.transfer(chip0_values[i]);
  }
  spi.transfer(static_cast<uint8_t>(0x40 | start_reg));
  spi.transfer(static_cast<uint8_t>(count - 1));
  for (size_t i = 0; i < count; ++i) {
    spi.transfer(chip1_values[i]);
  }
  digitalWrite(cs_pin, HIGH);
  spi.endTransaction();
  delayMicroseconds(3);
}

void SpiReadRegistersDaisy(SPIClass &spi, uint8_t cs_pin, uint8_t start_reg, uint8_t *out_values,
                           size_t count_per_chip, uint8_t chip_count) {
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_pin, LOW);
  for (uint8_t chip = 0; chip < chip_count; ++chip) {
    spi.transfer(static_cast<uint8_t>(0x20 | start_reg));
    spi.transfer(static_cast<uint8_t>(count_per_chip - 1));
  }
  for (size_t i = 0; i < (count_per_chip * chip_count); ++i) {
    out_values[i] = spi.transfer(0x00);
  }
  digitalWrite(cs_pin, HIGH);
  spi.endTransaction();
  delayMicroseconds(3);
}

void ConfigureChipCommon(SPIClass &spi, uint8_t cs_pin, bool daisy_enable, bool bias_enabled,
                         uint8_t channel_gain_code, bool enable_rdatac = true,
                         uint8_t channel_mux_code = ads1299::kMuxNormalElectrode) {
  uint8_t config1 = ads1299::kConfig1Base | ads1299::kRate1000Sps;
  if (!daisy_enable) {
    // Keep default daisy-chain mode bit cleared in v1 for stable streaming.
    // Multiple-readback mode (bit6=1) is not used.
    config1 &= static_cast<uint8_t>(~ads1299::kConfig1MultipleReadback);
  }

  SpiSendCommand(spi, cs_pin, ads1299::kCmdWakeup);
  SpiSendCommand(spi, cs_pin, ads1299::kCmdSdatac);

  const uint8_t common_registers[] = {config1, ads1299::kConfig2InternalRef,
                                      bias_enabled ? ads1299::kConfig3InternalRefBuffer
                                                   : ads1299::kConfig3NoBiasDrive};
  SpiWriteRegisters(spi, cs_pin, ads1299::kRegConfig1, common_registers,
                    sizeof(common_registers));

  const uint8_t misc1 = ads1299::kMisc1Srb1Enable;
  SpiWriteRegisters(spi, cs_pin, ads1299::kRegMisc1, &misc1, 1);

  const uint8_t bias_mask = bias_enabled ? 0xFF : 0x00;
  SpiWriteRegisters(spi, cs_pin, ads1299::kRegBiasSensp, &bias_mask, 1);
  SpiWriteRegisters(spi, cs_pin, ads1299::kRegBiasSensn, &bias_mask, 1);

  uint8_t ch_settings[cfg::kAdsChannelsPerChip];
  for (size_t i = 0; i < cfg::kAdsChannelsPerChip; ++i) {
    ch_settings[i] = ads1299::MakeChSetMux(channel_gain_code, channel_mux_code, false);
  }
  SpiWriteRegisters(spi, cs_pin, ads1299::kRegCh1Set, ch_settings, sizeof(ch_settings));

  if (enable_rdatac) {
    SpiSendCommand(spi, cs_pin, ads1299::kCmdRdatac);
  }
}

void ConfigureEegDaisyPair() {
  // Datasheet: DAISY_EN=0 selects daisy-chain mode.
  uint8_t config1 = ads1299::kConfig1Base | ads1299::kRate1000Sps;
  const uint8_t common_registers[] = {config1, ads1299::kConfig2InternalRef,
                                      ads1299::kConfig3InternalRefBuffer};
  const uint8_t misc1 = ads1299::kMisc1Srb1Enable;
  const uint8_t bias_sensp_mask = 0xFF;
  const uint8_t bias_sensn_mask = 0x00;
  const uint8_t eeg_gain_code = AdsGainCodeFromValue(cfg::kEegGain);

  uint8_t ch_settings[cfg::kAdsChannelsPerChip];
  for (size_t i = 0; i < cfg::kAdsChannelsPerChip; ++i) {
    ch_settings[i] = ads1299::MakeChSet(eeg_gain_code);
  }
  uint8_t ch_settings_short[cfg::kAdsChannelsPerChip];
  for (size_t i = 0; i < cfg::kAdsChannelsPerChip; ++i) {
    ch_settings_short[i] = ads1299::MakeChSetMux(eeg_gain_code, ads1299::kMuxInputShort, false);
  }
  uint8_t ch_settings_test[cfg::kAdsChannelsPerChip];
  for (size_t i = 0; i < cfg::kAdsChannelsPerChip; ++i) {
    ch_settings_test[i] =
        ads1299::MakeChSetMux(eeg_gain_code, ads1299::kMuxInternalTest, false);
  }

  // In daisy mode, mirror each command/register stream for both chips.
  SpiSendCommandDaisy(eeg_spi, cfg::kEegCsPin, ads1299::kCmdWakeup, cfg::kEegChipCount);
  SpiSendCommandDaisy(eeg_spi, cfg::kEegCsPin, ads1299::kCmdSdatac, cfg::kEegChipCount);
  SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegConfig1, common_registers,
                             sizeof(common_registers), cfg::kEegChipCount);
  SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegMisc1, &misc1, 1,
                             cfg::kEegChipCount);
  SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegBiasSensp, &bias_sensp_mask,
                             1,
                             cfg::kEegChipCount);
  SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegBiasSensn, &bias_sensn_mask,
                             1,
                             cfg::kEegChipCount);
  if (cfg::kEegInternalTestSignalDiagnostic) {
    SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegCh1Set, ch_settings_test,
                               sizeof(ch_settings_test), cfg::kEegChipCount);
  } else if (cfg::kEegAllInputShortDiagnostic) {
    SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegCh1Set, ch_settings_short,
                               sizeof(ch_settings_short), cfg::kEegChipCount);
  } else if (cfg::kEegSplitMuxDiagnostic) {
    // Diagnostic split: one chip shorted inputs, other chip normal electrode inputs.
    // If both daisy blocks still match exactly, chain data path is not distinct.
    SpiWriteRegistersDaisyDual(eeg_spi, cfg::kEegCsPin, ads1299::kRegCh1Set, ch_settings_short,
                               ch_settings, sizeof(ch_settings));
  } else {
    SpiWriteRegistersDaisySame(eeg_spi, cfg::kEegCsPin, ads1299::kRegCh1Set, ch_settings,
                               sizeof(ch_settings), cfg::kEegChipCount);
  }
}

bool EegRegisterSnapshotLooksValid(const uint8_t *regs, size_t count) {
  if (count < 5) {
    return false;
  }
  return regs[0] != 0x00 && regs[1] == kExpectedEegConfig1 && regs[2] == kExpectedEegConfig2 &&
         regs[3] == kExpectedEegConfig3;
}

bool ConfigureEegPairWithRetry() {
  constexpr uint8_t kMaxAttempts = 8;
  uint8_t regs[cfg::kEegChipCount * 5] = {0};  // (ID..LOFF) per chip
  for (uint8_t attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    ConfigureEegDaisyPair();
    SpiReadRegistersDaisy(eeg_spi, cfg::kEegCsPin, 0x00, regs, 5, cfg::kEegChipCount);
    const bool ok_chip0 = EegRegisterSnapshotLooksValid(&regs[0], 5);
    const bool ok_chip1 = EegRegisterSnapshotLooksValid(&regs[5], 5);
    const bool ok = ok_chip0 && ok_chip1;
    Serial.print("EEG_INIT attempt=");
    Serial.print(static_cast<unsigned>(attempt));
    Serial.print(" regs0=[");
    for (size_t i = 0; i < 5; ++i) {
      if (i) {
        Serial.print(',');
      }
      Serial.print(regs[i], HEX);
    }
    Serial.print("] regs1=[");
    for (size_t i = 0; i < 5; ++i) {
      if (i) {
        Serial.print(',');
      }
      Serial.print(regs[5 + i], HEX);
    }
    Serial.print("] ok=");
    Serial.println(ok ? 1 : 0);
    if (ok) {
      return true;
    }
    delay(4);
  }
  return false;
}

void StartEegPair() {
  SpiSendCommandDaisy(eeg_spi, cfg::kEegCsPin, ads1299::kCmdStart, cfg::kEegChipCount);
  delay(3);
  Serial.println("EEG_START pair");
}

bool EmgRegisterSnapshotLooksValid(const uint8_t *regs) {
  return regs[0] != 0x00 && regs[1] == kExpectedEmgConfig1 && regs[2] == kExpectedEmgConfig2 &&
         regs[3] == kExpectedEmgConfig3;
}

uint8_t EmgChannelMuxCode() {
  if (cfg::kEmgInternalTestSignalDiagnostic) {
    return ads1299::kMuxInternalTest;
  }
  if (cfg::kEmgAllInputShortDiagnostic) {
    return ads1299::kMuxInputShort;
  }
  return ads1299::kMuxNormalElectrode;
}

bool ConfigureEmgChipWithRetry(uint8_t cs_pin, uint8_t gain_code, const char *chip_name) {
  constexpr uint8_t kMaxAttempts = 8;
  const uint8_t mux_code = EmgChannelMuxCode();
  for (uint8_t attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    ConfigureChipCommon(emg_spi, cs_pin, false, false, gain_code, false, mux_code);
    uint8_t regs[5] = {0};
    SpiReadRegisters(emg_spi, cs_pin, 0x00, regs, sizeof(regs));
    const bool ok = EmgRegisterSnapshotLooksValid(regs);
    Serial.print("EMG_INIT chip=");
    Serial.print(chip_name);
    Serial.print(" attempt=");
    Serial.print(static_cast<unsigned>(attempt));
    Serial.print(" regs=[");
    for (size_t i = 0; i < sizeof(regs); ++i) {
      if (i) {
        Serial.print(',');
      }
      Serial.print(regs[i], HEX);
    }
    Serial.print("] ok=");
    Serial.println(ok ? 1 : 0);
    if (ok) {
      return true;
    }
    delay(4);
  }
  return false;
}

void StartEmgChip(uint8_t cs_pin, const char *chip_name) {
  SpiSendCommand(emg_spi, cs_pin, ads1299::kCmdStart);
  delay(3);
  Serial.print("EMG_START chip=");
  Serial.println(chip_name);
}

void RecoverEmgChipIfNoDrdy(uint8_t cs_pin, volatile uint32_t &edge_counter, uint8_t gain_code,
                            const char *chip_name) {
  const uint32_t before = edge_counter;
  delay(50);
  const uint32_t after = edge_counter;
  if (after != before) {
    return;
  }
  Serial.print("EMG_RECOVER chip=");
  Serial.print(chip_name);
  Serial.println(" reason=no_drdy_edges");
  const bool ok = ConfigureEmgChipWithRetry(cs_pin, gain_code, chip_name);
  if (ok) {
    StartEmgChip(cs_pin, chip_name);
  }
}

void RecoverEegPairIfNoDrdy() {
  const uint32_t before = eeg_drdy_edges;
  delay(50);
  const uint32_t after = eeg_drdy_edges;
  if (after != before) {
    return;
  }
  Serial.println("EEG_RECOVER reason=no_drdy_edges");
  // Daisy register readback validation can be brittle; recover by reapplying
  // config and restarting conversion unconditionally.
  ConfigureEegDaisyPair();
  StartEegPair();
}

void ConfigureAllAds() {
  digitalWrite(cfg::kStartPin, LOW);
  digitalWrite(cfg::kResetPin, LOW);
  delay(10);
  digitalWrite(cfg::kResetPin, HIGH);
  delay(kPostResetClockSettleMs);

  // EEG pair is daisy-chained under shared CS.
  // Do not gate startup on daisy readback validation.
  ConfigureEegDaisyPair();
  StartEegPair();

  // EMG chips are independent on separate CS lines.
  const uint8_t emg_gain_code = AdsGainCodeFromValue(cfg::kEmgGain);
  if (EmgUseA()) {
    const bool ok_a = ConfigureEmgChipWithRetry(EmgCsA(), emg_gain_code, "A");
    if (ok_a) {
      StartEmgChip(EmgCsA(), "A");
    }
  }
  if (EmgUseB()) {
    const bool ok_b = ConfigureEmgChipWithRetry(EmgCsB(), emg_gain_code, "B");
    if (ok_b) {
      StartEmgChip(EmgCsB(), "B");
    }
  }

  // START pin controls conversion for all chips simultaneously.
  digitalWrite(cfg::kStartPin, HIGH);
  delay(5);
  RecoverEegPairIfNoDrdy();
  if (EmgUseA()) {
    RecoverEmgChipIfNoDrdy(EmgCsA(), emg_a_drdy_edges, emg_gain_code, "A");
  }
  if (EmgUseB()) {
    RecoverEmgChipIfNoDrdy(EmgCsB(), emg_b_drdy_edges, emg_gain_code, "B");
  }
}

void ReadEegDaisy(int32_t *dst_16_channels) {
  constexpr size_t kReadBytes = cfg::kAdsFrameBytesPerChip * cfg::kEegChipCount;
  uint8_t buffer[kReadBytes] = {0};

  eeg_spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  digitalWrite(cfg::kEegCsPin, LOW);
  for (uint8_t i = 0; i < cfg::kEegChipCount; ++i) {
    eeg_spi.transfer(ads1299::kCmdRdata);
  }
  delayMicroseconds(3);
  for (size_t i = 0; i < kReadBytes; ++i) {
    buffer[i] = eeg_spi.transfer(0x00);
  }
  digitalWrite(cfg::kEegCsPin, HIGH);
  eeg_spi.endTransaction();

  // ADS1299 daisy-chain ordering is physical-chain dependent.
  // Default here assumes bytes nearest MCU MISO arrive first.
  const uint8_t *first_block = &buffer[0];
  const uint8_t *second_block = &buffer[cfg::kAdsFrameBytesPerChip];
  const uint8_t *chip_a = cfg::kEegDaisyMisoFirstIsChipA ? first_block : second_block;
  const uint8_t *chip_b = cfg::kEegDaisyMisoFirstIsChipA ? second_block : first_block;
  const uint8_t *chip_a_data = chip_a + cfg::kAdsStatusBytes;

  bool second_has_status = cfg::kEegDaisySecondHasStatus;
  if (cfg::kEegDaisyAutoDetectSecondStatus) {
    // ADS1299 status starts with fixed upper nibble 0b1100 (0xC*).
    second_has_status = (second_block[0] & 0xF0) == 0xC0;
  }
  const uint8_t *chip_b_data = chip_b + (second_has_status ? cfg::kAdsStatusBytes : 0);
  for (size_t ch = 0; ch < cfg::kAdsChannelsPerChip; ++ch) {
    dst_16_channels[ch] = SignExtend24(&chip_a_data[ch * cfg::kAdsSampleBytesPerChannel]);
    dst_16_channels[ch + cfg::kAdsChannelsPerChip] =
        SignExtend24(&chip_b_data[ch * cfg::kAdsSampleBytesPerChannel]);
  }
}

void ReadSingleChip(SPIClass &spi, uint8_t cs_pin, int32_t *dst_8_channels) {
  uint8_t buffer[cfg::kAdsFrameBytesPerChip] = {0};
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  SelectChipForSpi(cs_pin);
  for (size_t i = 0; i < cfg::kAdsFrameBytesPerChip; ++i) {
    buffer[i] = spi.transfer(0x00);
  }
  DeselectChipForSpi(cs_pin);
  spi.endTransaction();

  for (size_t ch = 0; ch < cfg::kAdsChannelsPerChip; ++ch) {
    dst_8_channels[ch] =
        SignExtend24(&buffer[cfg::kAdsStatusBytes + ch * cfg::kAdsSampleBytesPerChannel]);
  }
}

void ReadSingleChipRdata(SPIClass &spi, uint8_t cs_pin, int32_t *dst_8_channels,
                         uint8_t *status3 = nullptr) {
  uint8_t buffer[cfg::kAdsFrameBytesPerChip] = {0};
  spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  SelectChipForSpi(cs_pin);
  spi.transfer(ads1299::kCmdRdata);
  delayMicroseconds(3);
  for (size_t i = 0; i < cfg::kAdsFrameBytesPerChip; ++i) {
    buffer[i] = spi.transfer(0x00);
  }
  DeselectChipForSpi(cs_pin);
  spi.endTransaction();

  if (status3 != nullptr) {
    status3[0] = buffer[0];
    status3[1] = buffer[1];
    status3[2] = buffer[2];
  }

  for (size_t ch = 0; ch < cfg::kAdsChannelsPerChip; ++ch) {
    dst_8_channels[ch] =
        SignExtend24(&buffer[cfg::kAdsStatusBytes + ch * cfg::kAdsSampleBytesPerChannel]);
  }
}

bool ConsumeAlignedDrdyWindow() {
  bool ready = false;
  portENTER_CRITICAL(&drdy_mux);
  const bool emg_a_ready = !EmgUseA() || emg_a_drdy;
  const bool emg_b_ready = !EmgUseB() || emg_b_drdy;
  if (eeg_drdy && emg_a_ready && emg_b_ready) {
    eeg_drdy = false;
    if (EmgUseA()) {
      emg_a_drdy = false;
    }
    if (EmgUseB()) {
      emg_b_drdy = false;
    }
    ready = true;
  }
  portEXIT_CRITICAL(&drdy_mux);
  return ready;
}

bool ConsumeEegDrdyOnly() {
  bool ready = false;
  portENTER_CRITICAL(&drdy_mux);
  if (eeg_drdy) {
    eeg_drdy = false;
    ready = true;
  }
  portEXIT_CRITICAL(&drdy_mux);
  return ready;
}

bool ConsumeEegDrdyOrLineLow() {
  if (ConsumeEegDrdyOnly()) {
    return true;
  }
  // Fallback path when ISR edge capture is missed: DRDY low still means
  // conversion data is ready to be read.
  return digitalRead(cfg::kEegDrdyPin) == LOW;
}

bool ConsumeEmgAnyDrdy() {
  bool ready = false;
  portENTER_CRITICAL(&drdy_mux);
  const bool emg_a_ready = EmgUseA() && emg_a_drdy;
  const bool emg_b_ready = EmgUseB() && emg_b_drdy;
  if (emg_a_ready || emg_b_ready) {
    if (EmgUseA()) {
      emg_a_drdy = false;
    }
    if (EmgUseB()) {
      emg_b_drdy = false;
    }
    ready = true;
  }
  portEXIT_CRITICAL(&drdy_mux);
  return ready;
}

void FillSyntheticChannels(uint64_t sample_index, int32_t *channels) {
  const int32_t base = static_cast<int32_t>((sample_index % 2000) - 1000);
  for (size_t i = 0; i < cfg::kChannelCount; ++i) {
    const int32_t scale = (i < 16) ? 6 : 10;
    channels[i] = (base * scale) + static_cast<int32_t>(i * 100);
  }
}

void ReadEegDaisyRaw(uint8_t *dst_54_bytes) {
  constexpr size_t kReadBytes = cfg::kAdsFrameBytesPerChip * cfg::kEegChipCount;
  eeg_spi.beginTransaction(SPISettings(cfg::kSpiClockHz, MSBFIRST, SPI_MODE1));
  digitalWrite(cfg::kEegCsPin, LOW);
  for (uint8_t i = 0; i < cfg::kEegChipCount; ++i) {
    eeg_spi.transfer(ads1299::kCmdRdata);
  }
  delayMicroseconds(3);
  for (size_t i = 0; i < kReadBytes; ++i) {
    dst_54_bytes[i] = eeg_spi.transfer(0x00);
  }
  digitalWrite(cfg::kEegCsPin, HIGH);
  eeg_spi.endTransaction();
}

void PrintHexBytes(const uint8_t *data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i) {
      Serial.print(' ');
    }
    if (data[i] < 16) {
      Serial.print('0');
    }
    Serial.print(data[i], HEX);
  }
}

void EmitEegDiagnosticLine(uint64_t sample_index) {
  uint8_t raw[cfg::kAdsFrameBytesPerChip * cfg::kEegChipCount] = {0};
  int32_t eeg[16] = {0};
  const uint8_t *first_block = &raw[0];
  const uint8_t *second_block = &raw[cfg::kAdsFrameBytesPerChip];

  ReadEegDaisyRaw(raw);
  const uint8_t *chip_a = cfg::kEegDaisyMisoFirstIsChipA ? first_block : second_block;
  const uint8_t *chip_b = cfg::kEegDaisyMisoFirstIsChipA ? second_block : first_block;
  const uint8_t *chip_a_data = chip_a + cfg::kAdsStatusBytes;
  bool second_has_status = cfg::kEegDaisySecondHasStatus;
  if (cfg::kEegDaisyAutoDetectSecondStatus) {
    second_has_status = (second_block[0] & 0xF0) == 0xC0;
  }
  const uint8_t *chip_b_data = chip_b + (second_has_status ? cfg::kAdsStatusBytes : 0);

  for (size_t ch = 0; ch < cfg::kAdsChannelsPerChip; ++ch) {
    eeg[ch] = SignExtend24(&chip_a_data[ch * cfg::kAdsSampleBytesPerChannel]);
    eeg[ch + cfg::kAdsChannelsPerChip] =
        SignExtend24(&chip_b_data[ch * cfg::kAdsSampleBytesPerChannel]);
  }

  Serial.print("DIAG idx=");
  Serial.print(static_cast<unsigned long long>(sample_index));
  Serial.print(" blk0=[");
  PrintHexBytes(raw, cfg::kAdsFrameBytesPerChip);
  Serial.print("] blk1=[");
  PrintHexBytes(&raw[cfg::kAdsFrameBytesPerChip], cfg::kAdsFrameBytesPerChip);
  Serial.print("] eeg=[");
  for (size_t i = 0; i < 16; ++i) {
    if (i) {
      Serial.print(',');
    }
    Serial.print(eeg[i]);
  }
  Serial.println("]");
}

void EmitEmgDiagnosticLine(uint64_t sample_index) {
  int32_t emg_a[8] = {0};
  int32_t emg_b[8] = {0};
  uint8_t regs_a[5] = {0};  // ID..LOFF
  uint8_t regs_b[5] = {0};  // ID..LOFF
  uint8_t status_a[3] = {0};
  uint8_t status_b[3] = {0};
  if (EmgUseA()) {
    ReadSingleChipRdata(emg_spi, EmgCsA(), emg_a, status_a);
    SpiReadRegisters(emg_spi, EmgCsA(), 0x00, regs_a, sizeof(regs_a));
  }
  if (EmgUseB()) {
    ReadSingleChipRdata(emg_spi, EmgCsB(), emg_b, status_b);
    SpiReadRegisters(emg_spi, EmgCsB(), 0x00, regs_b, sizeof(regs_b));
  }

  Serial.print("EMG_DIAG idx=");
  Serial.print(static_cast<unsigned long long>(sample_index));
  Serial.print(" mode=");
  Serial.print(static_cast<unsigned>(cfg::kEmgActiveMode));
  Serial.print(" enabled=(");
  Serial.print(EmgUseA() ? 'A' : '-');
  Serial.print(',');
  Serial.print(EmgUseB() ? 'B' : '-');
  Serial.print(")");
  Serial.print(" drdy_edges=(");
  Serial.print(static_cast<unsigned long>(emg_a_drdy_edges));
  Serial.print(",");
  Serial.print(static_cast<unsigned long>(emg_b_drdy_edges));
  Serial.print(") drdy_level=(");
  Serial.print(digitalRead(EmgDrdyA()));
  Serial.print(",");
  Serial.print(digitalRead(EmgDrdyB()));
  Serial.print(") regsA=[");
  for (size_t i = 0; i < sizeof(regs_a); ++i) {
    if (i) {
      Serial.print(',');
    }
    Serial.print(regs_a[i], HEX);
  }
  Serial.print("] regsB=[");
  for (size_t i = 0; i < sizeof(regs_b); ++i) {
    if (i) {
      Serial.print(',');
    }
    Serial.print(regs_b[i], HEX);
  }
  Serial.print("] statA=[");
  Serial.print(status_a[0], HEX);
  Serial.print(",");
  Serial.print(status_a[1], HEX);
  Serial.print(",");
  Serial.print(status_a[2], HEX);
  Serial.print("] statB=[");
  Serial.print(status_b[0], HEX);
  Serial.print(",");
  Serial.print(status_b[1], HEX);
  Serial.print(",");
  Serial.print(status_b[2], HEX);
  Serial.print("] A=[");
  for (size_t i = 0; i < 8; ++i) {
    if (i) {
      Serial.print(',');
    }
    Serial.print(emg_a[i]);
  }
  Serial.print("] B=[");
  for (size_t i = 0; i < 8; ++i) {
    if (i) {
      Serial.print(',');
    }
    Serial.print(emg_b[i]);
  }
  Serial.println("]");
}

void ApplyAndPrintEmgDriveStrength() {
  if (!cfg::kEmgGpioDriveDiag) {
    return;
  }
  const gpio_num_t pins[] = {static_cast<gpio_num_t>(cfg::kEmgMosiPin),
                             static_cast<gpio_num_t>(cfg::kEmgSclkPin),
                             static_cast<gpio_num_t>(EmgCsA()),
                             static_cast<gpio_num_t>(EmgCsB())};
  for (auto pin : pins) {
    gpio_set_drive_capability(pin, GPIO_DRIVE_CAP_3);  // strongest drive
    gpio_drive_cap_t cap = GPIO_DRIVE_CAP_DEFAULT;
    const esp_err_t err = gpio_get_drive_capability(pin, &cap);
    Serial.print("GPIO_DRIVE pin=");
    Serial.print(static_cast<int>(pin));
    Serial.print(" err=");
    Serial.print(static_cast<int>(err));
    Serial.print(" cap=");
    Serial.println(static_cast<int>(cap));
  }
}

}  // namespace

void setup() {
  Serial.begin(cfg::kUsbBaud);

  if (cfg::kBypassAdsForBringup) {
    return;
  }

  pinMode(cfg::kStartPin, OUTPUT);
  pinMode(cfg::kResetPin, OUTPUT);

  pinMode(cfg::kEegCsPin, OUTPUT);
  pinMode(EmgCsA(), OUTPUT);
  pinMode(EmgCsB(), OUTPUT);
  digitalWrite(cfg::kEegCsPin, HIGH);
  digitalWrite(EmgCsA(), HIGH);
  digitalWrite(EmgCsB(), HIGH);

  pinMode(cfg::kEegDrdyPin, INPUT_PULLUP);
  pinMode(EmgDrdyA(), INPUT_PULLUP);
  pinMode(EmgDrdyB(), INPUT_PULLUP);

  eeg_spi.begin(cfg::kEegSclkPin, cfg::kEegMisoPin, cfg::kEegMosiPin, cfg::kEegCsPin);
  emg_spi.begin(cfg::kEmgSclkPin, cfg::kEmgMisoPin, cfg::kEmgMosiPin, EmgCsA());

  attachInterrupt(digitalPinToInterrupt(cfg::kEegDrdyPin), OnEegDrdy, FALLING);
  if (EmgUseA()) {
    attachInterrupt(digitalPinToInterrupt(EmgDrdyA()), OnEmgADrdy, FALLING);
  }
  if (EmgUseB()) {
    attachInterrupt(digitalPinToInterrupt(EmgDrdyB()), OnEmgBDrdy, FALLING);
  }

  ApplyAndPrintEmgDriveStrength();
  ConfigureAllAds();
}

void loop() {
  static uint64_t sample_index = 0;
  static uint64_t last_real_frame_us = 0;
  static uint64_t last_synth_frame_us = 0;
  static uint16_t diag_lines_emitted = 0;
  static uint16_t emg_diag_lines_emitted = 0;
  static uint64_t last_diag_wait_us = 0;
  static uint64_t last_eeg_recover_attempt_us = 0;
  static uint64_t last_emg_diag_emit_us = 0;

  if (cfg::kBypassAdsForBringup) {
    const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
    const uint64_t synth_period_us = 1000000ULL / cfg::kSyntheticRateHz;
    if ((now_us - last_synth_frame_us) < synth_period_us) {
      return;
    }
    SampleFrameV1 frame{};
    frame.sample_index = sample_index++;
    FillSyntheticChannels(frame.sample_index, frame.channels);
    EmitFrame(frame);
    last_synth_frame_us = now_us;
    return;
  }

  if (cfg::kDiagnosticTextOnly) {
    const bool got_eeg_drdy = ConsumeEegDrdyOrLineLow();
    if (got_eeg_drdy) {
      if (diag_lines_emitted < cfg::kDiagnosticLines) {
        EmitEegDiagnosticLine(sample_index++);
        ++diag_lines_emitted;
      }
    } else {
      const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
      if ((now_us - last_diag_wait_us) > 500000ULL) {
        Serial.print("DIAG waiting_drdy eeg_edges=");
        Serial.print(static_cast<unsigned long>(eeg_drdy_edges));
        Serial.print(" eeg_level=");
        Serial.println(digitalRead(cfg::kEegDrdyPin));
        last_diag_wait_us = now_us;
      }
      if ((now_us - last_eeg_recover_attempt_us) > 2000000ULL) {
        RecoverEegPairIfNoDrdy();
        last_eeg_recover_attempt_us = now_us;
      }
    }
    return;
  }

  if (cfg::kEmgDiagnosticTextOnly) {
    const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
    static bool pulse_state = false;
    pulse_state = !pulse_state;
    digitalWrite(cfg::kEmgMosiPin, pulse_state ? HIGH : LOW);  // scope GPIO34 quickly
    ConsumeEmgAnyDrdy();
    if ((now_us - last_emg_diag_emit_us) > 100000ULL &&
        emg_diag_lines_emitted < cfg::kEmgDiagnosticLines) {
      EmitEmgDiagnosticLine(sample_index++);
      ++emg_diag_lines_emitted;
      last_emg_diag_emit_us = now_us;
    }
    if ((now_us - last_diag_wait_us) > 500000ULL) {
      Serial.print("EMG_DIAG heartbeat drdy_edges=(");
      Serial.print(static_cast<unsigned long>(emg_a_drdy_edges));
      Serial.print(",");
      Serial.print(static_cast<unsigned long>(emg_b_drdy_edges));
      Serial.print(") drdy_level=(");
      Serial.print(digitalRead(EmgDrdyA()));
      Serial.print(",");
      Serial.print(digitalRead(EmgDrdyB()));
      Serial.println(")");
      last_diag_wait_us = now_us;
    }
    return;
  }

  const bool got_drdy = ConsumeAlignedDrdyWindow();

  if (got_drdy) {
    SampleFrameV1 frame{};
    frame.sample_index = sample_index++;
    ReadEegDaisy(&frame.channels[0]);
    if (cfg::kEmgChipAFirst) {
      if (EmgUseA()) {
        ReadSingleChipRdata(emg_spi, EmgCsA(), &frame.channels[16]);
      }
      if (EmgUseB()) {
        ReadSingleChipRdata(emg_spi, EmgCsB(), &frame.channels[24]);
      }
    } else {
      if (EmgUseB()) {
        ReadSingleChipRdata(emg_spi, EmgCsB(), &frame.channels[16]);
      }
      if (EmgUseA()) {
        ReadSingleChipRdata(emg_spi, EmgCsA(), &frame.channels[24]);
      }
    }
    EmitFrame(frame);
    last_real_frame_us = static_cast<uint64_t>(esp_timer_get_time());
    return;
  }

  if (!cfg::kEnableSyntheticFallback) {
    return;
  }

  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (last_real_frame_us == 0) {
    last_real_frame_us = now_us;
  }

  const uint64_t idle_us = now_us - last_real_frame_us;
  if (idle_us < static_cast<uint64_t>(cfg::kSyntheticStartAfterMs) * 1000ULL) {
    return;
  }

  const uint64_t synth_period_us = 1000000ULL / cfg::kSyntheticRateHz;
  if ((now_us - last_synth_frame_us) < synth_period_us) {
    return;
  }

  SampleFrameV1 frame{};
  frame.sample_index = sample_index++;
  FillSyntheticChannels(frame.sample_index, frame.channels);
  EmitFrame(frame);
  last_synth_frame_us = now_us;
}
