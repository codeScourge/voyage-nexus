#pragma once

#include <stdint.h>

namespace cfg {

constexpr uint32_t kUsbBaud = 2000000;
constexpr uint32_t kMagic = 0x574f4f44;  // "WOOD"
constexpr uint16_t kFrameVersion = 1;
constexpr uint16_t kChannelCount = 32;
constexpr uint16_t kEegChannels = 16;
constexpr uint16_t kEmgChannels = 16;
constexpr uint16_t kSampleRateHz = 1000;
constexpr uint16_t kSyntheticRateHz = 250;
constexpr uint32_t kSyntheticStartAfterMs = 1000;
constexpr bool kEnableSyntheticFallback = true;
constexpr bool kBypassAdsForBringup = false;

// Per-channel BIAS_SENSP enable (true = channel contributes to bias drive).
constexpr bool kEeg1BiasSensp = true;
constexpr bool kEeg2BiasSensp = true;
constexpr bool kEeg3BiasSensp = true;
constexpr bool kEeg4BiasSensp = true;
constexpr bool kEeg5BiasSensp = true;
constexpr bool kEeg6BiasSensp = true;
constexpr bool kEeg7BiasSensp = true;
constexpr bool kEeg8BiasSensp = true;
constexpr bool kEeg9BiasSensp = true;
constexpr bool kEeg10BiasSensp = true;
constexpr bool kEeg11BiasSensp = true;
constexpr bool kEeg12BiasSensp = true;
constexpr bool kEeg13BiasSensp = true;
constexpr bool kEeg14BiasSensp = true;
constexpr bool kEeg15BiasSensp = true;
constexpr bool kEeg16BiasSensp = true;

constexpr float kAdsVrefVolts = 4.5f;
constexpr uint8_t kEegGain = 6;
constexpr uint8_t kEmgGain = 1;

// EEG bus (FSPI, two ADS1299 chips in daisy chain with shared CS)
constexpr uint8_t kEegDrdyPin = 4;
constexpr uint8_t kEegCsPin = 10;
constexpr uint8_t kEegMosiPin = 11;
constexpr uint8_t kEegMisoPin = 13;
constexpr uint8_t kEegSclkPin = 12;

// EMG bus (HSPI, two ADS1299 chips in parallel with independent CS)
constexpr uint8_t kEmgDrdyAPin = 3;
constexpr uint8_t kEmgDrdyBPin = 2;
constexpr uint8_t kEmgCsAPin = 36;
constexpr uint8_t kEmgCsBPin = 37;
constexpr uint8_t kEmgMosiPin = 34;
constexpr uint8_t kEmgMisoPin = 35;
constexpr uint8_t kEmgSclkPin = 33;

// Shared control lines for all ADS1299 chips
constexpr uint8_t kStartPin = 16;
constexpr uint8_t kResetPin = 17;

constexpr uint32_t kSpiClockHz = 4000000;
constexpr uint8_t kAdsStatusBytes = 3;
constexpr uint8_t kAdsChannelsPerChip = 8;
constexpr uint8_t kAdsSampleBytesPerChannel = 3;
constexpr uint8_t kAdsFrameBytesPerChip =
    kAdsStatusBytes + (kAdsChannelsPerChip * kAdsSampleBytesPerChannel);

constexpr uint8_t kEegChipCount = 2;
constexpr uint8_t kEmgChipCount = 2;

// Channel ordering contract controls:
// - In ADS1299 daisy mode, the chip nearest MCU MISO typically shifts out first.
// - Keep these constants explicit so host labels can be kept stable.
constexpr bool kEegDaisyMisoFirstIsChipA = true;
constexpr bool kEmgChipAFirst = true;
constexpr bool kEegDaisyAutoDetectSecondStatus = false;
constexpr bool kEegDaisySecondHasStatus = true;
constexpr bool kDiagnosticTextOnly = false;
constexpr uint16_t kDiagnosticLines = 20;
constexpr bool kEegSplitMuxDiagnostic = false;
constexpr bool kEegAllInputShortDiagnostic = false;
constexpr bool kEegInternalTestSignalDiagnostic = false;
constexpr bool kEmgDiagnosticTextOnly = false;
constexpr uint16_t kEmgDiagnosticLines = 20;
constexpr bool kEmgAllInputShortDiagnostic = false;
constexpr bool kEmgInternalTestSignalDiagnostic = false;
constexpr bool kEmgSwapABPins = false;
constexpr bool kEmgGpioDriveDiag = true;
// 0 = both EMG chips required, 1 = chip A only, 2 = chip B only.
constexpr uint8_t kEmgActiveMode = 0;
}  // namespace cfg
