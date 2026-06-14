#pragma once

#include <stdint.h>

namespace ads1299 {

constexpr uint8_t kCmdWakeup = 0x02;
constexpr uint8_t kCmdStandby = 0x04;
constexpr uint8_t kCmdReset = 0x06;
constexpr uint8_t kCmdStart = 0x08;
constexpr uint8_t kCmdStop = 0x0A;
constexpr uint8_t kCmdRdatac = 0x10;
constexpr uint8_t kCmdSdatac = 0x11;
constexpr uint8_t kCmdRdata = 0x12;

constexpr uint8_t kRegConfig1 = 0x01;
constexpr uint8_t kRegConfig2 = 0x02;
constexpr uint8_t kRegConfig3 = 0x03;
constexpr uint8_t kRegLoff = 0x04;
constexpr uint8_t kRegCh1Set = 0x05;
constexpr uint8_t kRegBiasSensp = 0x0D;
constexpr uint8_t kRegBiasSensn = 0x0E;
constexpr uint8_t kRegMisc1 = 0x15;

constexpr uint8_t kRate1000Sps = 0b100;
constexpr uint8_t kConfig1Base = 0b10010000;
// CONFIG1 bit 6:
// 0 = Daisy-chain mode, 1 = Multiple readback mode.
constexpr uint8_t kConfig1MultipleReadback = 0b01000000;

constexpr uint8_t kConfig2InternalRef = 0b11010000;

constexpr uint8_t kConfig3InternalRefBuffer = 0b11101100;
constexpr uint8_t kConfig3NoBiasDrive = 0b11100000;

constexpr uint8_t kMisc1Srb1Enable = 0b00100000;

constexpr uint8_t kGain12 = 0b101;
constexpr uint8_t kGain24 = 0b110;
constexpr uint8_t kGain4 = 0b010;

constexpr uint8_t kMuxNormalElectrode = 0b000;
constexpr uint8_t kMuxInputShort = 0b001;
constexpr uint8_t kMuxInternalTest = 0b101;
constexpr uint8_t kSrb2Enable = 0b00001000;

inline uint8_t MakeChSet(uint8_t gain_code) {
  return static_cast<uint8_t>((gain_code << 4) | kMuxNormalElectrode);
}

inline uint8_t MakeChSetMux(uint8_t gain_code, uint8_t mux_code, bool srb2_enable = false) {
  return static_cast<uint8_t>((gain_code << 4) | (srb2_enable ? kSrb2Enable : 0) |
                              (mux_code & 0x07));
}

}  // namespace ads1299
