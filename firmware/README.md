
```bash
pio run -t upload
pio device monitor
```


# nerd shit 
## Pin map (exact)

EEG SPI (FSPI, daisy, shared CS):
- DRDY: GPIO4
- CS: GPIO10
- MOSI: GPIO11
- MISO: GPIO13
- SCLK: GPIO12

EMG SPI (HSPI, parallel, separate CS):
- DRDY A: GPIO3
- DRDY B: GPIO2
- CS A: GPIO36
- CS B: GPIO37
- MOSI: GPIO34
- MISO: GPIO35
- SCLK: GPIO33

Shared:
- START: GPIO16
- RESET: GPIO17

## Register intent (v1)

- Sample rate: `1000 SPS`
- Internal reference mode enabled
- EEG daisy uses `CONFIG1.DAISY_EN = 0` (datasheet daisy-chain mode)
- `MISC1.SRB1 = 1` for common reference mode (`INxP - SRB1`)
- EEG gain: `4x`
- EMG gain: `12x`
- Bias derivation:
  - EEG daisy pair: enabled
  - EMG chip A + B: disabled

This keeps configuration simple and centralized in `include/config.h` and `include/ads1299_regs.h`.

## Bring-up fallback (v1 convenience)

To verify USB + parser path before ADS DRDY is healthy, firmware can emit synthetic frames:
- enabled by `kEnableSyntheticFallback` in `include/config.h`
- starts after `kSyntheticStartAfterMs` without aligned DRDY
- emits at `kSyntheticRateHz`

Synthetic frames use the same binary packet format and channel ordering as real data.

For strict transport debugging, `kBypassAdsForBringup` skips ADS setup entirely and streams synthetic frames immediately. Set it back to `false` once serial framing is confirmed.

For EEG daisy debugging, `kDiagnosticTextOnly` prints raw EEG daisy blocks and decoded EEG channels as text for `kDiagnosticLines` samples (no binary streaming while enabled).

`kEegSplitMuxDiagnostic` can force one EEG daisy chip to `MUX=001` (input short) and the other to `MUX=000` (normal input) as a hard A/B identity test for chain distinctness.

`kEegAllInputShortDiagnostic` forces both EEG daisy chips to `MUX=001` (all inputs shorted) to validate non-saturated baseline/noise-floor behavior independent of electrodes.

`kEegInternalTestSignalDiagnostic` forces both EEG daisy chips to `MUX=101` (internal test source) for parser and channel-map verification without electrodes.

EMG diagnostics are available with analogous toggles:
- `kEmgSplitMuxDiagnostic`
- `kEmgAllInputShortDiagnostic`
- `kEmgInternalTestSignalDiagnostic`
- `kEmgDiagnosticTextOnly` + `kEmgDiagnosticLines`

## Frame contract (USB CDC payload)

Each packet is one sample row of 32 channels:

```
struct SampleFrameV1 {
  uint32 magic;        // 0x574F4F44 ("WOOD")
  uint16 version;      // 1
  uint16 frame_bytes;  // sizeof(SampleFrameV1)
  uint64 mcu_time_us;
  uint64 sample_index;
  int32  channels[32]; // 0..15 EEG, 16..31 EMG
  uint32 checksum;     // xor checksum over prior bytes
}
```

Channel order is fixed:
- `channels[0..15]`: EEG1..EEG16
- `channels[16..31]`: EMG1..EMG16

Chip-to-slot mapping is explicit in `include/config.h`:
- `kEegDaisyMisoFirstIsChipA`: controls daisy-chain block ordering
- `kEmgChipAFirst`: controls EMG chip order in channels `16..31`

Default daisy mapping assumes the MISO-near chip shifts out first (common ADS1299 daisy behavior).

EEG daisy parsing also supports optional second-chip status-word auto-detect:
- `kEegDaisyAutoDetectSecondStatus`
- `kEegDaisySecondHasStatus` (used when auto-detect is disabled)
