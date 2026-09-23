



(replace 1101 with whatever `ls /dev/tty*` ur device is conn to)

Then open the monitor in your browser (default): http://127.0.0.1:8765/

**macOS note:** port `5000` is often taken by AirPlay Receiver, which shows “Access denied / HTTP 403” in Chrome — not this app. Use `8765` (default) or `--http-port 5001`.









# nerd shit

Useful options:
- `--max-seconds 30` run a bounded test
- `--status-interval 1.0` print parser health every second
- `--preview-interval 2.0` print channel snapshot every 2 seconds
- `--list-ports` list serial ports and descriptions
- `--raw-probe --raw-seconds 3` inspect raw bytes and detect frame magic

Healthy run target:
- `rate` near ~1000 frames/s
- `desync` and `bad_crc` stay near zero after startup
- `index_gaps` stays zero

If frames stay at zero, run:

```bash
python serial_test.py --port /dev/tty.usbmodemXXXX --baud 2000000 --raw-probe --raw-seconds 3
```

You should see `magic_hits > 0`. If not, the host is connected but the stream is not in the expected binary protocol.


## Data contract
Frame parser (`protocol.py`) expects:
- `magic = 0x574F4F44`
- `version = 1`
- `frame_bytes = 156`
- `int32 channels[32]`

Channel order (`session_meta.json` → `channel_order`):
- per-index map, e.g. `"0": "EEG1"`, …, `"31": "EMG16"`
- indices `0..15`: EEG1..EEG16; `16..31`: EMG1..EMG16

## Unit conversion

ADS1299 signed codes are converted to microvolts in one place (`host/_protocol.py` → `codes_to_uv()`), used for live plots and session recording:

`LSB_volts = (2 * Vref / gain) / (2^24 - 1)`

with defaults in `_protocol.py`:
- `Vref = 4.5V`
- EEG gain `12x`
- EMG gain `12x`

## Recording and events

- Press `Start Session Recording` to create a new session under `./recordings/`.
- Each session contains:
  - `eeg_frames.bin` (continuous frames packed as `<QQ32f` = `sample_index, mcu_time_us, ch1..ch32` in **µV**)
  - `session_meta.json` includes `channel_units`, `ads_calibration` (Vref, gains, LSB values, formula), and `eeg_record_format`
  - `events.csv` (timestamped markers tied to EEG sample indices)
  - `session_meta.json` (sampling/channel metadata and file map)
- Press event buttons (`A/B/C`) or keys `1..9` to append `manual_marker` events.
- Events are ignored when recording is off.

`events.csv` now stores both:
- quantized alignment (`sample_index_start`, `sample_index_end`)
- fractional alignment (`sample_index_start_float`, `sample_index_end_float`)

Fractional alignment is estimated from recent host receive times vs sample indices, then quantized to nearest integer sample for compatibility.

## Visualization controls

- `Single Channel View` toggles one-channel-at-a-time plotting with `◀ / ▶` channel navigation.
- `Show All Channels` toggles the dense full-channel layout.
- `Scale: Auto` keeps y-axis autoscaling.
- `Scale: Fixed` enables a fixed symmetric y-range (`±uV`) with quick EEG/EMG/Wide presets.

## EEG channel ordering note

If your daisy-chain wiring produces EEG halves reversed (board EEG9..16 appearing as GUI EEG1..8), host applies a channel swap fix before plotting and saving session data.

- Toggle constant in `app.py`: `EEG_SWAP_HALVES`
- `True` = swap EEG `1..8` with EEG `9..16` on host
- `False` = keep firmware packet order as-is

## Thinking workflow

- Enter a target word in `Thinking Trial`.
- Click `Start Thinking Trial`:
  - logs `thinking_trial_armed`
  - runs 3..2..1 countdown (`thinking_countdown_tick`)
  - logs `thinking_trial_start` and shows `THINK <word>` state
- Click `End Thinking Trial` to log `thinking_trial_end`.

All trial events include quantized and fractional sample alignment.

## Speech workflow

- Click `Start Speech Block` to begin microphone capture (`speech_block_start`).
- Click `Stop Speech Block` to end capture (`speech_block_end`).
- Audio is saved to `speech_block_<id>.wav` in the session folder.
- After stop, ASR runs in background and writes `speech_word` events with:
  - word text
  - confidence
  - sample-aligned start/end indices

Requirements for speech:
- `sounddevice` for audio capture
- `openai-whisper` for word-level timestamps

## Export for training

Generate aligned windows and event tables:

```bash
python export_session.py recordings/session_a3f8b2c1_2026-06-15_21-27-46
```

Outputs in the session directory:
- `dataset_events.npz` with fixed windows around event start samples
- `aligned_events.csv` as normalized event table for downstream pipelines

## Session validator (GUI)

Use `Validate last session` in the GUI (or `python validate_session.py <session_dir>`) to run integrity checks on the
last stopped session:
- required files exist (`session_meta.json`, `eeg_frames.bin`, `events.csv`)
- `participant_id` present (FAIL) and of the form `P###` (warn); donning / board / perturbation set (warn)
- EEG binary size is a whole number of `<QQ32f` records; >= 20 s; `sample_index` monotonic; dropped-sample % (FAIL > 1 %)
- flat or railed channels (FAIL on EMG, warn on EEG)
- `silent_speech_word` counts per word x condition; every word window inside the recording; `--min-per-word`
- rest / activity / speech blocks start-end pairing; speech audio present

Exit code 0 = PASS, 1 = FAIL, 2 = unreadable. `python validate_session.py --selftest` needs no hardware.

## Collection sessions (`china_collecting`)

The session form above the collection panel is filled in **before** `Start Session Recording` and lands in
`session_meta.json`:

- top-level `participant_id` (what `voyage_protocols.py` reads to group sessions by person — a session without it
  is a lost session; the GUI refuses to start without one, use `P000` for bench tests)
- `collection`: `donning_index`, `perturbation` (`none | shift_fwd_1cm | shift_back_1cm | tilt_10deg | other`),
  `fit` (`tight | ok | loose`), `board` (`analog | digital`), `rig_id`, `operator`, `firmware_hash`,
  `electrode_set`, `collection_run`, `notes`, plus `rail_at_start`, `active_words` / `word_weights`, the timing
  constants, and at stop `duration_s`, `rail_at_stop`
- `client`: `version`, `git_commit`, `git_branch`, host, `test_mode`, `serial_port`, and both EEG swap flags
  (`app.EEG_SWAP_HALVES`, `_protocol.SWAP_EEG_DAISY_HALVES`) so the montage in force at record time is on disk

`Update notes / fit` pushes edits mid-session (`POST /session/meta`, logs `session_info_updated`). The donning
counter advances by one after every stop.

Conditions. Every `silent_speech_word` event carries `payload_json.condition`:

- `silent` — default
- `overt` — the prompt fired inside an open speech block (`Start speech block (overt)`; mic on, Whisper runs at
  stop). Scrambles are allowed inside a speech block; ending the block mid-scramble is refused.
- `rest` / `activity:<label>` — no prompts fire here; these blocks label everything between their start and end
  events (`rest_block_start/end`, `activity_block_start/end`, labels `chew swallow talk head yawn_cough walk other`).
  `Rest 30 s` stops itself (`reason: timer`); activities run until `Stop block` or the next activity button
  (`reason: superseded`). Collection is refused while a rest/activity block is open.

Endpoints: `POST /recording/start {session_info}`, `POST /session/meta`, `GET /session/options`,
`GET /session/validate[?dir=]`, `POST /blocks/rest/start {duration_s}`, `POST /blocks/activity/start {activity}`,
`POST /blocks/stop`, `POST /trials/speech/start|stop`.

End of day: `python pack_sessions.py --recordings recordings --dest <export> --zip` copies every session with a
`participant_id` into `<export>/<participant_id>/<session_dir>/`, validates each, writes sha256 manifests and
`<participant_id>.zip`; `python pack_sessions.py --verify <export>/P001` re-hashes a copy. The laptop copy stays
until the manifest verifies on the receiving side.
