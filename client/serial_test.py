from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import numpy as np
import serial
from serial.tools import list_ports

from _protocol import FRAME_SIZE, FrameParser, SampleFrame

DEFAULT_BAUD = 2_000_000


def pick_default_port() -> Optional[str]:
    ports = list(list_ports.comports())
    if not ports:
        return None
    usb_first = sorted(ports, key=lambda p: ("usb" not in p.description.lower(), p.device))
    return usb_first[0].device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and verify WOODSIDE ADS1299 sample frames over serial"
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port path")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between status logs",
    )
    parser.add_argument(
        "--preview-interval",
        type=float,
        default=2.0,
        help="Seconds between channel preview logs",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Auto-stop after N seconds (0 = run forever)",
    )
    parser.add_argument(
        "--raw-probe",
        action="store_true",
        help="Print raw serial bytes and magic-hit stats (no frame parsing)",
    )
    parser.add_argument(
        "--raw-seconds",
        type=float,
        default=3.0,
        help="Duration for --raw-probe capture",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected serial ports and exit",
    )
    parser.add_argument(
        "--quality-interval",
        type=float,
        default=0.0,
        help="If >0, print per-channel rail/duplication diagnostics every N seconds",
    )
    parser.add_argument(
        "--quality-window-frames",
        type=int,
        default=1000,
        help="Number of recent frames to use for quality diagnostics",
    )
    return parser.parse_args()


def frame_preview(frame: SampleFrame) -> str:
    eeg = frame.channels_i32[:16]
    emg = frame.channels_i32[16:]
    eeg_rms = float(math.sqrt((eeg.astype("float64") ** 2).mean()))
    emg_rms = float(math.sqrt((emg.astype("float64") ** 2).mean()))
    return (
        f"idx={frame.sample_index} "
        f"EEG[1]={int(eeg[0])} EEG[16]={int(eeg[15])} "
        f"EMG[1]={int(emg[0])} EMG[16]={int(emg[15])} "
        f"rms_codes(eeg={eeg_rms:.1f}, emg={emg_rms:.1f})"
    )


def list_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports detected.")
        return
    for p in sorted(ports, key=lambda x: x.device):
        print(f"{p.device} | {p.description} | hwid={p.hwid}")


def run_raw_probe(port: str, baud: int, capture_seconds: float) -> int:
    magic = bytes.fromhex("44 4F 4F 57")  # little-endian 0x574F4F44
    total = 0
    chunks = 0
    magic_hits = 0
    first_chunk = b""
    capture = bytearray()
    t0 = time.monotonic()

    print(f"raw-probe opening {port} @ {baud} for {capture_seconds:.1f}s")
    try:
        with serial.Serial(port, baud, timeout=0.05) as ser:
            while (time.monotonic() - t0) < capture_seconds:
                data = ser.read(2048)
                if not data:
                    continue
                chunks += 1
                total += len(data)
                if not first_chunk:
                    first_chunk = data[:64]
                capture.extend(data)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"serial error: {exc}")
        return 2

    magic_hits = capture.count(magic)
    sample = bytes(capture[:256])
    ascii_preview = "".join(chr(b) if 32 <= b <= 126 else "." for b in sample[:120])

    print(f"raw-probe bytes={total} chunks={chunks} magic_hits={magic_hits}")
    if first_chunk:
        print("raw-probe first64_hex=" + first_chunk.hex(" "))
    if sample:
        print("raw-probe first256_hex=" + sample.hex(" "))
        print("raw-probe ascii_preview=" + ascii_preview)
    else:
        print("raw-probe no bytes captured")

    if total == 0:
        print("raw-probe result: no serial data at all")
        return 3
    if magic_hits == 0:
        print("raw-probe result: stream active but frame magic never appears")
        return 4
    print("raw-probe result: frame magic present in stream")
    return 0


def print_quality_report(history: np.ndarray) -> None:
    if history.size == 0:
        print("quality waiting: no samples")
        return

    # history shape: (frames, 32)
    rail_pos = (history >= 0x7FFF00).mean(axis=0) * 100.0
    rail_neg = (history <= -0x7FFF00).mean(axis=0) * 100.0
    stds = history.std(axis=0)

    # Find likely duplicates (nearly identical streams).
    duplicate_pairs: list[str] = []
    max_channels_to_check = history.shape[1]
    for i in range(max_channels_to_check):
        for j in range(i + 1, max_channels_to_check):
            if np.array_equal(history[:, i], history[:, j]):
                duplicate_pairs.append(f"{i+1}-{j+1}(exact)")
                continue
            # Correlation only meaningful if both have variance.
            if stds[i] < 1.0 or stds[j] < 1.0:
                continue
            corr = float(np.corrcoef(history[:, i], history[:, j])[0, 1])
            if corr > 0.999:
                duplicate_pairs.append(f"{i+1}-{j+1}(corr={corr:.4f})")

    def top_rail(prefix: str, start: int, end: int) -> str:
        idx = np.argmax(np.maximum(rail_pos[start:end], rail_neg[start:end])) + start
        pct = max(rail_pos[idx], rail_neg[idx])
        return f"{prefix}{idx - start + 1} rail={pct:.1f}% std={stds[idx]:.1f}"

    print(
        "quality "
        f"EEG_top={top_rail('EEG', 0, 16)} "
        f"EMG_top={top_rail('EMG', 16, 32)} "
        f"dup_pairs={','.join(duplicate_pairs[:8]) if duplicate_pairs else 'none'}"
    )


def main() -> int:
    args = parse_args()
    if args.list_ports:
        list_serial_ports()
        return 0

    port = args.port or pick_default_port()
    if not port:
        print("No serial ports found. Pass --port explicitly.")
        return 1

    if args.raw_probe:
        return run_raw_probe(port=port, baud=args.baud, capture_seconds=args.raw_seconds)

    parser = FrameParser()
    total_frames = 0
    dropped_index_gaps = 0
    last_index: Optional[int] = None
    last_frame: Optional[SampleFrame] = None

    start_t = time.monotonic()
    last_status_t = start_t
    last_preview_t = start_t
    last_quality_t = start_t
    first_frame_t: Optional[float] = None
    history = np.zeros((args.quality_window_frames, 32), dtype=np.int32)
    history_count = 0
    history_idx = 0

    print(f"Opening {port} @ {args.baud} baud")
    print(f"Expected frame size: {FRAME_SIZE} bytes")
    print("Press Ctrl+C to stop")

    try:
        with serial.Serial(port, args.baud, timeout=0.05) as ser:
            while True:
                now = time.monotonic()
                if args.max_seconds > 0 and (now - start_t) >= args.max_seconds:
                    break

                chunk = ser.read(max(FRAME_SIZE * 8, 256))
                if chunk:
                    frames = parser.feed(chunk)
                    for frame in frames:
                        if first_frame_t is None:
                            first_frame_t = time.monotonic()
                        total_frames += 1
                        if last_index is not None and frame.sample_index != last_index + 1:
                            if frame.sample_index > last_index + 1:
                                dropped_index_gaps += frame.sample_index - (last_index + 1)
                        last_index = frame.sample_index
                        last_frame = frame
                        if args.quality_interval > 0:
                            history[history_idx, :] = frame.channels_i32
                            history_idx = (history_idx + 1) % args.quality_window_frames
                            history_count = min(history_count + 1, args.quality_window_frames)

                now = time.monotonic()
                if now - last_status_t >= args.status_interval:
                    elapsed = now - (first_frame_t or start_t)
                    rate = (total_frames / elapsed) if elapsed > 0 else 0.0
                    print(
                        "status "
                        f"frames={total_frames} rate={rate:.1f}/s "
                        f"desync={parser.desync_count} bad_crc={parser.bad_checksum_count} "
                        f"index_gaps={dropped_index_gaps}"
                    )
                    if first_frame_t is None:
                        print("waiting: no valid frames yet (check wiring, baud, frame format)")
                    last_status_t = now

                if last_frame is not None and (now - last_preview_t) >= args.preview_interval:
                    print("preview " + frame_preview(last_frame))
                    last_preview_t = now

                if args.quality_interval > 0 and (now - last_quality_t) >= args.quality_interval:
                    if history_count == 0:
                        print("quality waiting: no samples")
                    elif history_count < args.quality_window_frames:
                        view = history[:history_count, :]
                        print_quality_report(view)
                    else:
                        view = np.concatenate(
                            (history[history_idx:, :], history[:history_idx, :]),
                            axis=0,
                        )
                        print_quality_report(view)
                    last_quality_t = now

    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pylint: disable=broad-except
        print(f"serial error: {exc}")
        return 2

    print("done")
    if total_frames == 0:
        print("result: no valid frames parsed")
        return 3

    print(
        "result: valid frames parsed "
        f"(frames={total_frames}, desync={parser.desync_count}, bad_crc={parser.bad_checksum_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
