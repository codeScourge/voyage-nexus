#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "1.0.1"
RECORD_FORMAT = "<QQ32f"
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)
CHANNEL_COUNT = 32
EMG_CHANNELS = range(16, 32)
WORD_EVENT = "silent_speech_word"
PARTICIPANT_RE = re.compile(r"^P\d{3}$")
FLAT_STD_UV = 0.5
RAIL_FRACTION = 0.99
RAIL_PCT_FAIL = 10.0
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _protocol import EEG_LSB_UV, EMG_LSB_UV

    FULL_SCALE_UV = {"EEG": EEG_LSB_UV * 0x7FFFFF, "EMG": EMG_LSB_UV * 0x7FFFFF}
except Exception:
    _ref = 4.5 / (2**23 - 1) * 1e6
    FULL_SCALE_UV = {"EEG": _ref / 6.0 * 0x7FFFFF, "EMG": _ref * 0x7FFFFF}


def _group(ch: int) -> str:
    return "EEG" if ch < 16 else "EMG"


class Report:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.fails: list[str] = []
        self.warns: list[str] = []
        self.info: dict[str, Any] = {}

    def fail(self, msg: str) -> None:
        self.fails.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "verdict": "FAIL" if self.fails else "PASS",
            "fails": self.fails,
            "warns": self.warns,
            "info": self.info,
            "validator_version": VERSION,
        }


def _check_meta(rep: Report, meta: dict[str, Any]) -> None:
    pid = str(meta.get("participant_id") or "").strip()
    if not pid:
        rep.fail("participant_id missing in session_meta.json (a session without a participant is a lost session)")
    elif not PARTICIPANT_RE.match(pid):
        rep.warn(f"participant_id '{pid}' is not of the form P###")
    coll = meta.get("collection") or {}
    for key in ("donning_index", "board", "perturbation"):
        if coll.get(key) in (None, ""):
            rep.warn(f"collection.{key} not set")
    rep.info["participant_id"] = pid
    rep.info["donning_index"] = coll.get("donning_index")
    rep.info["perturbation"] = coll.get("perturbation")
    rep.info["board"] = coll.get("board")
    rep.info["client_version"] = (meta.get("client") or {}).get("version") or "pre-china_collecting"
    rep.info["git_commit"] = (meta.get("client") or {}).get("git_commit", "")
    fmt = meta.get("eeg_record_format", RECORD_FORMAT)
    if fmt != RECORD_FORMAT:
        rep.fail(f"eeg_record_format is {fmt!r}; this validator expects {RECORD_FORMAT!r}")
    rep.info["sample_rate_hz"] = meta.get("sample_rate_hz")


def _check_frames(rep: Report, bin_path: Path, fs: float, min_seconds: float, max_drop_pct: float) -> int:
    size = bin_path.stat().st_size
    n = size // RECORD_SIZE
    if size % RECORD_SIZE:
        rep.fail(f"eeg_frames.bin is {size} bytes, not a multiple of {RECORD_SIZE} (truncated write?)")
    rep.info["n_records"] = int(n)
    rep.info["duration_s"] = round(n / fs, 2) if fs else None
    if n == 0:
        rep.fail("eeg_frames.bin holds no records")
        return 0
    if fs and n / fs < min_seconds:
        rep.fail(f"only {n / fs:.1f} s of data (< {min_seconds:.0f} s)")
    dt = np.dtype([("idx", "<u8"), ("mcu_us", "<u8"), ("uv", "<f4", (CHANNEL_COUNT,))])
    data = np.memmap(bin_path, dtype=dt, mode="r", shape=(n,))
    idx = np.asarray(data["idx"], dtype=np.int64)
    d = np.diff(idx)
    if (d <= 0).any():
        rep.fail(f"sample_index not strictly increasing at {int((d <= 0).sum())} places")
    gaps = d[d > 1]
    dropped = int((gaps - 1).sum()) if gaps.size else 0
    span = int(idx[-1] - idx[0] + 1) if n else 1
    drop_pct = 100.0 * dropped / max(span, 1)
    rep.info["dropped_samples"] = dropped
    rep.info["dropped_pct"] = round(drop_pct, 3)
    rep.info["gap_events"] = int(gaps.size)
    if drop_pct > max_drop_pct:
        rep.fail(f"{drop_pct:.2f} % of samples dropped (> {max_drop_pct} %) in {gaps.size} gaps")
    elif dropped:
        rep.warn(f"{dropped} samples dropped in {gaps.size} gaps ({drop_pct:.3f} %)")
    rep.info["sample_index_first"] = int(idx[0])
    rep.info["sample_index_last"] = int(idx[-1])

    step = max(1, n // 120_000)
    uv = np.asarray(data["uv"][::step], dtype=np.float32)
    std = uv.std(axis=0)
    flat, railed = [], []
    for ch in range(CHANNEL_COUNT):
        fs_uv = FULL_SCALE_UV[_group(ch)]
        rail_pct = 100.0 * float((np.abs(uv[:, ch]) > RAIL_FRACTION * fs_uv).mean())
        if std[ch] < FLAT_STD_UV:
            flat.append(ch)
        if rail_pct > RAIL_PCT_FAIL:
            railed.append((ch, rail_pct))
    rep.info["flat_channels"] = [f"{_group(c)}{(c % 16) + 1}" for c in flat]
    rep.info["railed_channels"] = [f"{_group(c)}{(c % 16) + 1}:{p:.0f}%" for c, p in railed]
    rep.info["channel_std_uv"] = [round(float(s), 2) for s in std]
    for c in flat:
        (rep.fail if c in EMG_CHANNELS else rep.warn)(f"{_group(c)}{(c % 16) + 1} is flat (std {std[c]:.2f} uV)")
    for c, p in railed:
        (rep.fail if c in EMG_CHANNELS else rep.warn)(f"{_group(c)}{(c % 16) + 1} railed {p:.0f} % of the time")
    return int(n)


def _check_events(rep: Report, csv_path: Path, first_idx: int, last_idx: int, min_per_word: int,
                  session_dir: Path, win_samples: int = 1600) -> None:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rep.info["n_events"] = len(rows)
    types = Counter(r.get("event_type", "") for r in rows)
    rep.info["event_types"] = dict(types)
    if "session_recording_started" not in types:
        rep.warn("no session_recording_started event")
    if "session_recording_stopped" not in types:
        rep.warn("no session_recording_stopped event (recorder not stopped cleanly?)")

    words: Counter[str] = Counter()
    by_cond: dict[str, Counter[str]] = defaultdict(Counter)
    blocks: set[str] = set()
    outside = 0
    for r in rows:
        if r.get("event_type") != WORD_EVENT:
            continue
        w = (r.get("label_text") or "").strip().lower()
        try:
            payload = json.loads(r.get("payload_json") or "{}")
        except json.JSONDecodeError:
            payload = {}
        cond = str(payload.get("condition") or "silent")
        words[w] += 1
        by_cond[cond][w] += 1
        if payload.get("collection_block_id"):
            blocks.add(str(payload["collection_block_id"]))
        try:
            s = int(float(r["sample_index_start"]))
        except (KeyError, ValueError, TypeError):
            outside += 1
            continue
        try:
            e = int(float(r.get("sample_index_end") or 0)) or s + win_samples
        except (ValueError, TypeError):
            e = s + win_samples
        if s < first_idx or e > last_idx:
            outside += 1
    rep.info["word_counts"] = dict(words)
    rep.info["word_counts_by_condition"] = {c: dict(v) for c, v in by_cond.items()}
    rep.info["collection_blocks"] = len(blocks)
    if outside:
        rep.fail(f"{outside} word windows fall outside the recorded sample range")
    if not words:
        rep.warn("no silent_speech_word events (rest / activity-only session?)")
    silent = by_cond.get("silent", Counter())
    for w, c in silent.items():
        if min_per_word and c < min_per_word:
            rep.fail(f"only {c} silent trials of '{w}' (< {min_per_word})")

    for kind in ("speech_block", "rest_block", "activity_block"):
        n_start, n_end = types.get(f"{kind}_start", 0), types.get(f"{kind}_end", 0)
        if n_start != n_end:
            rep.fail(f"{kind}: {n_start} start / {n_end} end events")
    rep.info["activity_labels"] = sorted(
        {(r.get("label_text") or "") for r in rows if r.get("event_type") == "activity_block_start"}
    )
    rep.info["rest_blocks"] = types.get("rest_block_start", 0)
    for r in rows:
        if r.get("event_type") == "speech_block_end":
            try:
                wav = json.loads(r.get("payload_json") or "{}").get("audio_file", "")
            except json.JSONDecodeError:
                wav = ""
            if wav and not (session_dir / wav).exists():
                rep.warn(f"speech block audio {wav} missing")


def validate_session_dir(session_dir: Path | str, *, min_seconds: float = 20.0, max_drop_pct: float = 1.0,
                         min_per_word: int = 0) -> dict[str, Any]:
    session_dir = Path(session_dir)
    rep = Report(session_dir)
    meta_path, bin_path, csv_path = (session_dir / n for n in ("session_meta.json", "eeg_frames.bin", "events.csv"))
    for p in (meta_path, bin_path, csv_path):
        if not p.exists():
            rep.fail(f"missing {p.name}")
    if rep.fails:
        return rep.as_dict()
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rep.fail(f"session_meta.json does not parse: {exc}")
        return rep.as_dict()
    _check_meta(rep, meta)
    fs = float(meta.get("sample_rate_hz") or 1000.0)
    n = _check_frames(rep, bin_path, fs, min_seconds, max_drop_pct)
    first_idx = rep.info.get("sample_index_first", 0)
    last_idx = rep.info.get("sample_index_last", n)
    win_samples = int(round(1.6 * fs))
    try:
        _check_events(rep, csv_path, first_idx, last_idx, min_per_word, session_dir, win_samples)
    except Exception as exc:
        rep.fail(f"events.csv could not be read: {exc}")
    return rep.as_dict()


def format_report(res: dict[str, Any]) -> str:
    info = res["info"]
    lines = [f"{res['verdict']}  {res['session_dir']}"]
    lines.append(
        f"  participant {info.get('participant_id') or '-'} donning {info.get('donning_index') or '-'} "
        f"perturbation {info.get('perturbation') or '-'} board {info.get('board') or '-'} "
        f"client {info.get('client_version', '-')}"
    )
    if "n_records" in info:
        lines.append(
            f"  {info['n_records']} records = {info.get('duration_s')} s; dropped {info.get('dropped_samples', 0)} "
            f"({info.get('dropped_pct', 0)} %); flat {info.get('flat_channels') or '-'}; railed {info.get('railed_channels') or '-'}"
        )
    if "word_counts_by_condition" in info:
        for cond, counts in sorted(info["word_counts_by_condition"].items()):
            lines.append(f"  {cond:>10}: " + ", ".join(f"{w} {c}" for w, c in sorted(counts.items())))
        lines.append(f"  blocks {info.get('collection_blocks', 0)}, rest {info.get('rest_blocks', 0)}, "
                     f"activities {info.get('activity_labels') or '-'}")
    for f in res["fails"]:
        lines.append(f"  FAIL  {f}")
    for w in res["warns"]:
        lines.append(f"  warn  {w}")
    return "\n".join(lines)


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "2026-09-24_10-00-00_session_selftest"
        d.mkdir()
        fs = 1000
        n = 40 * fs
        rng = np.random.default_rng(0)
        uv = (rng.standard_normal((n, CHANNEL_COUNT)) * 20).astype(np.float32)
        uv[:, 20] = 0.0
        uv[: n // 5, 3] = FULL_SCALE_UV["EEG"] * 0.999
        idx = np.arange(n, dtype=np.uint64)
        idx[n // 2:] += 5
        recs = np.zeros(n, dtype=[("idx", "<u8"), ("mcu_us", "<u8"), ("uv", "<f4", (CHANNEL_COUNT,))])
        recs["idx"], recs["mcu_us"], recs["uv"] = idx, idx * 1000, uv
        (d / "eeg_frames.bin").write_bytes(recs.tobytes())
        (d / "session_meta.json").write_text(json.dumps({
            "participant_id": "P001", "sample_rate_hz": fs, "eeg_record_format": RECORD_FORMAT,
            "collection": {"donning_index": 1, "board": "analog", "perturbation": "none"},
            "client": {"version": "selftest"}}))
        with (d / "events.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["event_id", "event_type", "label_text", "host_time_iso", "host_time_ns", "sample_index_start",
                        "sample_index_start_float", "sample_index_end", "sample_index_end_float", "confidence",
                        "alignment_method", "payload_json"])
            w.writerow(["a", "session_recording_started", "", "", "", 0, "", "", "", "", "", "{}"])
            for i, word in enumerate(["bullshit", "gogogo", "highlight"] * 4):
                s = 2000 + i * 2500
                cond = "overt" if i >= 9 else "silent"
                w.writerow(["e%d" % i, WORD_EVENT, word, "", "", s, "", s + 1600, "", "", "",
                            json.dumps({"collection_block_id": "b1", "condition": cond})])
            w.writerow(["r", "rest_block_start", "rest", "", "", 100, "", "", "", "", "", "{}"])
            w.writerow(["r2", "rest_block_end", "rest", "", "", 100, "", 30100, "", "", "", "{}"])
            w.writerow(["z", "session_recording_stopped", "", "", "", n - 1, "", "", "", "", "", "{}"])
        res = validate_session_dir(d, min_per_word=3)
        print(format_report(res))
        assert res["verdict"] == "FAIL", res
        assert any("EMG5 is flat" in f for f in res["fails"]), res["fails"]
        assert any("EEG4 railed" in w for w in res["warns"]), res["warns"]
        assert res["info"]["dropped_samples"] == 5, res["info"]
        assert res["info"]["word_counts_by_condition"]["silent"] == {"bullshit": 3, "gogogo": 3, "highlight": 3}
        assert res["info"]["word_counts_by_condition"]["overt"] == {"bullshit": 1, "gogogo": 1, "highlight": 1}
        uv[:, 20] = rng.standard_normal(n) * 20
        recs["uv"] = uv
        (d / "eeg_frames.bin").write_bytes(recs.tobytes())
        res = validate_session_dir(d, min_per_word=3)
        assert res["verdict"] == "PASS", res["fails"]
        print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="PASS / FAIL check of one recorded session (session_meta.json, eeg_frames.bin, events.csv): "
        "participant_id, <QQ32f> record integrity, dropped samples, flat / railed channels, per-word x condition "
        "counts, word windows inside the recording, block start/end pairing. Exit 0 PASS, 1 FAIL, 2 unreadable."
    )
    ap.add_argument("path", nargs="?", help="session dir, or a recordings root with --all")
    ap.add_argument("--all", action="store_true", help="validate every session dir under PATH")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-seconds", type=float, default=20.0)
    ap.add_argument("--max-drop-pct", type=float, default=1.0)
    ap.add_argument("--min-per-word", type=int, default=0, help="FAIL below this many silent trials per word")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.path:
        ap.error("path is required")
    root = Path(args.path)
    if args.all:
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "session_meta.json").exists())
    else:
        dirs = [root]
    results = [validate_session_dir(d, min_seconds=args.min_seconds, max_drop_pct=args.max_drop_pct,
                                    min_per_word=args.min_per_word) for d in dirs]
    if args.json:
        print(json.dumps(results if args.all else results[0], indent=2))
    else:
        for r in results:
            print(format_report(r))
    return 1 if any(r["verdict"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
