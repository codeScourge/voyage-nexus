#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "1.0.0"
SESSION_FILES_REQUIRED = ("session_meta.json", "eeg_frames.bin", "events.csv")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def session_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "session_meta.json").exists())


def participant_of(session_dir: Path, participant_map: dict[str, str]) -> str:
    try:
        meta = json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(meta.get("participant_id") or participant_map.get(session_dir.name, "")).strip().upper()


def pack_session(session_dir: Path, dest_participant: Path, *, validate: bool) -> dict[str, Any]:
    target = dest_participant / session_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(session_dir, target)
    files = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(target)).replace("\\", "/"), "bytes": p.stat().st_size,
                          "sha256": sha256_of(p)})
    entry: dict[str, Any] = {"session_dir": session_dir.name, "files": files,
                             "missing_required": [f for f in SESSION_FILES_REQUIRED if not (target / f).exists()]}
    if validate:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from validate_session import validate_session_dir

            res = validate_session_dir(target)
            entry["validation"] = {"verdict": res["verdict"], "fails": res["fails"], "warns": res["warns"]}
        except Exception as exc:
            entry["validation"] = {"verdict": "ERROR", "fails": [str(exc)], "warns": []}
    (target / "manifest.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return entry


def pack(recordings: Path, dest: Path, *, participant: str | None, participant_map: dict[str, str],
         validate: bool, make_zip: bool) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    by_participant: dict[str, list[Path]] = {}
    skipped: list[str] = []
    for sd in session_dirs(recordings):
        pid = participant_of(sd, participant_map)
        if not pid:
            skipped.append(sd.name)
            continue
        if participant and pid != participant.upper():
            continue
        by_participant.setdefault(pid, []).append(sd)
    summary: dict[str, Any] = {"packed_at_iso": datetime.now(timezone.utc).isoformat(), "packer_version": VERSION,
                               "recordings": str(recordings), "dest": str(dest), "participants": {},
                               "skipped_no_participant": skipped}
    for pid, dirs in sorted(by_participant.items()):
        pdir = dest / pid
        pdir.mkdir(parents=True, exist_ok=True)
        entries = [pack_session(sd, pdir, validate=validate) for sd in dirs]
        manifest = {"participant_id": pid, "packed_at_iso": summary["packed_at_iso"], "sessions": entries,
                    "packer_version": VERSION}
        (pdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        info: dict[str, Any] = {"sessions": len(entries),
                                "bytes": sum(f["bytes"] for e in entries for f in e["files"]),
                                "verdicts": {e["session_dir"]: e.get("validation", {}).get("verdict", "-") for e in entries}}
        if make_zip:
            zpath = dest / f"{pid}.zip"
            with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for p in sorted(pdir.rglob("*")):
                    if p.is_file():
                        zf.write(p, arcname=str(Path(pid) / p.relative_to(pdir)))
            info["zip"] = str(zpath)
            info["zip_sha256"] = sha256_of(zpath)
        summary["participants"][pid] = info
    (dest / "pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def verify(participant_dir: Path) -> dict[str, Any]:
    manifest = json.loads((participant_dir / "manifest.json").read_text(encoding="utf-8"))
    bad: list[str] = []
    n = 0
    for entry in manifest["sessions"]:
        sdir = participant_dir / entry["session_dir"]
        for f in entry["files"]:
            n += 1
            p = sdir / f["path"]
            if not p.exists():
                bad.append(f"{entry['session_dir']}/{f['path']}: missing")
            elif p.stat().st_size != f["bytes"] or sha256_of(p) != f["sha256"]:
                bad.append(f"{entry['session_dir']}/{f['path']}: hash/size mismatch")
    return {"participant_id": manifest.get("participant_id"), "files_checked": n, "bad": bad,
            "verdict": "OK" if not bad else "MISMATCH"}


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec = root / "recordings"
        for i, pid in enumerate(["P001", "P001", "", "P002"]):
            d = rec / f"2026-09-24_10-0{i}-00_session_{i:08x}"
            d.mkdir(parents=True)
            (d / "session_meta.json").write_text(json.dumps({"participant_id": pid, "sample_rate_hz": 1000}))
            (d / "eeg_frames.bin").write_bytes(b"\0" * 144 * 10)
            (d / "events.csv").write_text("event_id,event_type\n")
        summary = pack(rec, root / "export", participant=None, participant_map={}, validate=False, make_zip=True)
        assert set(summary["participants"]) == {"P001", "P002"}, summary
        assert summary["participants"]["P001"]["sessions"] == 2
        assert len(summary["skipped_no_participant"]) == 1
        assert (root / "export" / "P001.zip").exists()
        v = verify(root / "export" / "P001")
        assert v["verdict"] == "OK" and v["files_checked"] == 6, v
        (root / "export" / "P001" / summary["participants"]["P001"]["verdicts"].__iter__().__next__() / "events.csv").write_text("tampered\n")
        assert verify(root / "export" / "P001")["verdict"] == "MISMATCH"
        print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Copy each session with a participant_id into <dest>/<participant_id>/<session_dir>/, validate it, "
        "write sha256 manifests (and a zip with --zip); --verify re-hashes a packed participant folder. Copy only, no network."
    )
    ap.add_argument("--recordings", help="recordings root (the client's default is <repo>/recordings)")
    ap.add_argument("--dest", help="export root; <dest>/<participant_id>/... is created")
    ap.add_argument("--participant", help="only this participant id")
    ap.add_argument("--participant-map", help="CSV 'session_dir,participant_id' for sessions whose meta lacks it")
    ap.add_argument("--no-validate", action="store_true", help="skip validate_session on each copy")
    ap.add_argument("--zip", action="store_true", help="also write <dest>/<participant_id>.zip")
    ap.add_argument("--verify", metavar="PARTICIPANT_DIR", help="re-hash a packed participant folder")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.verify:
        res = verify(Path(args.verify))
        print(json.dumps(res, indent=2))
        return 0 if res["verdict"] == "OK" else 1
    if not (args.recordings and args.dest):
        ap.error("--recordings and --dest are required (or --verify / --selftest)")
    pmap: dict[str, str] = {}
    if args.participant_map:
        with open(args.participant_map, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pmap[row["session_dir"]] = row["participant_id"]
    summary = pack(Path(args.recordings), Path(args.dest), participant=args.participant, participant_map=pmap,
                   validate=not args.no_validate, make_zip=args.zip)
    for pid, info in summary["participants"].items():
        print(f"{pid}: {info['sessions']} sessions, {info['bytes'] / 1e6:.1f} MB, verdicts {info['verdicts']}"
              + (f", zip {info['zip']}" if "zip" in info else ""))
    if summary["skipped_no_participant"]:
        print("skipped (no participant_id):", *summary["skipped_no_participant"], sep="\n  ")
    print(f"summary -> {Path(args.dest) / 'pack_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
