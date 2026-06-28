#!/usr/bin/env python3
"""
check_regimes.py — detect hardware-regime boundaries across the LOSO sessions.

Motivation
----------
The EA win (loso_align.py) rescued the three catastrophic baseline folds
(f1 0.530, f2 0.548, f15 0.489). Before calling the per-session problem closed,
we must rule out that those folds are a HARDWARE artifact (a gain / Vref change
mid-collection) rather than genuine re-donning variance. If a regime boundary
runs through the fold set, part of the "alignment win" is really alignment
papering over an acquisition-config change that belongs in firmware/protocol.

This reads each session's session_meta.json, groups sessions by their
acquisition config, flags boundaries, and (if given a loso_align log) cross-
references regime membership against the baseline per-fold balanced accuracy to
answer: are the bad folds confined to a minority regime, or spread across one
uniform regime?

LIMITATION: session_meta.json records gains, Vref, sample rate, channel
order/count, units and record format. It does NOT record DRL state. A DRL
on/off boundary therefore CANNOT be detected here — that gap is itself reported.

Usage
-----
    python check_regimes.py --recordings ./recordings
    python check_regimes.py --recordings ./recordings --cache loso_cache_global.npz \
        --loso-log loso_align_global.log
    python check_regimes.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SESSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_session_\w+")

# Fields that, if they differ between sessions, change how the stored signal must
# be interpreted. Split into "hard" (would corrupt windowing / channel indexing —
# should NEVER vary) and "soft" (amplitude/quantization regime — may vary and is
# the prime suspect for the bad folds).
HARD_FIELDS = ["sample_rate_hz", "channel_count", "channel_order_hash",
               "channel_units", "eeg_record_format"]
SOFT_FIELDS = ["eeg_gain", "emg_gain", "ads_vref_volts"]


def _round(v, nd=6):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


def load_meta(meta_path: Path) -> dict:
    """Extract the regime-defining fields from one session_meta.json."""
    d = json.loads(Path(meta_path).read_text())
    cal = d.get("ads_calibration", {}) or {}
    order = d.get("channel_order", {}) or {}
    # stable hash of channel order (values in integer-key order)
    order_items = sorted(((int(k), v) for k, v in order.items()), key=lambda kv: kv[0])
    order_hash = json.dumps([v for _, v in order_items])
    return {
        "created_at_iso": d.get("created_at_iso"),
        "sample_rate_hz": d.get("sample_rate_hz"),
        "channel_count": d.get("channel_count"),
        "channel_units": d.get("channel_units"),
        "eeg_record_format": d.get("eeg_record_format"),
        "channel_order_hash": order_hash,
        "eeg_gain": _round(cal.get("eeg_gain")),
        "emg_gain": _round(cal.get("emg_gain")),
        "ads_vref_volts": _round(cal.get("ads_vref_volts")),
        "eeg_lsb_uv": _round(cal.get("eeg_lsb_uv")),
        "emg_lsb_uv": _round(cal.get("emg_lsb_uv")),
        "has_drl_field": any("drl" in k.lower() for k in d.keys())
                         or any("drl" in k.lower() for k in cal.keys()),
    }


def regime_key(meta: dict) -> tuple:
    return tuple(meta.get(f) for f in (HARD_FIELDS + SOFT_FIELDS))


def parse_loso_log(log_path: Path) -> dict:
    """Pull baseline per-fold balanced_accuracy per session from the per-fold
    table in a loso_align log. Returns {session_name: baseline_bal_acc}."""
    out = {}
    in_table = False
    for line in Path(log_path).read_text().splitlines():
        if "per-fold balanced_accuracy" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("===") or not line.strip():
                if out:                      # table ended
                    break
                continue
            m = SESSION_RE.search(line)
            if not m:
                continue
            nums = re.findall(r"-?\d+\.\d+", line[m.end():])
            if nums:
                out[m.group(0)] = float(nums[0])   # first col = baseline
    return out


def discover_sessions(recordings: Path, cache: Path | None):
    """Return ordered list of session names. Prefer the cache's session order so
    indices line up with LOSO folds; else sorted dirs under recordings."""
    if cache is not None and Path(cache).exists():
        import numpy as np
        z = np.load(cache, allow_pickle=True)
        return [str(s) for s in z["session_names"]]
    names = sorted(p.name for p in Path(recordings).iterdir()
                   if p.is_dir() and SESSION_RE.fullmatch(p.name))
    return names


def analyze(sessions, metas, bal):
    """Group into regimes and assemble the report rows."""
    keys = {}
    for s in sessions:
        if metas.get(s) is None:
            continue
        k = regime_key(metas[s])
        keys.setdefault(k, []).append(s)
    # assign a stable regime id by first appearance
    order = []
    for s in sessions:
        if metas.get(s) is None:
            continue
        k = regime_key(metas[s])
        if k not in order:
            order.append(k)
    regime_id = {k: i for i, k in enumerate(order)}
    return keys, regime_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=Path("recordings"))
    ap.add_argument("--cache", type=Path, default=Path("loso_cache_global.npz"),
                    help="use this cache's session order (aligns with fold index)")
    ap.add_argument("--loso-log", type=Path, default=None,
                    help="a loso_align log; cross-ref regimes vs baseline bal_acc")
    ap.add_argument("--bad-threshold", type=float, default=0.70,
                    help="baseline bal_acc below this = 'catastrophic' fold")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cache = args.cache if (args.cache and Path(args.cache).exists()) else None
    sessions = discover_sessions(args.recordings, cache)
    if not sessions:
        print(f"no session dirs found under {args.recordings}")
        return

    metas, missing = {}, []
    for s in sessions:
        mp = Path(args.recordings) / s / "session_meta.json"
        if mp.exists():
            metas[s] = load_meta(mp)
        else:
            metas[s] = None
            missing.append(s)

    bal = parse_loso_log(args.loso_log) if args.loso_log else {}
    keys, regime_id = analyze(sessions, metas, bal)

    # ---- per-session table ----
    print(f"\n{len(sessions)} sessions | {len(keys)} distinct regime(s)"
          + (f" | {len(missing)} missing meta" if missing else ""))
    drl_logged = any(m and m.get("has_drl_field") for m in metas.values())
    print(f"DRL state recorded in meta: {'yes' if drl_logged else 'NO (cannot be checked)'}")

    hdr = (f"\n{'#':>2} {'session':<40} {'eeg_g':>6} {'emg_g':>6} {'vref':>5} "
           f"{'srate':>6} {'nch':>4} {'reg':>4}")
    if bal:
        hdr += f" {'base_bal':>9}"
    print(hdr)
    for i, s in enumerate(sessions):
        m = metas[s]
        if m is None:
            print(f"{i+1:>2} {s[:40]:<40} {'--- session_meta.json MISSING ---'}")
            continue
        rid = regime_id[regime_key(m)]
        row = (f"{i+1:>2} {s[:40]:<40} {str(m['eeg_gain']):>6} {str(m['emg_gain']):>6} "
               f"{str(m['ads_vref_volts']):>5} {str(m['sample_rate_hz']):>6} "
               f"{str(m['channel_count']):>4} {rid:>4}")
        if bal:
            b = bal.get(s)
            flag = "  <-- BAD" if (b is not None and b < args.bad_threshold) else ""
            row += f" {b:>9.3f}{flag}" if b is not None else f" {'n/a':>9}"
        print(row)

    # ---- hard-field sanity ----
    hard_vals = {f: {(metas[s][f]) for s in sessions if metas[s]} for f in HARD_FIELDS}
    hard_bad = {f: v for f, v in hard_vals.items() if len(v) > 1}
    print("\n--- structural fields (must be constant) ---")
    if hard_bad:
        for f, v in hard_bad.items():
            print(f"  !! {f} VARIES across sessions: {v}")
        print("  This corrupts channel indexing / windowing. The cache and all"
              "\n  LOSO results above are SUSPECT until this is reconciled.")
    else:
        print("  OK: sample_rate, channel_count/order, units, record format all constant.")

    # ---- regime composition ----
    print("\n--- regimes (by acquisition config) ---")
    for k, members in sorted(keys.items(), key=lambda kv: regime_id[kv[0]]):
        rid = regime_id[k]
        eeg_g, emg_g, vref = k[len(HARD_FIELDS)], k[len(HARD_FIELDS)+1], k[len(HARD_FIELDS)+2]
        line = f"  regime {rid}: eeg_gain={eeg_g} emg_gain={emg_g} vref={vref} | {len(members)} session(s)"
        if bal:
            bs = [bal[s] for s in members if s in bal]
            if bs:
                line += f" | base_bal mean={sum(bs)/len(bs):.3f} min={min(bs):.3f}"
        print(line)

    # ---- the verdict the user actually wants ----
    print("\n=== verdict ===")
    if hard_bad:
        print("Structural fields vary -> fix that FIRST; regime question is moot until then.")
        return
    if len(keys) == 1:
        print("All sessions share ONE acquisition regime (identical gains/Vref/format).")
        print("=> The catastrophic baseline folds are NOT explained by any recorded")
        print("   hardware-config change. They are re-donning / physiology variance.")
        print("   The EA win is correcting genuine donning shift, not a config artifact.")
        if not drl_logged:
            print("   CAVEAT: DRL state is not logged, so a DRL boundary cannot be excluded")
            print("   from these files alone. Start logging DRL state per session to close this.")
        if not bal:
            print("   (Pass --loso-log to confirm which folds are the bad ones.)")
    else:
        print(f"{len(keys)} distinct regimes detected -> a config boundary exists.")
        if bal:
            bad = {s for s, b in bal.items() if b < args.bad_threshold}
            bad_regimes = {regime_id[regime_key(metas[s])] for s in bad if metas.get(s)}
            per_regime_bad = {regime_id[regime_key(metas[s])]: [] for s in sessions if metas.get(s)}
            for s in sessions:
                if metas.get(s) and s in bal:
                    per_regime_bad[regime_id[regime_key(metas[s])]].append(bal[s])
            print(f"   catastrophic folds (<{args.bad_threshold}) live in regime(s): {sorted(bad_regimes)}")
            minority = [r for r, ss in per_regime_bad.items() if len(ss) <= 3]
            if bad_regimes and bad_regimes.issubset(set(minority)):
                print("   => Bad folds CONCENTRATE in a minority regime: LIKELY A HARDWARE")
                print("      ARTIFACT. Investigate that regime's gain/Vref/DRL before trusting")
                print("      the EA win — it may be masking a config change fixable upstream.")
            else:
                print("   => Bad folds are spread across regimes / the majority regime too:")
                print("      this looks like donning variance, not a clean config boundary.")
        else:
            print("   Pass --loso-log to test whether the bad folds align with a regime.")


# ---------------------------------------------------------------------------
def selftest():
    import tempfile, os
    print("[selftest] building synthetic recording trees")

    base_cal = {"units": "uV", "ads_vref_volts": 4.5, "eeg_gain": 6.0, "emg_gain": 1.0,
                "eeg_lsb_uv": 0.0894, "emg_lsb_uv": 0.5364}
    order = {str(i): (f"EEG{i+1}" if i < 16 else f"EMG{i-15}") for i in range(32)}

    def write_session(root, name, cal):
        d = root / name
        d.mkdir(parents=True)
        meta = {"created_at_iso": "2026-06-19T18:00:00+00:00", "sample_rate_hz": 1000,
                "channel_count": 32, "channel_units": "uV", "channel_order": order,
                "ads_calibration": cal, "eeg_record_format": "<QQ32f"}
        (d / "session_meta.json").write_text(json.dumps(meta))

    def fake_log(path, rows):
        lines = ["=== per-fold balanced_accuracy (rows=session) ===",
                 "session                                  baseline  adabn  ea  adabn+ea"]
        for name, b in rows:
            lines.append(f"{name:<40} {b:.3f}  {b:.3f}  {b:.3f}  {b:.3f}")
        lines.append("=== paired delta ===")
        Path(path).write_text("\n".join(lines))

    # --- Case A: single regime, one catastrophic fold -> "donning variance" ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "rec"
        names = [f"2026-06-19_2{i}-00-00_session_aaa{i:03d}" for i in range(3)]
        for n in names:
            write_session(root, n, dict(base_cal))
        metas = {n: load_meta(root / n / "session_meta.json") for n in names}
        keys, _ = analyze(names, metas, {})
        assert len(keys) == 1, "case A should be one regime"
        # bad fold parsing
        logp = Path(td) / "log.txt"
        fake_log(logp, [(names[0], 0.49), (names[1], 0.95), (names[2], 0.93)])
        bal = parse_loso_log(logp)
        assert abs(bal[names[0]] - 0.49) < 1e-9 and len(bal) == 3, bal
        print(f"  case A: 1 regime, parsed {len(bal)} bal_acc rows  OK")

    # --- Case B: gain boundary, bad fold in the minority regime -> "artifact" ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "rec"
        good = [f"2026-06-19_2{i}-00-00_session_bbb{i:03d}" for i in range(3)]
        bad_cal = dict(base_cal); bad_cal["eeg_gain"] = 12.0       # regime change
        oddname = "2026-06-21_00-00-00_session_ccc999"
        for n in good:
            write_session(root, n, dict(base_cal))
        write_session(root, oddname, bad_cal)
        alln = good + [oddname]
        metas = {n: load_meta(root / n / "session_meta.json") for n in alln}
        keys, regime_id = analyze(alln, metas, {})
        assert len(keys) == 2, "case B should detect two regimes"
        assert regime_id[regime_key(metas[oddname])] == 1
        print(f"  case B: gain change -> {len(keys)} regimes detected  OK")

    # --- Case C: hard-field drift (channel_count) is caught ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "rec"
        n1 = "2026-06-19_20-00-00_session_ddd001"
        n2 = "2026-06-19_21-00-00_session_ddd002"
        write_session(root, n1, dict(base_cal))
        m2 = {"created_at_iso": "x", "sample_rate_hz": 1000, "channel_count": 30,
              "channel_units": "uV", "channel_order": order,
              "ads_calibration": dict(base_cal), "eeg_record_format": "<QQ32f"}
        (root / n2).mkdir(parents=True)
        (root / n2 / "session_meta.json").write_text(json.dumps(m2))
        metas = {n: load_meta(root / n / "session_meta.json") for n in (n1, n2)}
        hv = {f: {metas[s][f] for s in (n1, n2)} for f in HARD_FIELDS}
        assert len(hv["channel_count"]) == 2, "should see channel_count drift"
        print("  case C: structural drift (channel_count) detected  OK")

    print("[selftest] all checks passed.")


if __name__ == "__main__":
    main()