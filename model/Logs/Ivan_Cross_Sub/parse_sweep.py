#!/usr/bin/env python3
"""
parse_sweep.py  -- W-aware aggregation of single-subject within-session probe logs.

Each log is ONE subject at ONE vocabulary size W:  single_w{W}_s{S}.log
Protocol inside every log: stratified 5-fold WITHIN one subject (optimistic ceiling),
balanced classes (per-class n equal), so macro_f1 needs no support filter here.

We report Cohen's kappa as the PRIMARY cross-W metric because it is chance-corrected:
bal_acc is NOT comparable across W (chance slides 0.333 -> 0.143 from W3 -> W7),
whereas kappa is on a common 0-baseline scale.

Outputs:
  - per (W, mode) aggregate: n, mean/std of kappa and bal_acc, chance
  - word-set lock QC per W (every subject must score the identical trigger-code set)
  - cohort roster per (W, mode) so cross-W deltas are only trusted on shared subjects
"""
import re, glob, os, sys
from collections import defaultdict
import numpy as np

# Splits the file into per-mode blocks. Matches the per-mode TRAINING headers, e.g.
#   ================ overt speech  (subjects [11]) ================
# but NOT the trailing "cross-mode summary (subjects ...)" header.
MODE_HDR = re.compile(r'^=+\s+(overt speech|silent speech|imagined speech)\s+\(subjects',
                      re.MULTILINE)
BAL = re.compile(r'^balanced_accuracy:\s+([0-9.]+)', re.MULTILINE)
KAP = re.compile(r'^cohen_kappa:\s+([0-9.]+)', re.MULTILINE)
F1  = re.compile(r'^macro_f1:\s+([0-9.]+)', re.MULTILINE)
CODES = re.compile(r'trigger codes \[([0-9,\s]+)\]')
CHANCE = re.compile(r'chance=([0-9.]+)')
# cross-mode summary rows, as an independent cross-check of bal_acc
XROW = re.compile(r'^(overt speech|silent speech|imagined speech)\s+([0-9.]+)\s+([0-9.]+)\s*$',
                  re.MULTILINE)

def parse_file(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r'single_w(\d+)_s(\d+)\.log', os.path.basename(path))
    W, S = int(m.group(1)), int(m.group(2))
    cm = CODES.search(txt)
    codeset = tuple(int(x) for x in cm.group(1).split(',')) if cm else None
    chance_vals = [float(c) for c in CHANCE.findall(txt)]
    chance = round(min(chance_vals), 3) if chance_vals else round(1.0/W, 3)

    # cross-mode summary bal_acc (cross-check source)
    xsum = {mode: float(b) for mode, _, b in XROW.findall(txt)}

    # split by per-mode training headers; element 0 is preamble
    parts = MODE_HDR.split(txt)
    rows = []
    # parts = [pre, mode1, body1, mode2, body2, ...]
    for i in range(1, len(parts), 2):
        mode = parts[i]
        body = parts[i+1]
        bal = BAL.search(body); kap = KAP.search(body); f1 = F1.search(body)
        if not (bal and kap):          # mode present but no metric block -> skip
            continue
        bal_v = float(bal.group(1)); kap_v = float(kap.group(1))
        f1_v = float(f1.group(1)) if f1 else float('nan')
        # cross-check against cross-mode summary if available
        xcheck = xsum.get(mode)
        if xcheck is not None and abs(xcheck - bal_v) > 1e-3:
            print(f"  WARN {os.path.basename(path)} {mode}: "
                  f"bal_acc block={bal_v} != summary={xcheck}")
        rows.append((W, S, mode, bal_v, kap_v, f1_v, chance, codeset))
    return rows

def main():
    files = sorted(glob.glob(f"single_w*_s*.log"))
    if not files:
        print("no logs found"); sys.exit(0)

    recs = []
    for f in files:
        recs.extend(parse_file(f))

    # ---- word-set lock QC per W ----
    print("=" * 78)
    print("WORD-SET LOCK QC (all subjects at a given W must share one trigger-code set)")
    print("=" * 78)
    by_W_codes = defaultdict(set)
    by_W_subj = defaultdict(set)
    for W, S, mode, *_rest, codeset in recs:
        by_W_codes[W].add(codeset)
        by_W_subj[W].add(S)
    for W in sorted(by_W_codes):
        sets = by_W_codes[W]
        ok = "OK " if len(sets) == 1 else "MIX"
        subj = sorted(by_W_subj[W])
        print(f"  W={W}  [{ok}] codes={sorted(sets)[0] if len(sets)==1 else sets}  "
              f"n_subj={len(subj)}  subjects={subj[0]}..{subj[-1]}")

    # ---- per (W, mode) aggregate ----
    print()
    print("=" * 78)
    print("AGGREGATE per (W, mode) -- mean +/- std across subjects present")
    print("PRIMARY metric = Cohen's kappa (chance-corrected, comparable across W)")
    print("=" * 78)
    agg = defaultdict(lambda: {"bal": [], "kap": [], "f1": [], "subj": [], "chance": None})
    for W, S, mode, bal_v, kap_v, f1_v, chance, codeset in recs:
        d = agg[(W, mode)]
        d["bal"].append(bal_v); d["kap"].append(kap_v); d["f1"].append(f1_v)
        d["subj"].append(S); d["chance"] = chance

    hdr = f"{'W':>2} {'mode':<14} {'n':>3} {'chance':>7} {'bal_acc':>16} {'kappa':>16} {'macroF1':>8} {'norm':>6}"
    print(hdr); print("-" * len(hdr))
    order = {"overt speech": 0, "silent speech": 1, "imagined speech": 2}
    for (W, mode) in sorted(agg, key=lambda k: (k[0], order.get(k[1], 9))):
        d = agg[(W, mode)]
        n = len(d["kap"])
        bal = np.array(d["bal"]); kap = np.array(d["kap"])
        ch = d["chance"]
        # chance-normalized bal_acc headroom recovered: (bal - chance)/(1 - chance)
        norm = (bal.mean() - ch) / (1 - ch)
        print(f"{W:>2} {mode:<14} {n:>3} {ch:>7.3f} "
              f"{bal.mean():>7.3f}+/-{bal.std():<6.3f} "
              f"{kap.mean():>7.3f}+/-{kap.std():<6.3f} "
              f"{np.nanmean(d['f1']):>8.3f} {norm:>6.3f}")

    # ---- common-cohort W->W+1 deltas (only leakage-free cross-W comparison) ----
    print()
    print("=" * 78)
    print("COMMON-COHORT cross-W kappa deltas (restricted to subjects present at BOTH W)")
    print("This is the only honest level-to-level comparison when cohorts differ.")
    print("=" * 78)
    # build {(W,mode): {S: kappa}}
    kap_by = defaultdict(dict)
    for W, S, mode, bal_v, kap_v, *_ in recs:
        kap_by[(W, mode)][S] = kap_v
    Ws = sorted(set(W for W, _ in kap_by))
    for mode in ["overt speech", "silent speech"]:
        for a, b in zip(Ws, Ws[1:]):
            sa = kap_by.get((a, mode), {}); sb = kap_by.get((b, mode), {})
            shared = sorted(set(sa) & set(sb))
            if not shared:
                continue
            ka = np.array([sa[s] for s in shared]); kb = np.array([sb[s] for s in shared])
            d = kb - ka
            print(f"  {mode:<14} W{a}->W{b}  shared_n={len(shared):>2}  "
                  f"kappa {ka.mean():.3f} -> {kb.mean():.3f}  "
                  f"delta={d.mean():+.3f} (per-subj std {d.std():.3f})  "
                  f"subjects={shared[0]}..{shared[-1]}")

if __name__ == "__main__":
    main()