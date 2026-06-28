"""
ablate_electrodes.py
====================
Find the smallest electrode montage that holds IntermediateFusionEEGNet's
cross-session (LOSO) balanced accuracy, from the 8 EEG + 8 EMG active set down
to a 3 EEG + 3 EMG floor.

WHY NOT EXHAUSTIVE
------------------
Joint exhaustive over both branches (subsets sized 3..8 of each) is
219 x 219 = 47,961 configs x 17 LOSO folds = 815,337 retrains (~280+ days on
one A10). The spatial conv is sized (n_channels, 1), so every channel-count
change forces a from-scratch retrain -- which is also the CORRECT experiment for
a hardware spec (it answers "what would a 3+3 headset score", not "how robust is
the 8+8 model to a dead lead"). So this runs GREEDY BACKWARD ELIMINATION
(recursive channel elimination): from 8+8, drop the channel whose removal costs
the least LOSO balanced accuracy, repeat to the 3+3 floor. ~115 configs ~= 2,000
retrains ~= overnight. Round 1 is leave-one-channel-out, so you get per-channel
marginal importance for free.

Also evaluates the EEG-heavy / full anchors so you can see whether the EEG branch
earns its electrodes (per the montage work, ear EEG word-level decoding sits near
chance; the bigger saving may be dropping EEG leads wholesale, not shaving one).

Optional --frontier-exhaustive verifies the greedy 3+3 is locally optimal by
enumerating the 56 EEG-triples at the chosen EMG-triple and the 56 EMG-triples at
the chosen EEG-triple (112 configs) -- the only place exhaustive is affordable.

WIRED TO YOUR REPO (verified against the uploaded train.py / loso.py):
  * Reads loso.py's window cache: keys Xe(N,8,T), Xm(N,8,T), y_all, fold_of, T,
    and the active_eeg/active_emg metadata (raw electrode IDs for the report).
    NOTE: cache channels are POSITIONAL 0..7 (already sliced to the active set);
    raw IDs [1,3,4,6,7,8,9,11] / [16,18,19,21,23,25,29,31] are only labels.
  * Trains each config with the EXACT loso.intermediate_fold recipe:
    IntermediateFusionEEGNet(n_eeg, n_emg, n_classes, T, p_drop=0.5), Adam,
    epochs=80, batch=32, lr=1e-3, seed=42, inverse-freq class weights,
    input (B,1,C,T), no early stopping. Same loop for every config -> all
    differences are attributable to channels alone.

USAGE
-----
  python ablate_electrodes.py --self-test                       # prove harness
  stdbuf -oL python -u ablate_electrodes.py \
      --cache loso_cache_global.npz --emg-norm global \
      --epochs 80 --out ablation_run/ 2>&1 | tee ablation_run/ablate.log
  python ablate_electrodes.py --cache loso_cache_global.npz --resume \
      --out ablation_run/ --frontier-exhaustive
"""
from __future__ import annotations
import argparse, itertools, json, os, time
from dataclasses import dataclass, field, asdict
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

# =============================================================================
#  DATA -- read loso.py's window cache (exact keys). Channels are positional.
# =============================================================================
def _raw_to_name(raw):
    raw = int(raw)
    return f"EEG{raw+1}" if raw < 16 else f"EMG{raw-16+1}"

def load_windows(args):
    """Return Xe(N,Ce,T), Xm(N,Cm,T), y(N,), groups(N,), eeg_names, emg_names.
    eeg_names[p]/emg_names[p] map array position p -> physical electrode label."""
    if args.cache:
        z = np.load(args.cache, allow_pickle=True)
        keys = set(z.files)
        assert {"Xe", "Xm"} <= keys, f"cache missing Xe/Xm; has {sorted(keys)}"
        Xe = np.asarray(z["Xe"], np.float32)
        Xm = np.asarray(z["Xm"], np.float32)
        yk = "y_all" if "y_all" in keys else ("y" if "y" in keys else None)
        gk = "fold_of" if "fold_of" in keys else ("groups" if "groups" in keys else None)
        assert yk and gk, f"cache missing y_all/fold_of; has {sorted(keys)}"
        y = np.asarray(z[yk]).astype(np.int64)
        groups = np.asarray(z[gk]).astype(np.int64)
        ae = [int(v) for v in z["active_eeg"]] if "active_eeg" in keys else list(range(Xe.shape[1]))
        am = [int(v) for v in z["active_emg"]] if "active_emg" in keys else list(range(16, 16 + Xm.shape[1]))
        return Xe, Xm, y, groups, [_raw_to_name(r) for r in ae], [_raw_to_name(r) for r in am]
    if args.bins:
        import train
        from train_fusion import load_bin_sessions #type: ignore
        Xe32, Xm32, y, g = load_bin_sessions(args.bins, args.events, group_field=args.group_field)
        ae, am = list(train.ACTIVE_EEG_INDICES), list(train.ACTIVE_EMG_INDICES)
        # NOTE: train_fusion decimates EEG (T differs); will NOT match the 0.873
        # baseline. Prefer --cache from loso.py. Slicing to active set here:
        Xe = Xe32[:, ae, :]; Xm = Xm32[:, [a - 16 for a in am], :]
        return Xe, Xm, np.asarray(y), np.asarray(g), [_raw_to_name(r) for r in ae], [_raw_to_name(r) for r in am]
    raise SystemExit("provide --cache, --bins/--events, or run --self-test")

# =============================================================================
#  MODEL -- exact loso.intermediate_fold recipe, with subset channel counts.
# =============================================================================
def make_torch_fit_predict(epochs=80, lr=1e-3, batch=32, seed=42, p_drop=0.5, device=None):
    import torch
    import train  # your train.py: IntermediateFusionEEGNet, seed_everything
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit_predict(Xe_tr, Xm_tr, y_tr, Xe_te, Xm_te, classes):
        train.seed_everything(seed)
        n_classes = len(classes)
        cls = {c: i for i, c in enumerate(classes)}
        yt = np.array([cls[v] for v in y_tr], dtype=np.int64)
        T = Xe_tr.shape[-1]
        Xe_trt = torch.from_numpy(Xe_tr).unsqueeze(1).float().to(dev)   # (B,1,C,T)
        Xm_trt = torch.from_numpy(Xm_tr).unsqueeze(1).float().to(dev)
        y_trt = torch.from_numpy(yt).to(dev)
        model = train.IntermediateFusionEEGNet(
            n_eeg=Xe_tr.shape[1], n_emg=Xm_tr.shape[1],
            n_classes=n_classes, T=T, p_drop=p_drop).to(dev)
        counts = np.bincount(yt, minlength=n_classes).astype(np.float64)
        safe = np.where(counts > 0, counts, 1.0)
        w = torch.tensor(counts.sum() / (safe * n_classes), dtype=torch.float32, device=dev)
        crit = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = Xe_trt.shape[0]; drop_last = n > batch
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                if drop_last and idx.numel() < batch:
                    continue
                opt.zero_grad()
                loss = crit(model(Xe_trt[idx], Xm_trt[idx]), y_trt[idx])
                loss.backward(); opt.step()
        model.eval()
        Xe_tet = torch.from_numpy(Xe_te).unsqueeze(1).float().to(dev)
        Xm_tet = torch.from_numpy(Xm_te).unsqueeze(1).float().to(dev)
        preds = []
        with torch.no_grad():
            for s in range(0, Xe_tet.shape[0], 256):
                preds.append(model(Xe_tet[s:s + 256], Xm_tet[s:s + 256]).argmax(1).cpu().numpy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inv = {i: c for c, i in cls.items()}
        return np.array([inv[i] for i in np.concatenate(preds)])
    return fit_predict

def make_selftest_fit_predict():
    """Torch-free stand-in: per-channel energy -> logistic reg. Only exercises the
    search/eval/checkpoint machinery offline."""
    from sklearn.linear_model import LogisticRegression
    def feat(Xe, Xm):
        return np.concatenate([np.abs(Xe).mean(-1), np.abs(Xm).mean(-1)], axis=1)
    def fit_predict(Xe_tr, Xm_tr, y_tr, Xe_te, Xm_te, classes):
        clf = LogisticRegression(max_iter=500)
        clf.fit(feat(Xe_tr, Xm_tr), y_tr)
        return clf.predict(feat(Xe_te, Xm_te))
    return fit_predict

# =============================================================================
#  Metrics + LOSO evaluation of ONE configuration  (positional indices)
# =============================================================================
@dataclass
class Result:
    eeg: list; emg: list; n_eeg: int; n_emg: int
    balacc_mean: float; balacc_std: float
    kappa: float; macro_f1: float; chance: float
    per_fold: list = field(default_factory=list)

def _macro_f1_support_filtered(y_true, y_pred):
    labels = np.unique(y_true)
    return float(np.mean(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)))

def eval_config(Xe, Xm, y, groups, eeg_idx, emg_idx, fit_predict, *, min_test=1):
    Xe_s, Xm_s = Xe[:, eeg_idx, :], Xm[:, emg_idx, :]
    classes = np.unique(y); chance = 1.0 / len(classes)
    per_fold, oof_true, oof_pred = [], [], []
    for s in np.unique(groups):
        te = groups == s; tr = ~te
        if te.sum() < min_test or len(np.unique(y[tr])) < 2:
            continue
        pred = fit_predict(Xe_s[tr], Xm_s[tr], y[tr], Xe_s[te], Xm_s[te], classes)
        per_fold.append(float(balanced_accuracy_score(y[te], pred)))
        oof_true.append(y[te]); oof_pred.append(pred)
    oof_true = np.concatenate(oof_true); oof_pred = np.concatenate(oof_pred)
    return Result(eeg=list(map(int, eeg_idx)), emg=list(map(int, emg_idx)),
                  n_eeg=len(eeg_idx), n_emg=len(emg_idx),
                  balacc_mean=float(np.mean(per_fold)), balacc_std=float(np.std(per_fold)),
                  kappa=float(cohen_kappa_score(oof_true, oof_pred)),
                  macro_f1=_macro_f1_support_filtered(oof_true, oof_pred),
                  chance=chance, per_fold=per_fold)

# =============================================================================
#  Checkpoint ledger (resume-safe)
# =============================================================================
def _ckey(eeg_idx, emg_idx):
    return "E" + ",".join(map(str, sorted(eeg_idx))) + "|M" + ",".join(map(str, sorted(emg_idx)))

class Ledger:
    def __init__(self, path):
        self.path = path; self.done = {}
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line:
                    r = json.loads(line); self.done[r["key"]] = r["result"]
    def get(self, e, m): return self.done.get(_ckey(e, m))
    def put(self, e, m, result):
        k = _ckey(e, m); self.done[k] = asdict(result)
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": k, "result": asdict(result)}) + "\n")

def evaluate_cached(Xe, Xm, y, groups, e, m, fit_predict, ledger, tag=""):
    hit = ledger.get(e, m)
    if hit is not None:
        return Result(**hit)
    t0 = time.time()
    r = eval_config(Xe, Xm, y, groups, e, m, fit_predict)
    ledger.put(e, m, r)
    print(f"  [{tag}] E{len(e)}+M{len(m)}  balacc={r.balacc_mean:.3f}+-{r.balacc_std:.3f}  "
          f"kappa={r.kappa:.3f}  f1={r.macro_f1:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return r

# =============================================================================
#  Greedy backward elimination + frontier exhaustive
# =============================================================================
def selection_score(r):  # prefer high mean, penalise per-fold variance (re-donning)
    return r.balacc_mean - 0.10 * r.balacc_std

def greedy_backward(Xe, Xm, y, groups, eeg0, emg0, fit_predict, ledger,
                    names_e, names_m, floor_eeg=3, floor_emg=3):
    eeg, emg = list(eeg0), list(emg0)
    base = evaluate_cached(Xe, Xm, y, groups, eeg, emg, fit_predict, ledger, "full")
    path = [("full", base)]
    while len(eeg) > floor_eeg or len(emg) > floor_emg:
        cands = ([("eeg", c) for c in eeg] if len(eeg) > floor_eeg else []) + \
                ([("emg", c) for c in emg] if len(emg) > floor_emg else [])
        scored = []
        for mod, ch in cands:
            e2 = [c for c in eeg if c != ch] if mod == "eeg" else eeg
            m2 = [c for c in emg if c != ch] if mod == "emg" else emg
            nm = names_e[ch] if mod == "eeg" else names_m[ch]
            r = evaluate_cached(Xe, Xm, y, groups, e2, m2, fit_predict, ledger, f"drop {nm}")
            scored.append((selection_score(r), mod, ch, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        _, mod, ch, r = scored[0]
        if mod == "eeg": eeg = [c for c in eeg if c != ch]
        else:            emg = [c for c in emg if c != ch]
        nm = names_e[ch] if mod == "eeg" else names_m[ch]
        print(f"==> removed {nm}: now E{len(eeg)}+M{len(emg)} "
              f"balacc={r.balacc_mean:.3f}+-{r.balacc_std:.3f}", flush=True)
        path.append((f"drop {nm}", r))
    return path, eeg, emg

def frontier_exhaustive(Xe, Xm, y, groups, eeg_final, emg_final, eeg0, emg0,
                        fit_predict, ledger, k_eeg=3, k_emg=3):
    best = []
    for trip in itertools.combinations(sorted(eeg0), k_eeg):
        r = evaluate_cached(Xe, Xm, y, groups, list(trip), emg_final, fit_predict, ledger, "frontier-eeg")
        best.append(("eeg-triple", list(trip), emg_final, r))
    for trip in itertools.combinations(sorted(emg0), k_emg):
        r = evaluate_cached(Xe, Xm, y, groups, eeg_final, list(trip), fit_predict, ledger, "frontier-emg")
        best.append(("emg-triple", eeg_final, list(trip), r))
    best.sort(key=lambda t: selection_score(t[3]), reverse=True)
    return best

# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", help="loso.py window cache npz (preferred)")
    ap.add_argument("--bins", nargs="+"); ap.add_argument("--events", nargs="+")
    ap.add_argument("--group-field", default="collection_block_id")
    ap.add_argument("--emg-norm", choices=["per_channel", "global"], default="global",
                    help="label only (cache is already normed); use the SAME as your baseline")
    ap.add_argument("--floor-eeg", type=int, default=3)
    ap.add_argument("--floor-emg", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ablation_run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--frontier-exhaustive", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ledger = Ledger(os.path.join(args.out, "ledger.jsonl"))
    if not args.resume and ledger.done:
        print(f"[note] {len(ledger.done)} cached configs present; pass --resume to "
              f"reuse or delete {ledger.path} to start clean.")

    if args.self_test:
        rng = np.random.default_rng(0)
        N, T, K, S = 340, 200, 3, 17
        y = rng.integers(0, K, N); groups = rng.integers(0, S, N)
        Xe = rng.standard_normal((N, 8, T)).astype(np.float32)
        Xm = rng.standard_normal((N, 8, T)).astype(np.float32)
        for k in range(K):
            Xe[y == k, k] += 0.8; Xm[y == k, k] += 1.2     # signal in ch 0,1,2
        for s in range(S):
            Xe[groups == s] += rng.standard_normal((1, 8, 1)).astype(np.float32) * 0.4
        ne = [f"EEG{i}" for i in range(8)]; nm = [f"EMG{i}" for i in range(8)]
        fp = make_selftest_fit_predict()
        path, ef, mf = greedy_backward(Xe, Xm, y, groups, list(range(8)), list(range(8)),
                                       fp, ledger, ne, nm, args.floor_eeg, args.floor_emg)
        _write_report(args.out, path, ef, mf, None, ne, nm)
        print(f"[self-test] PASS. final EEG{[ne[i] for i in sorted(ef)]} "
              f"EMG{[nm[i] for i in sorted(mf)]}")
        return

    Xe, Xm, y, groups, names_e, names_m = load_windows(args)
    print(f"loaded Xe{Xe.shape} Xm{Xm.shape} sessions={len(np.unique(groups))} "
          f"classes={len(np.unique(y))}\n  EEG={names_e}\n  EMG={names_m}", flush=True)
    fp = make_torch_fit_predict(epochs=args.epochs, lr=args.lr,
                                batch=args.batch_size, seed=args.seed)
    eeg0, emg0 = list(range(Xe.shape[1])), list(range(Xm.shape[1]))

    print("\n[anchor] EEG-full + EMG at floor (is the EMG branch enough?)")
    evaluate_cached(Xe, Xm, y, groups, eeg0, emg0[:args.floor_emg], fp, ledger, "emg-floor")
    print("\n[greedy backward elimination]")
    path, ef, mf = greedy_backward(Xe, Xm, y, groups, eeg0, emg0, fp, ledger,
                                   names_e, names_m, args.floor_eeg, args.floor_emg)
    frontier = None
    if args.frontier_exhaustive:
        print("\n[frontier exhaustive @ floor]")
        frontier = frontier_exhaustive(Xe, Xm, y, groups, ef, mf, eeg0, emg0,
                                        fp, ledger, args.floor_eeg, args.floor_emg)
    _write_report(args.out, path, ef, mf, frontier, names_e, names_m)
    print(f"\nwrote {os.path.join(args.out, 'report.md')}")

def _write_report(out, path, ef, mf, frontier, names_e, names_m):
    L = ["# Electrode ablation report", "",
         "Legend: positions map to physical electrodes as:",
         "  EEG " + ", ".join(f"{i}->{n}" for i, n in enumerate(names_e)),
         "  EMG " + ", ".join(f"{i}->{n}" for i, n in enumerate(names_m)), "",
         "## Greedy backward-elimination path",
         "| step | montage | balacc | std | kappa | macroF1 | chance |",
         "|---|---|---|---|---|---|---|"]
    for tag, r in path:
        L.append(f"| {tag} | E{r.n_eeg}+M{r.n_emg} | {r.balacc_mean:.3f} | {r.balacc_std:.3f} "
                 f"| {r.kappa:.3f} | {r.macro_f1:.3f} | {r.chance:.3f} |")
    L += ["", f"**Greedy final montage:** EEG {[names_e[i] for i in sorted(ef)]}  "
              f"EMG {[names_m[i] for i in sorted(mf)]}", ""]
    budgets = {}
    for _, r in path:
        b = r.n_eeg + r.n_emg
        if b not in budgets or r.balacc_mean > budgets[b].balacc_mean:
            budgets[b] = r
    L += ["## Accuracy vs total electrodes (pick the knee)",
          "| total ch | E+M | balacc | std |", "|---|---|---|---|"]
    for b in sorted(budgets, reverse=True):
        r = budgets[b]
        L.append(f"| {b} | {r.n_eeg}+{r.n_emg} | {r.balacc_mean:.3f} | {r.balacc_std:.3f} |")
    if frontier:
        L += ["", "## Frontier exhaustive (top 5)",
              "| kind | EEG | EMG | balacc | std |", "|---|---|---|---|---|"]
        for kind, e, m, r in frontier[:5]:
            L.append(f"| {kind} | {[names_e[i] for i in sorted(e)]} | "
                     f"{[names_m[i] for i in sorted(m)]} | {r.balacc_mean:.3f} | {r.balacc_std:.3f} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
