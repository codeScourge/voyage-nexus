"""
tmspd_smoke.py  —  T-MSPD cross-subject RANKING harness (LOSO over 30 subjects).
Wired into your loso.py / loso_align.py / train.py, on top of tmspd_loader.py.

Unlike the Gowda smoke test, this is the real thing: paired EEG+EMG, ~30
subjects, the actual IntermediateFusionEEGNet, and a real meta-distribution.

--modality {fusion, eeg, emg}   (run one per invocation)
  fusion : train.IntermediateFusionEEGNet (the shipped model). REQUIRES EEG+EMG
           at a COMMON 1 kHz (loader eeg_fs_out=1000) so time axes correspond;
           the model's min(T) crop is a padding safety, NOT a rate adapter.
  eeg    : EEGNet branch on the peri-auricular subset  (montage-honest arm).
  emg    : EMGNet branch on the 6 perioral channels    (algorithm prior only).

ARMS (leave-one-SUBJECT-out):
  Group A  unsupervised / transductive (full-target eval):
    baseline | ea | adabn | adabn+ea  (reuse loso_align.ea_align/adabn_recompute)
    mmd                               shared enc + MMD(src, tgt-unlabeled)
    dape                              K private encoders + MMD across them
  Group B  few-shot supervised (target remainder eval):
    peft_linear | peft_lora | peft_dora | peft_full
    reptile                           first-order meta-learn over source subjects

REPORT: loso.compute_metrics four-number table + paired-delta vs baseline AND vs
ea (your locked base). At ~30 folds the Wilcoxon gate you trust on the rig is
finally powered.

DESIGN NOTES YOU SHOULD KNOW
  * DAPE with one private encoder PER subject is impractical at n=30 (each sees
    1/29 of data). We cap at --dape-k clusters (default 4, the paper's M); source
    subjects are partitioned round-robin. Target routes through the mean of the K
    encoders. This is the scalable, faithful instantiation.
  * EA is applied to EEG by default (--ea-modality); on EMG it can destroy the
    cross-channel ratios global-norm preserves (your rig finding).
  * meta = Reptile (first-order), the stable choice for a screen; full MAML is
    finicky and not worth it before Reptile says meta-adaptation helps at all.
"""
from __future__ import annotations
import argparse, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

from tmspd_loader import load_tmspd_dataset, PERIAURIC_LEFT

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it=None, **k):
        return it if it is not None else iter([])

import loso
import loso_align
Path("checkpoints").mkdir(exist_ok=True)
try:
    import train as _train
    ModalityBranch = _train.ModalityBranch
    TimeAvgPool = _train.TimeAvgPool
    RealFusion = _train.IntermediateFusionEEGNet
    seed_everything = _train.seed_everything
    get_device = _train.get_device
    _SRC = "train.IntermediateFusionEEGNet (shipped)"
except Exception as e:
    print(f"[warn] train.py import failed ({type(e).__name__}); inline fallbacks.")
    RealFusion = None

    class ModalityBranch(nn.Module):
        def __init__(self, n_channels, F1, D, kernel_length):
            super().__init__()
            self.temporal = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.spatial = nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(D * F1)

        def forward(self, x):
            return F.elu(self.bn2(self.spatial(self.bn1(self.temporal(x)))))

    class TimeAvgPool(nn.Module):
        def __init__(self, out_len):
            super().__init__(); self.out_len = out_len

        def forward(self, x):
            t = x.shape[-1]; out = self.out_len
            if t == out:
                return x
            if t < out:
                return F.interpolate(x, size=(1, out), mode="linear", align_corners=False)
            trim = t - (t % out); x = x[..., :trim]; s = trim // out
            return F.avg_pool2d(x, (1, s), (1, s))

    def seed_everything(seed=0, deterministic=False):
        import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _SRC = "inline fallback"


# ---------------------------------------------------------------------------
#  Models — every model takes forward(eeg, emg) so the method code is uniform.
#  Single-modality nets ignore the unused input. classifier can be swapped to
#  Identity to expose flattened features (used by DAPE/MMD via a tap).
# ---------------------------------------------------------------------------
def _block2(fused_maps, T, F2, sep_kernel, p):
    pool1_out = max(1, T // 4); pool2_out = max(1, pool1_out // 8)
    mods = nn.ModuleDict(dict(
        pool1=TimeAvgPool(pool1_out), drop1=nn.Dropout(p),
        sep_depth=nn.Conv2d(fused_maps, fused_maps, (1, sep_kernel), padding=(0, sep_kernel // 2), groups=fused_maps, bias=False),
        sep_point=nn.Conv2d(fused_maps, F2, (1, 1), bias=False),
        bn3=nn.BatchNorm2d(F2), pool2=TimeAvgPool(pool2_out), drop2=nn.Dropout(p)))
    return mods, F2 * pool2_out


def _run_block2(m, x):
    x = m["drop1"](m["pool1"](x))
    x = m["sep_point"](m["sep_depth"](x))
    x = F.elu(m["bn3"](x))
    return torch.flatten(m["drop2"](m["pool2"](x)), 1)


class FusionNet(nn.Module):
    def __init__(self, n_eeg, n_emg, n_classes, T, F1=8, D=2, F2=16,
                 kern_eeg=500, kern_emg=128, sep_kernel=64, p=0.25):
        super().__init__()
        if RealFusion is not None:          # use the shipped model verbatim
            self.net = RealFusion(n_eeg, n_emg, n_classes, T, F1, D, F2, kern_eeg, kern_emg, sep_kernel, p)
            self.classifier = self.net.classifier
            self._real = True
        else:
            self.eeg_branch = ModalityBranch(n_eeg, F1, D, kern_eeg)
            self.emg_branch = ModalityBranch(n_emg, F1, D, kern_emg)
            self.b2, feat = _block2(2 * D * F1, T, F2, sep_kernel, p)
            self.classifier = nn.Linear(feat, n_classes)
            self._real = False

    def features(self, eeg, emg):
        if self._real:
            e = self.net.eeg_branch(eeg); m = self.net.emg_branch(emg)
            t = min(e.shape[-1], m.shape[-1]); x = torch.cat([e[..., :t], m[..., :t]], 1)
            x = self.net.drop1(self.net.pool1(x))
            x = self.net.sep_point(self.net.sep_depth(x))
            x = F.elu(self.net.bn3(x))
            return torch.flatten(self.net.drop2(self.net.pool2(x)), 1)
        e = self.eeg_branch(eeg); m = self.emg_branch(emg)
        t = min(e.shape[-1], m.shape[-1])
        return _run_block2(self.b2, torch.cat([e[..., :t], m[..., :t]], 1))

    def embed(self, eeg, emg):
        return self.features(eeg, emg)

    def forward(self, eeg, emg):
        return self.classifier(self.features(eeg, emg))


class SingleNet(nn.Module):
    """EEG-only or EMG-only EEGNet. which='eeg' uses arg1, 'emg' uses arg2."""

    def __init__(self, which, n_ch, n_classes, T, F1=8, D=2, F2=16, kern=500, sep_kernel=64, p=0.25):
        super().__init__()
        self.which = which
        self.branch = ModalityBranch(n_ch, F1, D, kern)
        self.b2, feat = _block2(D * F1, T, F2, sep_kernel, p)
        self.classifier = nn.Linear(feat, n_classes)

    def _pick(self, eeg, emg):
        return eeg if self.which == "eeg" else emg

    def embed(self, eeg, emg=None):
        return _run_block2(self.b2, self.branch(self._pick(eeg, emg)))

    def forward(self, eeg, emg=None):
        return self.classifier(self.embed(eeg, emg))


def make_model(modality, n_eeg, n_emg, n_classes, T, **kw):
    if modality == "fusion":
        return FusionNet(n_eeg, n_emg, n_classes, T, **kw)
    if modality == "eeg":
        return SingleNet("eeg", n_eeg, n_classes, T, kern=kw.get("kern_eeg", 500))
    if modality == "emg":
        return SingleNet("emg", n_emg, n_classes, T, kern=kw.get("kern_emg", 128))
    raise ValueError(modality)


class DAPENet(nn.Module):
    """K private feature extractors (full models w/ classifier->Identity) + head."""

    def __init__(self, K, modality, n_eeg, n_emg, n_classes, T, **kw):
        super().__init__()
        self.feats = nn.ModuleList([make_model(modality, n_eeg, n_emg, n_classes, T, **kw) for _ in range(K)])
        feat_dim = self.feats[0].classifier.in_features
        for f in self.feats:
            f.classifier = nn.Identity()
        self.head = nn.Linear(feat_dim, n_classes)

    def embed_src(self, eeg, emg, gid):
        z = None
        for e in torch.unique(gid):
            m = gid == e
            zz = self.feats[int(e)](eeg[m], emg[m])
            if z is None:
                z = eeg.new_zeros(eeg.shape[0], zz.shape[1])
            z[m] = zz
        return z

    def embed_tgt(self, eeg, emg):
        return torch.stack([f(eeg, emg) for f in self.feats], 0).mean(0)

    def forward_tgt(self, eeg, emg):
        return self.head(self.embed_tgt(eeg, emg))


# ---------------------------------------------------------------------------
#  PEFT / MMD utilities
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8.0):
        super().__init__()
        self.W0 = base.weight.detach().clone(); self.W0.requires_grad_(False)
        self.b0 = base.bias.detach().clone() if base.bias is not None else None
        if self.b0 is not None:
            self.b0.requires_grad_(False)
        o, i = self.W0.shape
        self.A = nn.Parameter(torch.zeros(r, i)); self.B = nn.Parameter(torch.zeros(o, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5); self.s = alpha / r

    def _w0(self, d):
        if self.W0.device != d:
            self.W0 = self.W0.to(d); self.b0 = self.b0.to(d) if self.b0 is not None else None
        return self.W0

    def forward(self, x):
        return F.linear(x, self._w0(x.device) + self.s * (self.B @ self.A), self.b0)


class DoRALinear(LoRALinear):
    def __init__(self, base, r=4, alpha=8.0):
        super().__init__(base, r, alpha); self.m = nn.Parameter(self.W0.norm(dim=1).detach().clone())

    def forward(self, x):
        V = self._w0(x.device) + self.s * (self.B @ self.A)
        return F.linear(x, self.m.unsqueeze(1) * V / (V.norm(dim=1, keepdim=True) + 1e-6), self.b0)


def _mmd2(x, y, mults=(0.25, 0.5, 1, 2, 4)):
    xx = torch.cdist(x, x) ** 2; yy = torch.cdist(y, y) ** 2; xy = torch.cdist(x, y) ** 2
    with torch.no_grad():
        med = torch.median(torch.cat([xx, yy, xy]).reshape(-1)).clamp_min(1e-6)
    o = x.new_zeros(())
    for mu in mults:
        s = 1.0 / (2 * mu * med)
        o = o + torch.exp(-s * xx).mean() + torch.exp(-s * yy).mean() - 2 * torch.exp(-s * xy).mean()
    return o / len(mults)


def _wce(y, K, d):
    c = np.bincount(y, minlength=K).astype(float); safe = np.where(c > 0, c, 1)
    return torch.tensor(c.sum() / (safe * K), dtype=torch.float32, device=d)


def _batches(n, bs, d, drop):
    p = torch.randperm(n, device=d)
    for s in range(0, n, bs):
        idx = p[s:s + bs]
        if drop and idx.numel() < bs:
            continue
        yield idx


# ---------------------------------------------------------------------------
#  Early stopping. The validation signal is a stratified slice of the TRAINING
#  data passed in (source subjects for Group A, the few-shot set for PEFT) --
#  never the held-out test subject. Each arm stops on its OWN convergence, so
#  the slow alignment/DAPE/PEFT arms are not handicapped by a fixed epoch count.
#  --epochs is the CEILING; training usually stops earlier.
# ---------------------------------------------------------------------------
def _val_split(y_np, frac, seed, K):
    """Stratified (train_idx, val_idx); None if disabled or too small to split."""
    if frac is None or frac <= 0:
        return None
    n = len(y_np)
    n_val = min(max(K, int(round(frac * n))), n - K)
    if n_val < K or n - n_val < K:
        return None
    from sklearn.model_selection import StratifiedShuffleSplit
    tr, va = next(StratifiedShuffleSplit(1, test_size=n_val, random_state=seed).split(np.zeros(n), y_np))
    return tr, va


@torch.no_grad()
def _val_ce(model, Xe, Xm, y, crit, fwd, bs=256):
    model.eval(); tot = 0.0; nb = 0
    for i in range(0, len(y), bs):
        tot += float(crit(fwd(Xe[i:i + bs], Xm[i:i + bs]), y[i:i + bs])); nb += 1
    model.train(); return tot / max(1, nb)


class _Stopper:
    def __init__(self, model, patience, min_epochs):
        self.m = model; self.pat = patience; self.min_ep = min_epochs
        self.best = float("inf"); self.state = None; self.bad = 0; self.stop_ep = None

    def step(self, ep, vloss):
        if vloss < self.best - 1e-4:
            self.best = vloss; self.bad = 0
            self.state = {k: v.detach().cpu().clone() for k, v in self.m.state_dict().items()}
        else:
            self.bad += 1
        if ep + 1 >= self.min_ep and self.bad >= self.pat:
            self.stop_ep = ep + 1; return True
        return False

    def restore(self):
        if self.state is not None:
            self.m.load_state_dict(self.state)


def _epoch_loop(model, train_step, val_fn, *, max_epochs, patience, min_epochs,
                tag, verbose):
    """Run train_step() per epoch; if val_fn given, early-stop + restore best."""
    stop = _Stopper(model, patience, min_epochs) if val_fn is not None else None
    for ep in tqdm(range(max_epochs), desc=tag, disable=not verbose, leave=False):
        train_step()
        if stop is not None and stop.step(ep, val_fn()):
            break
    if stop is not None:
        stop.restore()
        if verbose:
            print(f"     [{tag}] stop@{stop.stop_ep or max_epochs} (best_val={stop.best:.3f})")
    return model


# ---------------------------------------------------------------------------
#  Training loops (eeg,emg tensors; single-modality nets ignore the unused one)
# ---------------------------------------------------------------------------
def _fit(model, Xe, Xm, y, *, K, epochs, lr, bs, dev, seed, mmd_tgt=None, lam=0.0,
         patience=10, min_epochs=12, val_frac=0.15, verbose=False, tag="fit"):
    seed_everything(seed); model.to(dev).train()
    Xe, Xm, y = Xe.to(dev), Xm.to(dev), y.to(dev)
    crit = nn.CrossEntropyLoss(weight=_wce(y.cpu().numpy(), K, dev))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sp = _val_split(y.cpu().numpy(), val_frac, seed, K)
    if sp is None:
        tr_t = torch.arange(Xe.shape[0], device=dev); val_fn = None
    else:
        tr_t = torch.as_tensor(sp[0], device=dev); va = torch.as_tensor(sp[1], device=dev)
        val_fn = lambda: _val_ce(model, Xe[va], Xm[va], y[va], crit, lambda a, b: model(a, b))
    Te = Tm = None
    if mmd_tgt is not None:
        Te, Tm = mmd_tgt[0].to(dev), mmd_tgt[1].to(dev)
    drop = len(tr_t) > bs

    def step():
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if drop and idx.numel() < bs:
                continue
            opt.zero_grad()
            if mmd_tgt is None:
                loss = crit(model(Xe[idx], Xm[idx]), y[idx])
            else:
                ti = torch.randint(0, Te.shape[0], (idx.numel(),), device=dev)
                zs = model.embed(Xe[idx], Xm[idx]); zt = model.embed(Te[ti], Tm[ti])
                head = model.classifier if hasattr(model, "classifier") else model.head
                loss = crit(head(zs), y[idx]) + lam * _mmd2(zs, zt)
            loss.backward(); opt.step()

    return _epoch_loop(model, step, val_fn, max_epochs=epochs, patience=patience,
                       min_epochs=min_epochs, tag=tag, verbose=verbose)


def _fit_dape(model, Xe, Xm, y, gid, *, K, epochs, lr, bs, kappa, dev, seed,
              patience=10, min_epochs=12, val_frac=0.15, verbose=False):
    seed_everything(seed); model.to(dev).train()
    Xe, Xm, y = Xe.to(dev), Xm.to(dev), y.to(dev)
    gid_t = torch.as_tensor(gid, device=dev)
    crit = nn.CrossEntropyLoss(weight=_wce(y.cpu().numpy(), K, dev))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sp = _val_split(y.cpu().numpy(), val_frac, seed, K)
    if sp is None:
        tr_t = torch.arange(Xe.shape[0], device=dev); val_fn = None
    else:
        tr_t = torch.as_tensor(sp[0], device=dev); va = torch.as_tensor(sp[1], device=dev)
        val_fn = lambda: _val_ce(model, Xe[va], Xm[va], y[va], crit, lambda a, b: model.forward_tgt(a, b))
    drop = len(tr_t) > bs

    def step():
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if drop and idx.numel() < bs:
                continue
            gb = gid_t[idx]; opt.zero_grad()
            ce = crit(model.head(model.embed_src(Xe[idx], Xm[idx], gb)), y[idx])
            mmd = Xe.new_zeros(()); pres = [int(e) for e in torch.unique(gb)]
            for i in range(len(pres)):
                for j in range(i + 1, len(pres)):
                    mmd = mmd + _mmd2(model.feats[pres[i]](Xe[idx], Xm[idx]),
                                      model.feats[pres[j]](Xe[idx], Xm[idx]))
            (ce + kappa * mmd).backward(); opt.step()

    return _epoch_loop(model, step, val_fn, max_epochs=epochs, patience=patience,
                       min_epochs=min_epochs, tag="dape", verbose=verbose)


def _adapt(pre, Xe, Xm, y, *, mode, K, epochs, lr, bs, r, alpha, dev, seed,
           patience=6, min_epochs=5, val_frac=0.2, verbose=False):
    seed_everything(seed); model = copy.deepcopy(pre).to(dev)
    head = model.head if hasattr(model, "head") else model.classifier
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        params = model.parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        new = (LoRALinear(head, r, alpha) if mode == "lora" else
               DoRALinear(head, r, alpha) if mode == "dora" else
               nn.Linear(head.in_features, head.out_features))
        if hasattr(model, "head"):
            model.head = new.to(dev)
        else:
            model.classifier = new.to(dev)
            if getattr(model, "_real", False):
                model.net.classifier = model.classifier
        params = [p for p in new.parameters() if p.requires_grad]
    Xe, Xm, y = Xe.to(dev), Xm.to(dev), y.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params, lr=lr); model.train(); n = Xe.shape[0]
    sp = _val_split(y.cpu().numpy(), val_frac, seed, K)
    if sp is None:
        tr_t = torch.arange(n, device=dev); val_fn = None
    else:
        tr_t = torch.as_tensor(sp[0], device=dev); va = torch.as_tensor(sp[1], device=dev)
        val_fn = lambda: _val_ce(model, Xe[va], Xm[va], y[va], crit, lambda a, b: model(a, b))
    drop = len(tr_t) > bs

    def step():
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), min(bs, len(perm))):
            idx = perm[s:s + bs]
            if drop and idx.numel() < bs:
                continue
            opt.zero_grad(); crit(model(Xe[idx], Xm[idx]), y[idx]).backward(); opt.step()

    return _epoch_loop(model, step, val_fn, max_epochs=epochs, patience=patience,
                       min_epochs=min_epochs, tag=f"peft:{mode}", verbose=verbose)


def _reptile(proto_factory, subj_data, *, K, meta_iters, inner_steps, inner_lr, meta_lr, bs, dev, seed, verbose=False):
    """First-order Reptile over source subjects-as-tasks. subj_data: list of
    (Xe,Xm,y) per source subject. Returns a meta-initialized model."""
    seed_everything(seed); meta = proto_factory().to(dev)
    rng = np.random.default_rng(seed)
    for it in tqdm(range(meta_iters), desc="reptile", disable=not verbose, leave=False):
        Xe, Xm, y = subj_data[rng.integers(len(subj_data))]
        task = copy.deepcopy(meta).to(dev)
        opt = torch.optim.Adam(task.parameters(), lr=inner_lr)
        Xe, Xm, y = Xe.to(dev), Xm.to(dev), y.to(dev); n = Xe.shape[0]
        task.train()
        for _ in range(inner_steps):
            idx = torch.randint(0, n, (min(bs, n),), device=dev)
            opt.zero_grad(); F.cross_entropy(task(Xe[idx], Xm[idx]), y[idx]).backward(); opt.step()
        with torch.no_grad():                       # meta update: move toward task
            for pm, pt in zip(meta.parameters(), task.parameters()):
                pm.add_(meta_lr * (pt.detach() - pm))
    return meta


@torch.no_grad()
def _predict(forward, Xe, Xm, dev, bs=256):
    out = []
    for i in range(0, Xe.shape[0], bs):
        out.append(forward(Xe[i:i + bs].to(dev), Xm[i:i + bs].to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
GROUP_A = ["baseline", "ea", "adabn", "adabn+ea", "mmd", "dape"]
GROUP_B = ["peft_linear", "peft_lora", "peft_dora", "peft_full", "reptile"]


def run(Xe, Xm, y, groups, *, modality, methods, K, T, dev, epochs, lr, bs, lam,
        kappa, dape_k, shots, peft_r, peft_alpha, ea_modality, ea_reg, seed, verbose,
        meta_iters, inner_steps, inner_lr, meta_lr,
        patience=10, min_epochs=12, val_frac=0.15):
    es = dict(patience=patience, min_epochs=min_epochs, val_frac=val_frac)
    n_eeg, n_emg = Xe.shape[1], Xm.shape[1]
    Ee = torch.from_numpy(Xe).unsqueeze(1); Em = torch.from_numpy(Xm).unsqueeze(1)
    yy = torch.from_numpy(y); subjects = np.unique(groups)

    # EA per subject (reuse loso_align.ea_align) on chosen modality/ies
    need_ea = any(m in ("ea", "adabn+ea") for m in methods)
    Ee_ea, Em_ea = Ee, Em
    if need_ea:
        if ea_modality in ("eeg", "both"):
            Ee_ea = torch.from_numpy(loso_align.ea_align(Xe, groups, reg=ea_reg)).unsqueeze(1)
        if ea_modality in ("emg", "both"):
            Em_ea = torch.from_numpy(loso_align.ea_align(Xm, groups, reg=ea_reg)).unsqueeze(1)

    proto = lambda: make_model(modality, n_eeg, n_emg, K, T)
    foldA = {m: [] for m in methods if m in GROUP_A}; poolA = {m: ([], []) for m in foldA}
    foldB = {m: [] for m in methods if m in GROUP_B}; poolB = {m: ([], []) for m in foldB}
    bn_seen = 0

    for fi, held in enumerate(subjects):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        yt = y[te]
        print(f"\n--- fold {fi+1}/{len(subjects)}: hold out S{held} (src {len(tr)}, tgt {len(te)}) ---")
        pre = pre_ea = None
        if {"baseline", "adabn"} & set(methods) or (set(GROUP_B) & set(methods)):
            pre = _fit(proto(), Ee[tr], Em[tr], yy[tr], K=K, epochs=epochs, lr=lr, bs=bs, dev=dev, seed=seed, verbose=verbose, tag="base", **es)
        if {"ea", "adabn+ea"} & set(methods):
            pre_ea = _fit(proto(), Ee_ea[tr], Em_ea[tr], yy[tr], K=K, epochs=epochs, lr=lr, bs=bs, dev=dev, seed=seed, verbose=verbose, tag="ea", **es)

        def rec(fold, pool, m, yt_, yp_):
            mm = loso.compute_metrics(yt_, yp_, K, "_")
            fold[m].append(mm["balanced_accuracy"]); pool[m][0].append(yt_); pool[m][1].append(yp_)
            print(f"   {m:12s} bal_acc={mm['balanced_accuracy']:.3f} kappa={mm['kappa']:.3f} mF1={mm['macro_f1']:.3f}")

        for m in methods:
            if m not in GROUP_A:
                continue
            if m == "baseline":
                yp = _predict(pre.eval(), Ee[te], Em[te], dev)
            elif m == "ea":
                yp = _predict(pre_ea.eval(), Ee_ea[te], Em_ea[te], dev)
            elif m in ("adabn", "adabn+ea"):
                base = pre_ea if m == "adabn+ea" else pre
                ee, em = (Ee_ea, Em_ea) if m == "adabn+ea" else (Ee, Em)
                mdl = copy.deepcopy(base)
                bn_seen = max(bn_seen, loso_align.adabn_recompute(mdl, ee[te].to(dev), em[te].to(dev), dev))
                yp = _predict(mdl.eval(), ee[te], em[te], dev)
            elif m == "mmd":
                mdl = _fit(proto(), Ee[tr], Em[tr], yy[tr], K=K, epochs=epochs, lr=lr, bs=bs, dev=dev, seed=seed,
                           mmd_tgt=(Ee[te], Em[te]), lam=lam, verbose=verbose, tag="mmd", **es)
                yp = _predict(mdl.eval(), Ee[te], Em[te], dev)
            elif m == "dape":
                g_src = groups[tr]; uniq = np.unique(g_src)
                cl = {s: (i % dape_k) for i, s in enumerate(uniq)}   # round-robin K clusters
                gid = np.array([cl[s] for s in g_src])
                mdl = DAPENet(min(dape_k, len(uniq)), modality, n_eeg, n_emg, K, T).to(dev)
                _fit_dape(mdl, Ee[tr], Em[tr], yy[tr], gid, K=K, epochs=epochs, lr=lr, bs=bs, kappa=kappa, dev=dev, seed=seed, verbose=verbose, **es)
                mdl.eval(); yp = _predict(lambda a, b: mdl.forward_tgt(a, b), Ee[te], Em[te], dev)
            rec(foldA, poolA, m, yt, yp)

        if set(GROUP_B) & set(methods):
            na = min(shots * K, len(yt) - K)
            ai, ei = next(StratifiedShuffleSplit(1, train_size=na, random_state=seed).split(np.zeros(len(yt)), yt))
            for m in methods:
                if m not in GROUP_B:
                    continue
                if m == "reptile":
                    subj_data = [(Ee[groups == s], Em[groups == s], yy[groups == s]) for s in np.unique(groups[tr])]
                    meta = _reptile(proto, subj_data, K=K, meta_iters=meta_iters, inner_steps=inner_steps,
                                    inner_lr=inner_lr, meta_lr=meta_lr, bs=bs, dev=dev, seed=seed, verbose=verbose)
                    mdl = _adapt(meta, Ee[te][ai], Em[te][ai], yy[te][ai], mode="full", K=K, epochs=max(5, epochs // 4),
                                 lr=lr, bs=bs, r=peft_r, alpha=peft_alpha, dev=dev, seed=seed)
                else:
                    mdl = _adapt(pre, Ee[te][ai], Em[te][ai], yy[te][ai], mode=m.split("_", 1)[1], K=K,
                                 epochs=epochs, lr=lr, bs=bs, r=peft_r, alpha=peft_alpha, dev=dev, seed=seed, verbose=verbose)
                yp = _predict(mdl.eval(), Ee[te][ei], Em[te][ei], dev)
                rec(foldB, poolB, m, yt[ei], yp)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return foldA, poolA, foldB, poolB, bn_seen


def report(foldA, poolA, foldB, poolB, *, K, chance, methods, modality, shots, bn_seen):
    A = [m for m in methods if m in GROUP_A]; B = [m for m in methods if m in GROUP_B]
    print("\n" + "=" * 78)
    print(f" T-MSPD CROSS-SUBJECT RANKING  —  modality={modality}  —  {len(foldA[A[0]]) if A else len(foldB[B[0]])}-fold LOSO")
    print(f" encoder: {_SRC} | chance bal_acc={chance:.3f} ({K} classes)")
    if modality == "eeg":
        print(" EEG peri-auricular subset = montage-honest (≈ your rig)")
    elif modality == "emg":
        print(" EMG perioral = algorithm-level prior only (not your arc)")
    print("=" * 78)
    if A:
        print("\n## Group A — unsupervised / transductive (full-target eval)")
        loso.print_summary_table([loso.compute_metrics(np.concatenate(poolA[m][0]), np.concatenate(poolA[m][1]), K, m) for m in A], chance=chance)
        print("\n per-fold balanced_accuracy:")
        print(" subj   " + "".join(f"{m:>12}" for m in A))
        for f in range(len(foldA[A[0]])):
            print(f" {f:>4}   " + "".join(f"{foldA[m][f]:>12.3f}" for m in A))
        for ref in ("baseline", "ea"):
            if ref in A:
                base = np.array(foldA[ref])
                try:
                    from scipy.stats import wilcoxon
                except Exception:
                    wilcoxon = None
                print(f"\n paired delta vs {ref}:")
                for m in A:
                    if m == ref:
                        continue
                    d = np.array(foldA[m]) - base
                    p = ""
                    if wilcoxon is not None and len(d) >= 6 and np.any(d != 0):
                        try:
                            p = f"  wilcoxon_p={wilcoxon(foldA[m], base).pvalue:.3f}"
                        except Exception:
                            p = ""
                    print(f"   {m:12s} mean_delta={d.mean():+.4f}{p}")
        if ("adabn" in A or "adabn+ea" in A) and bn_seen == 0:
            print("\n !!! AdaBN found 0 BatchNorm -> no-op.")
    if B:
        print("\n## Group B — few-shot supervised / meta (target remainder eval)")
        loso.print_summary_table([loso.compute_metrics(np.concatenate(poolB[m][0]), np.concatenate(poolB[m][1]), K, m) for m in B], chance=chance)
        print(" peft_linear = head-from-scratch floor; reptile = meta-init + few-shot.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="path to '2.raw data'")
    ap.add_argument("--mode", default="overt speech", choices=["overt speech", "silent speech", "imagined speech"])
    ap.add_argument("--modality", default="fusion", choices=["fusion", "eeg", "emg"])
    ap.add_argument("--subjects", default="1-30", help="e.g. 1-30 or 1,2,5")
    ap.add_argument("--methods", nargs="+", default=GROUP_A + GROUP_B)
    ap.add_argument("--tmin", type=float, default=-0.1); ap.add_argument("--tmax", type=float, default=1.0)
    ap.add_argument("--eeg-channels", default="periauric")
    ap.add_argument("--common-fs", type=int, default=1000, help="EEG+EMG common rate (fusion needs this)")
    ap.add_argument("--emg-offset-s", type=float, default=0.0)
    ap.add_argument("--ea-modality", default="eeg", choices=["eeg", "emg", "both"])
    ap.add_argument("--ea-reg", type=float, default=1e-6)
    ap.add_argument("--dape-k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10, help="early-stop patience (epochs w/o val improvement)")
    ap.add_argument("--min-epochs", type=int, default=12, help="floor before early stop can trigger")
    ap.add_argument("--val-frac", type=float, default=0.15, help="stratified source slice for val signal; 0 disables")
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lambda-mmd", type=float, default=1.0)
    ap.add_argument("--kappa-dape", type=float, default=1.0); ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--peft-r", type=int, default=4); ap.add_argument("--peft-alpha", type=float, default=8.0)
    ap.add_argument("--meta-iters", type=int, default=400); ap.add_argument("--inner-steps", type=int, default=8)
    ap.add_argument("--inner-lr", type=float, default=1e-3); ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if "-" in a.subjects:
        lo, hi = a.subjects.split("-"); subs = range(int(lo), int(hi) + 1)
    else:
        subs = [int(s) for s in a.subjects.split(",")]
    dev = get_device(); print(f"[device] {dev} | {_SRC}")
    ch = PERIAURIC_LEFT if a.eeg_channels == "periauric" else a.eeg_channels.split(",")
    Xe, Xm, y, groups = load_tmspd_dataset(
        a.root, mode=a.mode, subjects=subs, eeg_channels=ch,
        tmin=a.tmin, tmax=a.tmax, eeg_fs_out=a.common_fs, emg_fs_out=a.common_fs,
        emg_offset_s=a.emg_offset_s)
    K = int(len(np.unique(y))); T = Xe.shape[2]
    chance = 1.0 / K
    print(f"[data] Xeeg{Xe.shape} Xemg{Xm.shape} y{y.shape} subjects={len(np.unique(groups))} classes={K} T={T}")
    if len(np.unique(groups)) < 2:
        raise SystemExit("need >=2 subjects for LOSO")

    fA, pA, fB, pB, bn = run(Xe, Xm, y, groups, modality=a.modality, methods=a.methods, K=K, T=T, dev=dev,
                             epochs=a.epochs, lr=a.lr, bs=a.bs, lam=a.lambda_mmd, kappa=a.kappa_dape,
                             dape_k=a.dape_k, shots=a.shots, peft_r=a.peft_r, peft_alpha=a.peft_alpha,
                             ea_modality=a.ea_modality, ea_reg=a.ea_reg, seed=a.seed, verbose=not a.quiet,
                             meta_iters=a.meta_iters, inner_steps=a.inner_steps, inner_lr=a.inner_lr, meta_lr=a.meta_lr,
                             patience=a.patience, min_epochs=a.min_epochs, val_frac=a.val_frac)
    report(fA, pA, fB, pB, K=K, chance=chance, methods=a.methods, modality=a.modality, shots=a.shots, bn_seen=bn)


if __name__ == "__main__":
    main()
