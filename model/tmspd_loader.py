"""
tmspd_loader.py
===============
Load one T-MSPD subject/mode (raw) into paired (Xeeg, Xemg, y).

FORMAT (verified on S01 silent speech):
  EEG : NeuroScan Curry 8 -> S01.cdt (+ .cdt.ceo events, .cdt.dpo params),
        1000 Hz, 64 EEG + 'Trigger'. The Curry trigger holds 750 clean events,
        codes 1..10 = word label, 75 each, fully interleaved.
  sEMG: BioSemi BDF -> data.bdf, 1000 Hz, 8ch (S01..S08; keep 1..6, drop 7,8).
  evt.bdf is CORRUPT for trials (1 Hz timing, 352 != 750 events, mismatched
  codes -- it is the audio-onset detection track). We DO NOT use it.

KEY DECISION: EEG and EMG are hardware-triggered to a shared timebase (durations
match to 0.1 s). So we take trial onsets+labels from the EEG Curry trigger and
epoch BOTH modalities at the same sample indices (+ optional --emg-offset). A
built-in check reports post/pre-onset EMG energy so you can confirm/measure the
offset empirically (strong on overt speech, weak on silent).

Returns Xeeg (N,Ceeg,Teeg) @ eeg_fs_out, Xemg (N,Cemg,Temg) @ emg_fs_out, y(N,).
"""
from __future__ import annotations
import numpy as np
import mne
mne.set_log_level("ERROR")

# Left unilateral peri-auricular arc -- the montage-honest approximation of your
# over-ear EEG rig (extend with F5,P5,PO7,CB1 if you want a wider ring).
PERIAURIC_LEFT = ["F7", "FT7", "T7", "TP7", "P7", "FC5", "C5", "CP5", "M1"]
EMG_KEEP = [0, 1, 2, 3, 4, 5]                     # S01..S06 (drop S07,S08)


def _epoch(data, onsets, t0, t1, fs):
    """data:(C,Ttot) -> (N,C,win) windows at onset+[t0,t1) sec; drop OOB."""
    a, b = int(round(t0 * fs)), int(round(t1 * fs))
    win = b - a
    out, keep = [], []
    for k, s in enumerate(onsets):
        i0 = s + a
        if i0 < 0 or i0 + win > data.shape[1]:
            continue
        out.append(data[:, i0:i0 + win]); keep.append(k)
    return np.stack(out).astype(np.float32), np.array(keep)


def _norm(X, mode):
    if mode == "none":
        return X
    Xc = X - X.mean(-1, keepdims=True)
    if mode == "pertrial":
        return Xc / (X.std(-1, keepdims=True) + 1e-6)
    if mode == "global":
        return Xc / (Xc.std() + 1e-6)
    raise ValueError(mode)


def load_tmspd_subject(eeg_cdt, emg_bdf, *, eeg_channels="periauric",
                       emg_channels=EMG_KEEP, tmin=-0.1, tmax=1.0,
                       eeg_fs_out=250, emg_fs_out=1000, emg_offset_s=0.0,
                       eeg_band=(1.0, 40.0), emg_band=(20.0, 450.0), notch=50.0,
                       norm_eeg="pertrial", norm_emg="global", verify=True):
    # ---- EEG: master events + labels ----
    eeg = mne.io.read_raw_curry(eeg_cdt, preload=False)
    fs_e = eeg.info["sfreq"]
    ev, eid = mne.events_from_annotations(eeg)
    inv = {v: k for k, v in eid.items()}
    labels = np.array([int(inv[c]) for c in ev[:, 2]]) - 1      # 0..9
    onsets_e = ev[:, 0].astype(int)                              # EEG samples
    eeg.pick([c for c in (PERIAURIC_LEFT if eeg_channels == "periauric"
              else eeg.ch_names[:64] if eeg_channels == "all" else eeg_channels)
              if c in eeg.ch_names])
    eeg.load_data()
    if eeg_band:
        eeg.filter(*eeg_band, verbose=False)
    if notch:
        eeg.notch_filter(notch, verbose=False)
    Xe, keep_e = _epoch(eeg.get_data(), onsets_e, tmin, tmax, fs_e)
    ye = labels[keep_e]

    # ---- EMG: epoch at the SAME (EEG) onsets, shared timebase ----
    emg = mne.io.read_raw_bdf(emg_bdf, preload=True)
    fs_m = emg.info["sfreq"]
    emg.pick([emg.ch_names[i] for i in emg_channels])
    Xm_full = emg.get_data()
    if emg_band:
        from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
        ny = fs_m / 2
        sos = butter(4, [emg_band[0] / ny, min(emg_band[1], ny - 1) / ny],
                     btype="band", output="sos")
        Xm_full = sosfiltfilt(sos, Xm_full, axis=-1)
        if notch:
            bn, an = iirnotch(notch / ny, 30)
            Xm_full = filtfilt(bn, an, Xm_full, axis=-1)
    off = int(round(emg_offset_s * fs_m))
    # EEG onset samples are in fs_e; convert to fs_m timebase (same here, 1k==1k)
    onsets_m = np.round(onsets_e * (fs_m / fs_e)).astype(int) + off
    Xm, keep_m = _epoch(Xm_full.astype(np.float32), onsets_m, tmin, tmax, fs_m)
    ym = labels[keep_m]

    # ---- align the two epoch sets to common kept trials ----
    common = np.intersect1d(keep_e, keep_m)
    ie = np.searchsorted(keep_e, common); im = np.searchsorted(keep_m, common)
    Xe, Xm, y = Xe[ie], Xm[im], labels[common]

    # ---- shared-timebase verification (post vs pre onset EMG energy) ----
    if verify:
        a = int(round(-tmin * fs_m))                # onset index within window
        pre = np.sqrt((Xm[..., :a] ** 2).mean()); post = np.sqrt((Xm[..., a:] ** 2).mean())
        print(f"  [verify] EMG energy post/pre-onset = {post/ (pre+1e-9):.3f} "
              f"(>1 => time-locked; strong on overt, weak on silent)")

    # ---- resample EEG to your 250 Hz branch (EMG already 1k) ----
    if eeg_fs_out and eeg_fs_out != fs_e:
        from scipy.signal import resample_poly
        g = np.gcd(int(eeg_fs_out), int(fs_e))
        Xe = resample_poly(Xe, int(eeg_fs_out) // g, int(fs_e) // g, axis=-1).astype(np.float32)
    Xe = _norm(Xe, norm_eeg); Xm = _norm(Xm, norm_emg)
    return Xe, Xm, y


def load_tmspd_dataset(root, mode="overt speech", subjects=range(1, 31), *,
                       cache=".cache_tmspd", **kw):
    """Loop subjects -> concat (Xeeg, Xemg, y, groups), cached. Path pattern:
      {root}/{mode}/S{nn}/EEG/S{nn}.cdt   and   {root}/{mode}/S{nn}/sEMG/data.bdf
    The fusion / eeg / emg runs all SHARE this cache (same windows, diff model).
    """
    import os, hashlib
    subs = list(subjects)
    ch = kw.get("eeg_channels", "periauric")
    chtag = "periauric" if ch == "periauric" else "-".join(ch) if isinstance(ch, (list, tuple)) else str(ch)
    key = hashlib.md5((f"{os.path.abspath(root)}|{mode}|{subs[0]}-{subs[-1]}|{chtag}|"
                       f"{kw.get('tmin')}|{kw.get('tmax')}|{kw.get('eeg_fs_out')}|"
                       f"{kw.get('emg_fs_out')}|{kw.get('emg_offset_s')}").encode()).hexdigest()[:10]
    os.makedirs(cache, exist_ok=True)
    cf = os.path.join(cache, f"tmspd_{mode.replace(' ', '_')}_{chtag}_{key}.npz")
    if os.path.exists(cf):
        z = np.load(cf); print(f"  [cache] {cf}")
        return z["Xe"], z["Xm"], z["y"], z["groups"]

    Xe_l, Xm_l, y_l, g_l = [], [], [], []
    for sid in subs:
        s = f"S{sid:02d}"
        eeg = os.path.join(root, mode, s, "EEG", f"{s}.cdt")
        emg = os.path.join(root, mode, s, "sEMG", "data.bdf")
        if not (os.path.exists(eeg) and os.path.exists(emg)):
            print(f"  [skip] {s}: missing files"); continue
        print(f"  [load] {s}")
        Xe, Xm, y = load_tmspd_subject(eeg, emg, verify=False, **kw)
        Xe_l.append(Xe); Xm_l.append(Xm); y_l.append(y)
        g_l.append(np.full(len(y), sid, dtype=int))
    if not Xe_l:
        raise FileNotFoundError(f"no subjects found under {root}/{mode}")
    Xe, Xm = np.concatenate(Xe_l), np.concatenate(Xm_l)
    y, groups = np.concatenate(y_l), np.concatenate(g_l)
    np.savez_compressed(cf, Xe=Xe, Xm=Xm, y=y, groups=groups)
    print(f"  [cache->] {cf}")
    return Xe, Xm, y, groups


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", required=True); ap.add_argument("--emg", required=True)
    ap.add_argument("--eeg-channels", default="periauric")
    ap.add_argument("--emg-offset-s", type=float, default=0.0)
    a = ap.parse_args()
    Xe, Xm, y = load_tmspd_subject(a.eeg, a.emg, eeg_channels=a.eeg_channels,
                                   emg_offset_s=a.emg_offset_s)
    print(f"  Xeeg {Xe.shape} @250Hz  Xemg {Xm.shape} @1kHz  y{y.shape} "
          f"classes={sorted(set(y.tolist()))}")
    print(f"  per-class counts: {np.bincount(y).tolist()}")
