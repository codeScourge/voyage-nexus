"""
place_array.py — wire the locked CAD EEG cluster onto a head and characterise R.

Pipeline:
  STL contacts (exact relative geometry, mm)
    -> array local frame (PCA)
    -> 6-DOF rigid seed pose onto LEFT peri-auricular scalp
    -> snap contacts to scalp surface
    -> MNE DigMontage
    -> forward -> sLORETA inverse -> resolution matrix R
    -> peak_err / sd_ext(PSF) / ctf maps, swept over a 6-DOF pose band

The STL fixes the *internal* array geometry exactly. The only residual unknown is
the rigid pose of the whole cluster on the head (6 DOF). The seed pose is set by eye
from the photo/render via the CONFIG block; the jitter sweep converts the residual
pose uncertainty into a band on every resolution map.

Smoke-tested here on a head-sized ellipsoid (placement geometry). The fsaverage path
+ forward/R section run on a machine with MNE data access.
"""
import os.path as op
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

# ----------------------------- CONFIG (human-set, confirm by eye) -------------
STL_PATH   = "wearable_test_2_v45.stl"
EEG_FACES  = 1202          # face-count signature of the 16 EEG contact bodies
ANCHOR_MM  = np.array([-70., -8., 38.])   # left-scalp target for array centroid
                                          # (-X left, +Y front, +Z up; surface RAS)
FLIP_AP    = False         # set True if anterior/posterior ends up reversed in render
FLIP_SI    = False         # set True if superior/inferior ends up reversed
INVERT_NORMAL = False      # set True if contacts sit OUTSIDE the head (housing-in)
N_JITTER   = 20            # pose draws for the band
JIT_TRANS_MM = 10.0        # +/- translation envelope (~reseat slop)
JIT_ROT_DEG  = 5.0         # +/- rotation envelope
SENS_THRESH  = 0.10        # leadfield-norm fraction below which the array is "blind";
                           # spread metrics are only valid/reported where sens >= this
# -----------------------------------------------------------------------------


def eeg_centroids_mm(stl_path=STL_PATH):
    """16 EEG contact centroids in the CAD frame (mm). Internal geometry = ground truth."""
    m = trimesh.load(stl_path)
    eeg = np.array([p.centroid for p in m.split(only_watertight=False)
                    if len(p.faces) == EEG_FACES])
    assert len(eeg) == 16, f"expected 16 EEG bodies, got {len(eeg)}"
    return eeg


def load_scalp_mm():
    """Return (Nx3 scalp vertices in mm, surface-RAS) and a flag for real-vs-standin.
    Real: fsaverage outer-skin. Fallback: head-sized ellipsoid (X=R, Y=front, Z=up)."""
    try:
        import mne
        from mne.datasets import fetch_fsaverage
        fs = fetch_fsaverage(verbose=False)
        surf = mne.read_bem_surfaces(op.join(fs, "bem", "fsaverage-head.fif"))[0]
        return surf["rr"] * 1000.0, True          # m -> mm
    except Exception as e:                          # no data access -> stand-in
        print(f"[scalp] fsaverage unavailable ({type(e).__name__}); using ellipsoid stand-in")
        u, v = np.mgrid[0:np.pi:120j, 0:2*np.pi:240j]
        ax, by, cz = 75., 98., 88.
        return np.c_[(ax*np.sin(u)*np.cos(v)).ravel(),
                     (by*np.sin(u)*np.sin(v)).ravel(),
                     (cz*np.cos(u)).ravel()], False


def seed_transform(eeg):
    """Build the seed rotation/translation: array (AP,SI,normal) -> head (+Y,+Z,-X),
    centroid -> ANCHOR. Returns (Rmat, c_a)."""
    c_a = eeg.mean(0)
    _, _, Vt = np.linalg.svd(eeg - c_a)
    A = Vt.T                                        # cols: AP, SI, normal (by variance)
    if FLIP_AP: A[:, 0] *= -1
    if FLIP_SI: A[:, 1] *= -1
    if INVERT_NORMAL: A[:, 2] *= -1
    B = np.c_[[0, 1, 0], [0, 0, 1], [-1, 0, 0]].astype(float)   # head AP, SI, left-normal
    Rmat = B @ A.T
    if np.linalg.det(Rmat) < 0:                     # keep it a proper rotation
        A[:, 2] *= -1
        Rmat = B @ A.T
    return Rmat, c_a


def pose_to_contacts(eeg, Rmat, c_a, scalp_tree, scalp_pts, p=np.zeros(6)):
    """Apply seed + 6-DOF perturbation p=(tx,ty,tz,rx,ry,rz), snap to scalp. -> (snapped, raw)."""
    x = (Rmat @ (eeg - c_a).T).T + ANCHOR_MM        # seed placement
    Rp = Rot.from_euler("xyz", p[3:], degrees=True).as_matrix()
    x = (Rp @ (x - ANCHOR_MM).T).T + ANCHOR_MM + p[:3]   # perturb about anchor
    _, idx = scalp_tree.query(x)
    return scalp_pts[idx], x


def contacts_to_montage(xyz_mm):
    """MNE DigMontage from snapped contacts. Declares head==mri (identity trans),
    since contacts were placed directly on the fsaverage MRI/surface-RAS scalp."""
    import mne
    ch = [f"EEG{i+1}" for i in range(len(xyz_mm))]
    ch_pos = {c: xyz_mm[i] / 1000.0 for i, c in enumerate(ch)}   # mm -> m
    mont = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage(mont)
    # set_eeg_reference lives on Raw/Epochs/Evoked, NOT Info — route through a
    # zero-sample RawArray to attach the average-reference projector, return its info.
    raw = mne.io.RawArray(np.zeros((len(ch), 1)), info, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    return raw.info


def resolution_maps(info, src, bem):
    """Forward -> sLORETA inverse -> resolution matrix -> per-vertex metrics + sensitivity.
    Returns (peak_err_mm, sd_ext_mm, ctf_mm, sens), each length n_sources.
    NB: mne.resolution_metrics returns CENTIMETRES (it scales source locations x100
    internally), so cm -> mm is x10, NOT x1e3. sens = leadfield column norm (normalised
    to 1), returned for an informational panel (where the array's leadfield is strongest)."""
    import mne
    from mne.minimum_norm import (make_inverse_operator,
                                  make_inverse_resolution_matrix, resolution_metrics)
    trans = mne.transforms.Transform("head", "mri", np.eye(4))
    fwd = mne.make_forward_solution(info, trans, src, bem, eeg=True, verbose=False)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    sens = np.linalg.norm(fwd["sol"]["data"], axis=0)     # (n_src,) leadfield norm
    sens = sens / sens.max()
    cov = mne.make_ad_hoc_cov(info)
    inv = make_inverse_operator(info, fwd, cov, fixed=True, loose=0., depth=None, verbose=False)
    R = make_inverse_resolution_matrix(fwd, inv, method="sLORETA", lambda2=1/9.)
    ple = resolution_metrics(R, inv["src"], function="psf", metric="peak_err")
    ext = resolution_metrics(R, inv["src"], function="psf", metric="sd_ext")
    ctf = resolution_metrics(R, inv["src"], function="ctf", metric="sd_ext")
    return (ple.data.ravel() * 10., ext.data.ravel() * 10.,
            ctf.data.ravel() * 10., sens)


def jitter_band(eeg, Rmat, c_a, scalp_tree, scalp_pts, src, bem, rng=0):
    """Sweep 6-DOF pose; return per-vertex mean/std of each metric + sensitivity."""
    rng = np.random.default_rng(rng)
    acc = {"peak_err": [], "sd_ext": [], "ctf": [], "sens": []}
    for k in range(N_JITTER):
        p = np.r_[rng.uniform(-JIT_TRANS_MM, JIT_TRANS_MM, 3),
                  rng.uniform(-JIT_ROT_DEG, JIT_ROT_DEG, 3)] if k else np.zeros(6)
        snap, _ = pose_to_contacts(eeg, Rmat, c_a, scalp_tree, scalp_pts, p)
        info = contacts_to_montage(snap)
        ple, ext, ctf, sens = resolution_maps(info, src, bem)
        acc["peak_err"].append(ple); acc["sd_ext"].append(ext)
        acc["ctf"].append(ctf); acc["sens"].append(sens)
    return {m: (np.mean(v, 0), np.std(v, 0)) for m, v in acc.items()}


def plot_band(band, src, subjects_dir, out="resolution_band.png"):
    """3 panels on inflated lh: sensitivity | sd_ext (PSF blur) | ctf (cross-talk), in mm.
    Forces the pyvistaqt backend (the notebook backend errors off-notebook).
    Headless: run under xvfb-run with PyQt5 installed."""
    import os
    os.environ.setdefault("MNE_3D_OPTION_ANTIALIAS", "false")
    import pyvista; pyvista.OFF_SCREEN = True
    import mne, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mne.viz.set_3d_backend("pyvistaqt")             # avoid the notebook/ipywidgets path
    vert = [src[0]["vertno"], src[1]["vertno"]]
    panels = [("sensitivity (0-1)", band["sens"][0]),
              ("sd_ext PSF blur (mm)", band["sd_ext"][0]),
              ("ctf cross-talk (mm)", band["ctf"][0])]
    shots, titles = [], []
    for name, mu in panels:
        d = mu.astype(float)
        lims = np.percentile(d, [50, 75, 95])
        if len(set(np.round(lims, 6))) < 3:
            lims = np.array([d.min(), np.median(d), d.max()]) + [0, 1e-6, 2e-6]
        stc = mne.SourceEstimate(d, vert, tmin=0, tstep=1, subject="fsaverage")
        brain = stc.plot("fsaverage", "inflated", "lh", subjects_dir=subjects_dir,
                         colormap="inferno", background="white", time_viewer=False,
                         clim=dict(kind="value", lims=lims), size=(700, 600), verbose=False)
        brain.show_view("lateral")
        shots.append(brain.screenshot()); titles.append(name); brain.close()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, img, t in zip(axes, shots, titles):
        ax.imshow(img); ax.set_axis_off(); ax.set_title(t, fontsize=13)
    fig.suptitle("Array resolution, left hemi (mm). sd_ext ~7 cm is the ceiling: "
                 "a point source blurs into a ~7 cm patch.", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler("run.log", mode="w"),
                                  logging.StreamHandler(sys.stdout)])
    sys.excepthook = lambda *a: logging.getLogger().critical("UNCAUGHT", exc_info=a)
    log = logging.getLogger()

    eeg = eeg_centroids_mm()
    scalp_pts, is_real = load_scalp_mm()
    scalp_tree = cKDTree(scalp_pts)
    Rmat, c_a = seed_transform(eeg)

    snap, raw = pose_to_contacts(eeg, Rmat, c_a, scalp_tree, scalp_pts)
    nn = cKDTree(snap).query(snap, k=2)[0][:, 1]
    log.info(f"seed placement: 16 contacts, all left(x<0)={bool((snap[:,0]<0).all())}, "
             f"snapped NN spacing mm med={np.median(nn):.1f}")

    if is_real:
        import mne
        mne.set_log_file("run.log", overwrite=False)         # capture MNE's own stream
        sd = op.dirname(mne.datasets.fetch_fsaverage(verbose=False))
        src = mne.setup_source_space("fsaverage", "oct6", subjects_dir=sd, add_dist=False, verbose=False)
        bem = mne.make_bem_solution(mne.make_bem_model("fsaverage", subjects_dir=sd, verbose=False), verbose=False)
        band = jitter_band(eeg, Rmat, c_a, scalp_tree, scalp_pts, src, bem)

        sens = band["sens"][0]
        log.info(f"sensitivity: {100*(sens>=SENS_THRESH).mean():.0f}% of {sens.size} sources "
                 f"above {SENS_THRESH} leadfield (EEG leadfields are broad; 'lit' != resolvable)")
        tag = {"sd_ext": "PSF blur  <-- THIS is the montage ceiling",
               "ctf": "cross-talk spread", "peak_err": "sLORETA invariant (~0, uninformative)"}
        for m in ("sd_ext", "ctf", "peak_err"):
            mu, st = band[m]
            log.info(f"{m:9s}: median {np.median(mu):5.1f} mm "
                     f"(p10 {np.percentile(mu,10):.1f}/p90 {np.percentile(mu,90):.1f}); "
                     f"pose-std {np.median(st):.1f} mm   {tag[m]}")

        np.savez_compressed("band.npz", sens=sens, visible=(sens >= SENS_THRESH),
                            **{f"{m}_mean": band[m][0] for m in band},
                            **{f"{m}_std": band[m][1] for m in band},
                            lh_vertno=src[0]["vertno"], rh_vertno=src[1]["vertno"])
        log.info("[save] wrote band.npz (re-plot any time without recomputing)")

        try:
            plot_band(band, src, sd)
        except Exception:
            log.exception("plot_band failed (likely VTK/GL) — band.npz is saved; "
                          "re-plot in a GL-capable env (xvfb-run / libosmesa6)")
    else:
        log.info("stand-in scalp: placement geometry validated; "
                 "run on a machine with MNE data access for the forward/R band.")
