"""replot_band.py — render band.npz onto the fsaverage inflated left hemisphere.

Pure pyvista off_screen (software GL via the bundled VTK) — NO Qt, NO MNE Brain,
NO notebook backend. This deliberately avoids the entire xcb/pyvistaqt failure class.
Runs headless without xvfb (software fallback); xvfb-run is fine too but not required.

Usage:  python replot_band.py [band.npz] [out.png]
"""
import os, sys
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pyvista as pv
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True


def fsaverage_lh_inflated():
    """Locate fsaverage (cached; no download) and read lh.inflated -> (coords_mm, faces)."""
    import mne
    fs = mne.datasets.fetch_fsaverage(verbose=False)          # cached path, no re-download
    subjects_dir = os.path.dirname(fs)
    surf = os.path.join(subjects_dir, "fsaverage", "surf", "lh.inflated")
    coords, faces = nib.freesurfer.read_geometry(surf)
    return coords.astype(float), faces.astype(int)


def main(npz="band.npz", out="resolution_band.png"):
    d = np.load(npz)
    lhv = d["lh_vertno"]; nlh = len(lhv)
    coords, faces = fsaverage_lh_inflated()
    faces_pv = np.hstack([np.full((len(faces), 1), 3, int), faces]).ravel()

    # spread each oct6 source value over the full surface via nearest source vertex
    nn = cKDTree(coords[lhv]).query(coords)[1]

    panels = [("sensitivity (0-1)",     d["sens_mean"][:nlh]),
              ("sd_ext PSF blur (mm)",  d["sd_ext_mean"][:nlh]),
              ("ctf cross-talk (mm)",   d["ctf_mean"][:nlh])]

    focal = coords.mean(0)
    cam = [(focal[0] - 400, focal[1], focal[2]), tuple(focal), (0, 0, 1)]  # lh lateral

    shots, titles = [], []
    for name, vals in panels:
        full = vals[nn]
        mesh = pv.PolyData(coords, faces_pv); mesh["v"] = full
        lo, hi = np.percentile(vals, [5, 95])
        if hi <= lo:
            hi = lo + 1e-6
        p = pv.Plotter(off_screen=True, window_size=(700, 600))
        p.background_color = "white"
        p.add_mesh(mesh, scalars="v", cmap="inferno", clim=(lo, hi),
                   smooth_shading=True, scalar_bar_args=dict(title=name, color="black"))
        p.camera_position = cam
        shots.append(p.screenshot(return_img=True)); titles.append(name); p.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, img, t in zip(axes, shots, titles):
        ax.imshow(img); ax.set_axis_off(); ax.set_title(t, fontsize=13)
    fig.suptitle("Array resolution, fsaverage lh (mm) — Qt-free render from band.npz. "
                 "sd_ext ~7 cm = the montage ceiling.", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "band.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "resolution_band.png"
    main(npz, out)
