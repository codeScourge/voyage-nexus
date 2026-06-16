"""
persistence.py
==============
Save and reload the artifacts produced by hpo.py's outer refit, so a tuning run
leaves you (a) a loadable best model per outer fold and (b) the outer-fold
predictions + true labels needed to plot confusions WITHOUT refitting.

Per outer fold, save_fold writes into <save_dir>/:
  fold{k}_predictions.npz   y_true, y_pred, groups, label_names   (always; numpy)
  fold{k}_model.joblib      the whole fitted estimator             (when picklable)
  fold{k}_model.state_dict.pt + fold{k}_recipe.json
                            Torch fallback: nets whose architecture classes are
                            defined in a builder closure can't be plain-pickled,
                            so we store weights + a rebuild recipe instead.

load_fold(save_dir, k) returns (estimator, data) where `estimator` is ready to
.predict() and `data` is the predictions dict. Reload order: joblib if present,
else rebuild the architecture via make_model + the model's builder and load the
state_dict.

Non-Torch estimators (dda; cspdnn / rusnac in their sklearn-fallback mode;
OvO3D-wrapped CSP with an nn.Sequential head) pickle directly via joblib and need
no recipe. The recipe path is exercised only by eegnet / deepconvnet / rusnac when
a real Torch CNN was trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _jsonable(obj):
    """Make hp / config JSON-serializable (tuples->lists, numpy->python)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_estimator(est, stub: str) -> dict:
    """Try joblib for the whole estimator; fall back to Torch state_dict."""
    info: dict = {}
    try:
        import joblib
        joblib.dump(est, stub + ".joblib")
        info["format"] = "joblib"
        info["model_file"] = stub + ".joblib"
        return info
    except Exception as e:                       # e.g. closure-defined nn.Module class
        # joblib.dump may leave a truncated file behind -> remove it so loaders
        # don't mistake it for a valid pickle.
        try:
            p = Path(stub + ".joblib")
            if p.exists():
                p.unlink()
        except Exception:
            pass
        info["format"] = "state_dict+recipe"
        info["joblib_error"] = str(e)[:200]
    try:
        import torch
        if hasattr(est, "model_") and isinstance(est.model_, torch.nn.Module):
            torch.save(est.model_.state_dict(), stub + ".state_dict.pt")
            info["model_file"] = stub + ".state_dict.pt"
    except Exception as e:                        # pragma: no cover
        info["state_dict_error"] = str(e)[:200]
    return info


def save_fold(save_dir, model_name, ofold, est, X_te, y_true, y_pred, groups,
              label_names, dims, hp, preproc_dict, win_sec, hop_sec,
              inner_score, outer_score) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stub = str(save_dir / f"fold{ofold}")

    np.savez(stub + "_predictions.npz",
             y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
             groups=np.asarray(groups),
             label_names=np.array(label_names, dtype=object))

    save_info = _save_estimator(est, stub + "_model")

    recipe = dict(
        model=model_name, ofold=int(ofold),
        n_channels=int(dims[0]), n_times=int(dims[1]), n_classes=int(dims[2]),
        hp=_jsonable(hp),
        classes=_jsonable(getattr(est, "classes_", np.arange(dims[2]))),
        preproc=_jsonable(preproc_dict), win_sec=float(win_sec), hop_sec=float(hop_sec),
        inner_score=float(inner_score), outer_balanced_acc=float(outer_score),
        save_info=save_info,
        predictions_file=stub + "_predictions.npz",
    )
    Path(stub + "_recipe.json").write_text(json.dumps(recipe, indent=2))
    return recipe


def load_fold(save_dir, ofold):
    """Return (estimator, predictions_dict). estimator is ready to .predict()."""
    save_dir = Path(save_dir)
    stub = str(save_dir / f"fold{ofold}")
    recipe = json.loads(Path(stub + "_recipe.json").read_text())
    data = dict(np.load(stub + "_predictions.npz", allow_pickle=True))

    joblib_path = Path(stub + "_model.joblib")
    if joblib_path.exists():
        try:
            import joblib
            return joblib.load(joblib_path), data
        except Exception:
            pass   # truncated/incompatible pickle -> fall through to rebuild

    # Torch rebuild path
    import torch
    from models import make_model
    r = recipe
    est = make_model(r["model"], r["n_channels"], r["n_times"], r["n_classes"], **r["hp"])
    sd_path = stub + "_model.state_dict.pt"
    if r["model"] in ("eegnet", "deepconvnet"):
        est.model_ = est.builder(est.n_channels, est.n_times, est.n_classes,
                                 **est.builder_kwargs)
        est.model_.load_state_dict(torch.load(sd_path, map_location="cpu"))
        est.classes_ = np.asarray(r["classes"]); est.device = "cpu"
    elif r["model"] == "rusnac":
        from grigore_rusnac import build_rusnac_cnn
        est.model_ = build_rusnac_cnn(r["n_channels"], r["n_classes"],
                                      tuple(est.conv_filters), est.dense,
                                      est.conv_act, est.dense_act)
        est.model_.load_state_dict(torch.load(sd_path, map_location="cpu"))
        est.classes_ = np.asarray(r["classes"]); est._backend = "torch"; est._dev = "cpu"
    else:
        raise RuntimeError(f"no joblib and no torch rebuild path for {r['model']}")
    return est, data


def confusion_from_fold(save_dir, ofold):
    """Convenience: confusion matrix + label names from saved predictions only."""
    from sklearn.metrics import confusion_matrix
    data = dict(np.load(Path(save_dir) / f"fold{ofold}_predictions.npz", allow_pickle=True))
    cm = confusion_matrix(data["y_true"], data["y_pred"])
    return cm, list(data["label_names"])


if __name__ == "__main__":
    # round-trip test on a non-torch estimator (dda) -- joblib path
    import numpy as np
    from models import make_model
    rng = np.random.default_rng(0)
    n, C, T = 40, 6, 200
    X = rng.standard_normal((n, C, T)).astype("float32"); y = (np.arange(n) % 2)
    est = make_model("dda", C, T, 2, delay_grid=tuple(range(1, 6))).fit(X, y)
    pred = est.predict(X)
    save_fold("/tmp/persist_test", "dda", 0, est, X, y, pred, np.arange(n),
              ["a", "b"], (C, T, 2), {"delay_grid": (1, 2, 3, 4, 5)},
              {"bp_low": 1.0}, 0.5, 0.25, 0.9, 0.88)
    est2, data = load_fold("/tmp/persist_test", 0)
    ok = np.array_equal(est2.predict(X), pred)
    cm, names = confusion_from_fold("/tmp/persist_test", 0)
    print("reloaded predict matches:", ok, "| confusion shape:", cm.shape, "| labels:", names)
    print("PERSISTENCE OK" if ok else "MISMATCH")
