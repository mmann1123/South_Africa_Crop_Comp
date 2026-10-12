"""Regenerate the exact MiniRocket features of the reference runs, seed by seed.

Each seed's fitted MiniRocket + feature StandardScaler is loaded from
../models/minirocket_{level}/minirocket_{level}_seed_{s}.joblib and applied with the
same code (`train_minirocket.features`) to the same rows (pixel: the same
`cap_pixels(fid_train, 50, seed)` draw). `identity_check` proves it: the saved L2 head
applied to the regenerated in-region test features must reproduce the saved
proba_test_seed_{s}.npy.
"""
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PARENT)

import common  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_minirocket as tm  # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
BATCH = tm.BATCH


def seed_model_path(level, seed):
    return os.path.join(common.MODELS_DIR, f"minirocket_{level}", f"minirocket_{level}_seed_{seed}.joblib")


def load_seed_model(level, seed):
    path = seed_model_path(level, seed)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    m = joblib.load(path)
    return m, {"source": os.path.relpath(path, common.REPO), "sha256": h}


def train_rows(data, level, seed):
    """Indices of the training rows the reference MiniRocket used for this seed."""
    if level == "pixel":
        return tm.cap_pixels(data["fid_train"], tm.PIXEL_CAP, seed)
    return np.arange(len(data["y_train"]))


MISMATCH_LOG = os.path.join(RESULTS_DIR, "transform_mismatches.log")


def checked_transform(m, X):
    """MiniRocket features for one batch, computed twice and required to agree.

    The transform is deterministic in isolation (10 passes x 300k rows, 0 differences),
    but inside the training processes it twice produced corrupted features (README).
    On disagreement a third pass is made, the two agreeing copies are kept, and the
    event is logged; if no two copies agree the run stops.
    """
    a = tm.features(m["minirocket"], m["feature_scaler"], X)
    b = tm.features(m["minirocket"], m["feature_scaler"], X)
    if np.array_equal(a, b):
        return a
    c = tm.features(m["minirocket"], m["feature_scaler"], X)
    n_bad = int(np.any(a != b, axis=1).sum())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MISMATCH_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()} rows={len(X)} "
                f"rows_differing={n_bad} a==c={np.array_equal(a, c)} b==c={np.array_equal(b, c)}\n")
    if np.array_equal(a, c) or np.array_equal(b, c):
        return c
    raise RuntimeError("MiniRocket transform gave three different results for one batch")


def features(m, X, dtype=np.float32):
    """Standardised MiniRocket features for X, computed in checked batches."""
    out = np.empty((len(X), len(m["feature_scaler"].mean_)), dtype=dtype)
    for i in range(0, len(X), BATCH):
        out[i:i + BATCH] = checked_transform(m, X[i:i + BATCH])
    return out


def iter_features(m, X, batch=BATCH):
    """Yield checked float32 feature batches without materialising all of X's features."""
    for i in range(0, len(X), batch):
        yield checked_transform(m, X[i:i + batch])


def identity_check(m, data, level, seed, atol=1e-5):
    """Saved L2 head on regenerated test features must reproduce the saved test probabilities."""
    torch.set_num_threads(tm.TORCH_THREADS)
    head = tm.LogRegHead(m["C"])
    head.W, head.b = torch.from_numpy(m["W"]), torch.from_numpy(m["b"])
    p = np.concatenate([head.predict_proba(F) for F in iter_features(m, data["X_test"])])
    saved = np.load(os.path.join(common.MODELS_DIR, f"minirocket_{level}", f"proba_test_seed_{seed}.npy"))
    max_diff = float(np.abs(p - saved).max())
    if not max_diff <= atol:
        raise AssertionError(f"MiniRocket features differ from reference ({level}, seed {seed}): "
                             f"max |dp| = {max_diff:.2e}")
    print(f"Feature identity check ({level}, seed {seed}): max |dp| = {max_diff:.1e}  OK")
    return max_diff
