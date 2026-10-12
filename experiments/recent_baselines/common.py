"""Shared data, split and provenance helpers for the R2-Q3a recent baselines.

Every data step mirrors the published L-TAE-S scripts line for line
(experiments/field_reduction/train_ltae_sparse_{field,pixel}.py for training,
predict_oos.predict_ltae_sparse_{field,pixel} for the holdout), and reuses their
helpers rather than re-implementing them. `check_against_published` then asserts
that the split sizes, feature columns, label encoder and fitted scaler equal those
of the published L-TAE-S models, so the new baselines see exactly the same inputs.

Read-only with respect to everything outside experiments/recent_baselines/.
"""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIELD_REDUCTION = os.path.join(REPO, "experiments", "field_reduction")
sys.path.insert(0, FIELD_REDUCTION)

from experiment_config import (  # noqa: E402
    MERGED_DL_PATH, MERGED_DL_TEST_PATH, SEEDS_ENSEMBLE, SUBSAMPLE_SEED, TEST_LABELS_GEOJSON,
)
from models_arch import get_chrono_feature_cols, T_SEQ, N_BANDS  # noqa: E402
from subsample import get_fid_split_dl, subsample_train_fids  # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
PUBLISHED_LTAE_S = {
    "field": os.path.join(FIELD_REDUCTION, "models", "ltae_sparse_field", "frac_1.00"),
    "pixel": os.path.join(FIELD_REDUCTION, "models", "ltae_sparse_pixel", "frac_1.00"),
}
FRACTION = 1.0  # full training set, as in the published L-TAE-S runs


# ----------------------------------------------------------------- determinism

def set_determinism():
    """Deterministic CUDA kernels. Call once, before any CUDA work."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------- data

def load_training_data(level):
    """Return the train / val / in-region-test arrays for `level` in {'field', 'pixel'}.

    Mirrors train_ltae_sparse_{level}.py: chronological 60 feature columns, all-NaN
    columns dropped, fillna(0); for 'field' the pixels are averaged per fid (label =
    modal crop) BEFORE the FID split, exactly as published; FID-wise 64/16/20 split
    (random_state=42); StandardScaler fitted on training rows only; nan_to_num.
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = pd.read_parquet(MERGED_DL_PATH)
    feature_cols = get_chrono_feature_cols(df)
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    df[feature_cols] = df[feature_cols].fillna(0)

    if level == "field":
        agg = {c: "mean" for c in feature_cols}
        agg["crop_name"] = lambda x: x.mode()[0]
        df = df.groupby("fid").agg(agg).reset_index()
        df[feature_cols] = df[feature_cols].fillna(0)
    else:
        df = df[["fid", "crop_name"] + feature_cols]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["crop_name"])

    train_fids, val_fids, test_fids = get_fid_split_dl(df)
    train_fids = subsample_train_fids(df, train_fids, FRACTION, seed=SUBSAMPLE_SEED)
    masks = {"train": df["fid"].isin(train_fids), "val": df["fid"].isin(val_fids),
             "test": df["fid"].isin(test_fids)}

    scaler = StandardScaler()
    out = {"feature_cols": feature_cols, "le": le, "scaler": scaler, "level": level}
    for split in ["train", "val", "test"]:
        X = df.loc[masks[split], feature_cols].values
        X = scaler.fit_transform(X) if split == "train" else scaler.transform(X)
        out[f"X_{split}"] = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out[f"y_{split}"] = df.loc[masks[split], "label"].values
        out[f"fid_{split}"] = df.loc[masks[split], "fid"].values
    out["n_fields"] = {s: int(len(np.unique(out[f"fid_{s}"]))) for s in ["train", "val", "test"]}
    out["n_rows"] = {s: int(masks[s].sum()) for s in ["train", "val", "test"]}
    del df
    return out


def load_holdout(level, feature_cols, scaler):
    """Holdout tile, preprocessed exactly as predict_oos.predict_ltae_sparse_{level}."""
    df = pd.read_parquet(MERGED_DL_TEST_PATH)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df[feature_cols] = df[feature_cols].fillna(0)
    if level == "field":
        df = df.groupby("fid")[feature_cols].mean().reset_index()
    X = scaler.transform(df[feature_cols].values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, df["fid"].values


def to_sequences(X):
    """(N, 60) chronological month-major features -> (N, T=10, C=6)."""
    return X.reshape(-1, T_SEQ, N_BANDS)


def field_majority_vote(fids, preds):
    """Per-field majority vote, as in predict_oos (Counter.most_common)."""
    s = pd.DataFrame({"fid": fids, "pred": preds}).groupby("fid")["pred"].agg(
        lambda x: Counter(x).most_common(1)[0][0])
    return s.index.values, s.values


def write_predictions(fids, labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"fid": fids, "crop_name": labels}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------- published check

def check_against_published(data):
    """Assert this run sees exactly the inputs of the published L-TAE-S model."""
    level = data["level"]
    pub_dir = PUBLISHED_LTAE_S[level]
    meta = json.load(open(os.path.join(pub_dir, "metadata.json")))
    pub_scaler = joblib.load(os.path.join(pub_dir, "scaler.joblib"))
    pub_cols = joblib.load(os.path.join(pub_dir, "feature_columns.joblib"))
    pub_le = joblib.load(os.path.join(pub_dir, "label_encoder.joblib"))

    checks = {
        "feature_columns": list(pub_cols) == list(data["feature_cols"]),
        "label_classes": list(pub_le.classes_) == list(data["le"].classes_),
        "train_fields": meta["train_fields"] == data["n_fields"]["train"],
        "test_fields": meta["test_fields"] == data["n_fields"]["test"],
        "scaler_mean": bool(np.allclose(pub_scaler.mean_, data["scaler"].mean_, rtol=0, atol=1e-12)),
        "scaler_scale": bool(np.allclose(pub_scaler.scale_, data["scaler"].scale_, rtol=0, atol=1e-12)),
    }
    if level == "field":
        checks["val_fields"] = meta["val_fields"] == data["n_fields"]["val"]
    else:
        checks["train_pixels"] = meta["train_pixels"] == data["n_rows"]["train"]
        checks["val_pixels"] = meta["val_pixels"] == data["n_rows"]["val"]
        checks["test_pixels"] = meta["test_pixels"] == data["n_rows"]["test"]
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Inputs differ from published L-TAE-S ({level}): {failed}")
    print(f"Input check vs published L-TAE-S ({level}): all {len(checks)} checks passed")
    return checks


# ---------------------------------------------------------------- provenance

def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def data_hashes():
    """SHA-256 of the input parquets, cached by (size, mtime) to avoid rehashing 3 GB."""
    cache = os.path.join(RESULTS_DIR, "data_hashes.json")
    stored = json.load(open(cache)) if os.path.exists(cache) else {}
    out = {}
    for p in [MERGED_DL_PATH, MERGED_DL_TEST_PATH]:
        st = os.stat(p)
        key = os.path.relpath(p, REPO)
        rec = stored.get(key)
        if not rec or rec["size"] != st.st_size or rec["mtime"] != st.st_mtime:
            rec = {"size": st.st_size, "mtime": st.st_mtime, "sha256": _sha256(p)}
        out[key] = rec
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(cache, "w"), indent=2)
    return {k: v["sha256"] for k, v in out.items()}


def provenance():
    import sklearn
    import torch

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", REPO, *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "torch": torch.__version__}
    try:
        import aeon
        versions["aeon"] = aeon.__version__
    except ImportError:
        pass
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python_executable": sys.executable,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_sha256": data_hashes(),
    }


def write_metadata(out_dir, payload):
    os.makedirs(out_dir, exist_ok=True)
    payload = {**payload, "provenance": provenance()}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def in_region_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    return {"accuracy": round(accuracy_score(y_true, y_pred), 6),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 6),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 6),
            "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 6)}


def in_region_field_metrics(level, fids, y_true, y_pred):
    """In-region score on the internal 20% split, at field level (pixel: majority vote)."""
    if level == "pixel":
        _, t = field_majority_vote(fids, y_true)
        _, p = field_majority_vote(fids, y_pred)
        return in_region_metrics(t, p)
    return in_region_metrics(y_true, y_pred)
