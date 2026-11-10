"""FID splitting and stratified subsampling for the field reduction experiment.

Replicates the exact train/val/test splits from each original training script,
then subsamples the train FIDs to a given fraction (stratified by crop class).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Paired mode. When the environment variable PAIRED_FID_FILE names a JSON file written by
# experiments/missingness/learning_curves/paired/make_splits.py, every split function below returns
# that file's train/val/test FIDs and subsample_train_fids returns its stored draw for
# (fraction, seed), so different models train on identical fields and draws pair across models.
# Unset (the default), nothing changes and every published split is reproduced.
_PAIRED = None


def paired_fid_file():
    """Load (once) the file named by PAIRED_FID_FILE, or return None when unset."""
    global _PAIRED
    path = os.environ.get("PAIRED_FID_FILE")
    if not path:
        return None
    if _PAIRED is None or _PAIRED["_path"] != path:
        with open(path) as f:
            d = json.load(f)
        for k in ("train", "val", "test", "draws"):
            if k not in d:
                raise ValueError(f"{path}: missing key {k!r}")
        d["_path"] = path
        _PAIRED = d
        print(f"[paired] splits and draws from {path}: {len(d['train'])} train, {len(d['val'])} val, "
              f"{len(d['test'])} test FIDs; fractions {sorted(d['draws'])}", flush=True)
    return _PAIRED


def _paired_split(df):
    d = paired_fid_file()
    if d is None:
        return None
    have = set(df["fid"].unique())
    tr, va, te = (np.array([f for f in d[k] if f in have]) for k in ("train", "val", "test"))
    lost = sum(len(d[k]) for k in ("train", "val", "test")) - len(tr) - len(va) - len(te)
    if lost:
        print(f"[paired] {lost} FIDs of the shared split are absent from this dataset", flush=True)
    return tr, va, te


def get_fid_split_dl(df):
    """Replicate the DL FID split: test_size=0.2 twice → 64/16/20.

    Used by: TabNet pixel, L-TAE field, L-TAE pixel, XGBoost field.
    """
    paired = _paired_split(df)
    if paired is not None:
        return paired
    fids = df['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)
    return train_fids, val_fids, test_fids


def get_fid_split_base_ml(df):
    """Replicate the base_ml FID split: test_size=0.2 then test_size=0.15 → 68/12/20.

    Used by: Base LightGBM pixel, Base LR pixel.
    """
    paired = _paired_split(df)
    if paired is not None:
        return paired
    fids = df['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.15, random_state=42)
    return train_fids, val_fids, test_fids


def subsample_train_fids(df, train_fids, fraction, seed=42):
    """Subsample train FIDs to a given fraction, stratified by crop class.

    Parameters
    ----------
    df : DataFrame
        Must contain 'fid' and 'crop_name' columns.
    train_fids : array-like
        Full set of training FIDs.
    fraction : float
        Fraction of training FIDs to keep (e.g. 0.25, 0.50, 0.75).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sub_fids : ndarray
        Subsampled training FIDs.
    """
    d = paired_fid_file()
    if d is not None:
        key = f"{fraction:.2f}"
        if key not in d["draws"] or str(seed) not in d["draws"][key]:
            raise KeyError(f"paired FID file has no draw for fraction {key}, seed {seed}")
        have = set(np.asarray(train_fids).tolist())
        sub = [f for f in d["draws"][key][str(seed)] if f in have]
        if len(sub) != len(d["draws"][key][str(seed)]):
            print(f"[paired] {len(d['draws'][key][str(seed)]) - len(sub)} FIDs of draw {key}/{seed} absent", flush=True)
        return np.array(sub)

    if fraction >= 1.0:
        return np.array(train_fids)

    # Build a FID → crop_name mapping (mode per field)
    fid_crop = (
        df[df['fid'].isin(train_fids)]
        .groupby('fid')['crop_name']
        .agg(lambda x: x.mode()[0])
    )

    n_keep = max(1, int(len(train_fids) * fraction))

    # Try stratified sampling
    try:
        _, sub_fids = train_test_split(
            fid_crop.index.values,
            test_size=n_keep,
            stratify=fid_crop.values,
            random_state=seed,
        )
    except ValueError:
        # Fallback: non-stratified if any class has too few samples
        rng = np.random.RandomState(seed)
        sub_fids = rng.choice(fid_crop.index.values, size=n_keep, replace=False)

    return np.array(sub_fids)
