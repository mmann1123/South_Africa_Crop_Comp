"""FID splitting and stratified subsampling for the field reduction experiment.

Replicates the exact train/val/test splits from each original training script,
then subsamples the train FIDs to a given fraction (stratified by crop class).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_fid_split_dl(df):
    """Replicate the DL FID split: test_size=0.2 twice → 64/16/20.

    Used by: TabNet pixel, L-TAE field, L-TAE pixel, XGBoost field.
    """
    fids = df['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)
    return train_fids, val_fids, test_fids


def get_fid_split_base_ml(df):
    """Replicate the base_ml FID split: test_size=0.2 then test_size=0.15 → 68/12/20.

    Used by: Base LightGBM pixel, Base LR pixel.
    """
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
