"""Pure functions for the TimeMatch cloud-removal re-run (no torch, no Earth Engine).

Date-set conventions follow the authors' code: a parcel is a (T, C, S) pixel set,
the temporal encoder adds a sinusoidal day-of-year embedding of `positions + shift`
to each time step, and there is no padding mask over time, so a parcel with fewer
dates is simply a shorter sequence. Parcels are batched by identical (T, S)."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

MIN_KEEP = 3  # never leave a parcel with fewer than this many dates


def make_folds(n_source, n_target, source, target, seed=1, num_folds=3, val_ratio=0.1, test_ratio=0.2):
    """Verbatim logic of train.create_train_val_test_folds, including the global
    `random` stream: seed once, then per fold shuffle the source and the target
    index lists in that order. Returns a list of {dataset: {train,val,test}}."""
    random.seed(seed)
    sizes = {source: n_source, target: n_target}
    folds = []
    for _ in range(num_folds):
        splits = {}
        for ds in [source, target]:
            idx = list(range(sizes[ds]))
            n = len(idx)
            n_test, n_val = int(test_ratio * n), int(val_ratio * n)
            n_train = n - n_test - n_val
            random.shuffle(idx)
            splits[ds] = {
                "train": set(idx[:n_train]),
                "val": set(idx[n_train : n_train + n_val]),
                "test": set(idx[-n_test:]),
            }
        folds.append(splits)
    return folds


def cloud_flags(frac_cloudy: np.ndarray, min_pixel_frac: float = 0.5) -> np.ndarray:
    """(P, T) fraction of a parcel's pixels under Cloud Score+ < threshold -> bool.
    A parcel-date is cloudy when at least `min_pixel_frac` of its pixels are
    cloudy. NaN (no same-day image over the parcel) counts as not cloudy."""
    f = np.asarray(frac_cloudy, dtype=float)
    return np.where(np.isnan(f), False, f >= min_pixel_frac)


def n_to_remove(flags: np.ndarray, min_keep: int = MIN_KEEP) -> np.ndarray:
    """Per parcel, how many dates cloud removal drops: all flagged ones, but at
    most T - min_keep (the same count is removed at random in the control)."""
    T = flags.shape[1]
    return np.minimum(flags.sum(axis=1), max(T - min_keep, 0)).astype(int)


def keep_mask_cloud(flags: np.ndarray, score: np.ndarray, min_keep: int = MIN_KEEP) -> np.ndarray:
    """(P, T) bool keep mask: drop flagged dates; when a parcel has more than
    T - min_keep flagged dates, drop only the T - min_keep with highest score."""
    P, T = flags.shape
    keep = ~flags
    nrem = n_to_remove(flags, min_keep)
    over = flags.sum(axis=1) > nrem
    sc = np.where(np.isnan(score), -np.inf, score)
    for i in np.where(over)[0]:
        order = np.lexsort((np.arange(T), -sc[i]))  # highest score first, ties by index
        keep[i] = True
        keep[i, order[: nrem[i]]] = False
    return keep


def keep_mask_random(nrem: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """(P, T) bool keep mask removing nrem[i] uniformly random dates of parcel i."""
    keep = np.ones((len(nrem), T), dtype=bool)
    for i, k in enumerate(nrem):
        if k > 0:
            keep[i, rng.choice(T, size=int(k), replace=False)] = False
    return keep


def group_by_shape(t_counts, s_counts, batch_size):
    """Batches of parcel indices with identical (T, S), at most batch_size each
    (mirrors dataset.GroupByShapesBatchSampler)."""
    groups = defaultdict(list)
    for i, (t, s) in enumerate(zip(t_counts, s_counts)):
        groups[(int(t), int(s))].append(i)
    out = []
    for idx in groups.values():
        out.extend(idx[j : j + batch_size] for j in range(0, len(idx), batch_size))
    return out


def metrics(y_true, y_pred, n_classes):
    """Same calls as evaluation.evaluation (accuracy, macro and weighted F1 over the
    labels present, kappa over all classes)."""
    import sklearn.metrics as skm

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "macro_f1": float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(skm.cohen_kappa_score(y_true, y_pred, labels=list(range(n_classes)))),
    }


def stratified_sample(labels, n_per_class_max, seed=42):
    """Seeded stratified sample of indices: up to n_per_class_max per class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    out = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k = min(len(idx), n_per_class_max)
        out.extend(rng.choice(idx, size=k, replace=False).tolist())
    return np.array(sorted(out))


def cloud_effect(clean: float, control_mean: float) -> float:
    """Effect of cloud-targeted removal over random removal of as many dates."""
    return clean - control_mean


def residual_share(effect: float, shifted_score: float, upper_bound: float) -> float:
    """Cloud effect as a share of the remaining gap to the target-trained bound."""
    gap = upper_bound - shifted_score
    return float("nan") if gap <= 0 else effect / gap


def block_flags_to_parcels(block_ids, block_frac, min_block_frac=0.5):
    """Parcel-date flags from block-level cloudy fractions. block_ids: (P,), block_frac:
    (n_blocks, T) indexed by block id; returns (P, T) bool."""
    f = np.asarray(block_frac, dtype=float)[np.asarray(block_ids, dtype=int)]
    return cloud_flags(f, min_block_frac)


def add_nodata(flags: np.ndarray, score: np.ndarray, nodata: np.ndarray):
    """Treat zero-filled parcel-dates (no S2 coverage; shipped as all-zero pixels) as removable too:
    flags OR nodata, with nodata dates ranked first when the minimum-keep floor binds."""
    return flags | nodata, np.where(nodata, 2.0, np.nan_to_num(np.asarray(score, dtype=float), nan=0.0))


def nodata_dates(parcel_arrays) -> np.ndarray:
    """(P, T) bool: the parcel's pixels are zero in every band on that date."""
    return np.stack([p.reshape(p.shape[0], -1).max(axis=1) == 0 for p in parcel_arrays])
