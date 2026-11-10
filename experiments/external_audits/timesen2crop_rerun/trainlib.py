"""Loading and scoring shared by the tuning, training and test-time scripts."""
from __future__ import annotations

import numpy as np
import sklearn.metrics as skm

from experiments.external_audits.common import paths

from . import tslib as tl

PAPER = "timesen2crop_rerun"


def comp(tile: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(paths.paper_root(PAPER) / "composites" / f"{tile}.npz")
    return z["C"], z["y"].astype(np.int64)


def load_train() -> tuple[np.ndarray, np.ndarray]:
    parts = [comp(t) for t in tl.TRAIN_TILES]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def pool_medians(C: np.ndarray, seed: int = 42, n: int = 200000) -> np.ndarray:
    """Per-month band medians of the training pool over present pixel-months, from a fixed subsample."""
    idx = np.random.default_rng(seed).choice(len(C), min(n, len(C)), replace=False)
    return np.nanmedian(C[idx], axis=0).astype(np.float32)


def score(y: np.ndarray, p: np.ndarray, prior: np.ndarray | None = None) -> dict:
    """OA, macro F1 over the 16 classes, and OA reweighted to a class prior (recall per class x prior)."""
    cm = skm.confusion_matrix(y, p, labels=np.arange(16))
    out = {"oa": float(np.trace(cm) / cm.sum()), "f1m": float(skm.f1_score(y, p, labels=np.arange(16), average="macro", zero_division=0))}
    if prior is not None:
        rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
        out["oa_reweighted"] = float((rec * prior).sum())
    return out
