"""Pure statistics for the Stoian audit: composition index and rank correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def composition_index(areas: pd.DataFrame, tiles: list[str], exclude: tuple[str, ...] = ("WAT",)) -> pd.Series:
    """Jensen-Shannon divergence (base 2, so in [0, 1]) between each tile's class-area
    distribution and the pooled distribution of the other tiles, water excluded.

    `areas` has one row per class (index or column 'class') and one column per tile.
    """
    a = areas.set_index("class") if "class" in areas.columns else areas
    a = a.drop(index=[c for c in exclude if c in a.index])[tiles].astype(float)
    out = {}
    for t in tiles:
        own = a[t].to_numpy()
        rest = a.drop(columns=t).sum(axis=1).to_numpy()
        out[t] = float(jensenshannon(own / own.sum(), rest / rest.sum(), base=2) ** 2)
    return pd.Series(out, name="composition_jsd")


def rank_and_linear(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p_rho = stats.spearmanr(x, y)
    r, p_r = stats.pearsonr(x, y)
    return {"n": int(len(x)), "spearman_rho": float(rho), "spearman_p": float(p_rho), "pearson_r": float(r), "pearson_p": float(p_r)}


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x and y given z: Pearson partial on the ranks."""
    rx, ry, rz = (stats.rankdata(np.asarray(v, float)) for v in (x, y, z))
    exy, eyz, exz = (np.corrcoef(a, b)[0, 1] for a, b in ((rx, ry), (ry, rz), (rx, rz)))
    return float((exy - exz * eyz) / np.sqrt((1 - exz**2) * (1 - eyz**2)))


def leave_one_out_range(x, y) -> tuple[float, float]:
    """Min and max Spearman rho over all leave-one-out subsets."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rhos = [stats.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
    return float(min(rhos)), float(max(rhos))


def ols_r2(y, *xs) -> float:
    """R squared of an OLS fit of y on the given regressors (with intercept)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(v, float) for v in xs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())))


def share_of_classes(areas: pd.DataFrame, tiles: list[str], classes: list[str], exclude: tuple[str, ...] = ("WAT",)) -> pd.Series:
    """Share of each tile's (water-excluded) area held by the given classes."""
    a = areas.set_index("class") if "class" in areas.columns else areas
    a = a.drop(index=[c for c in exclude if c in a.index])[tiles].astype(float)
    return a.loc[classes].sum() / a.sum()
