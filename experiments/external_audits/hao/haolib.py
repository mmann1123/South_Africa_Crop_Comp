"""Pure functions for the Hao et al. 2020 audit: composite windows, gap structure of an
18-window validity series, the 100 km grid cap on training samples, per-length summaries and
the bar-chart digitising helpers for Figs. 4, 6 and 8.

A validity matrix is (n_points, 18) boolean: True where the 15-day window holds at least one
observation that passes the quality rule (so the maximum-NDVI composite exists).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats

from experiments.external_audits.hao.catalogue import N_WINDOWS


def window_index(doy: int, start: int = 60, width: int = 15, n: int = N_WINDOWS) -> int | None:
    """Window 0..n-1 holding a calendar day of year, or None outside DOY [start, start + n*width)."""
    k = (doy - start) // width
    return int(k) if 0 <= k < n and doy >= start else None


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of consecutive True runs in a 1-D boolean array."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cuts + 1]].tolist(), np.r_[idx[cuts], idx[-1]].tolist()))


def longest_run(flags: np.ndarray) -> int:
    r = runs(np.asarray(flags, bool))
    return max((b - a + 1 for a, b in r), default=0)


def fill_classes(valid: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean (n, T) maps of how each empty window is treated by the paper's fill rule.

    interior: empty with a valid window somewhere before and after (the average of the two
    neighbouring high-quality values exists). edge: empty with no valid window on at least one
    side (no two-sided average; the series start or end is empty). strict: empty with both
    adjacent windows valid (the literal two-neighbour reading); interior but not strict windows
    sit in a run of two or more empty windows, where the average spans a longer gap.
    """
    v = np.asarray(valid, bool)
    before = np.maximum.accumulate(v, axis=1)
    after = np.maximum.accumulate(v[:, ::-1], axis=1)[:, ::-1]
    # "before" and "after" must exclude the window itself, but an empty window is not valid anyway
    empty = ~v
    interior = empty & before & after
    edge = empty & ~(before & after)
    left = np.zeros_like(v)
    right = np.zeros_like(v)
    left[:, 1:] = v[:, :-1]
    right[:, :-1] = v[:, 1:]
    strict = empty & left & right
    return {"empty": empty, "interior": interior, "edge": edge, "strict": strict}


def window_shares(valid: np.ndarray) -> pd.DataFrame:
    """Per window: share of points with no observation, filled by a two-sided average
    (interior), filled only one-sidedly or not at all (edge), and strictly neighbour-fillable."""
    c = fill_classes(valid)
    n = valid.shape[0]
    return pd.DataFrame(
        {
            "window": np.arange(1, valid.shape[1] + 1),
            "empty_share": c["empty"].mean(0),
            "interior_share": c["interior"].mean(0),
            "edge_share": c["edge"].mean(0),
            "strict_share": c["strict"].mean(0),
            "n_points": n,
        }
    )


def prefix_metrics(valid: np.ndarray, k: int) -> dict[str, float]:
    """Gap summary of the first k windows (a series of DOY 60 to 60 + 15 k), as in the in-season
    classifications. The series is cut before filling, so empty windows at its end count as edge."""
    v = np.asarray(valid, bool)[:, :k]
    c = fill_classes(v)
    n_empty = c["empty"].sum(1)
    lr = np.array([longest_run(~r) for r in v])
    return {
        "k": k,
        "doy_end": 60 + 15 * k,
        "empty_share": float(c["empty"].mean()),
        "edge_share": float(c["edge"].mean()),
        "interior_share": float(c["interior"].mean()),
        "share_points_any_empty": float((n_empty > 0).mean()),
        "share_points_run_ge2": float((lr >= 2).mean()),
        "share_points_no_obs": float((n_empty == k).mean()),
        "longest_run_mean": float(lr.mean()),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 500, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of a 1-D array with a percentile interval from resampling points."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


# ---- Training-sample design --------------------------------------------------------

def cap_per_cell(df: pd.DataFrame, cell_col: str, crop_col: str, cap: int, seed: int = 42) -> pd.DataFrame:
    """At most `cap` rows per (grid cell, crop), chosen at random with a fixed seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    shuffled = df.iloc[order]
    return shuffled.groupby([cell_col, crop_col], sort=False).head(cap).sort_index()


def subsample_to_counts(df: pd.DataFrame, crop_col: str, counts: dict[str, int], seed: int = 42) -> pd.DataFrame:
    """Random subset with at most counts[crop] rows per crop (all rows if fewer are available)."""
    parts = [g.sample(n=min(len(g), counts[c]), random_state=seed) for c, g in df.groupby(crop_col) if c in counts]
    return pd.concat(parts).sort_index()


def grid_cell(x: np.ndarray, y: np.ndarray, size: float = 100_000.0) -> np.ndarray:
    """Integer cell id of projected coordinates on a size x size grid."""
    ix = np.floor(np.asarray(x, float) / size).astype(np.int64)
    iy = np.floor(np.asarray(y, float) / size).astype(np.int64)
    return ix * 100_000 + iy


# ---- Correlation -----------------------------------------------------------------

def spearman(x, y) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = stats.spearmanr(x, y)
    return {"n": int(len(x)), "spearman_rho": float(rho), "spearman_p": float(p)}


# ---- Bar-chart digitising ---------------------------------------------------------

def orange_bars(rgb: np.ndarray, colour=(255, 108, 0), tol: int = 40, min_h: int = 40, min_w: int = 15) -> list[tuple[int, int, int, int]]:
    """(x0, x1, top, bottom) of each orange bar, left to right (legend squares are too short to pass min_h)."""
    mask = (np.abs(rgb.astype(int) - np.asarray(colour)) <= tol).all(-1)
    lab, _ = ndi.label(mask)
    out = [(s[1].start, s[1].stop, s[0].start, s[0].stop) for s in ndi.find_objects(lab) if s[0].stop - s[0].start > min_h and s[1].stop - s[1].start > min_w]
    return sorted(out)


def axis_label_rows(rgb: np.ndarray, base: int, x_lo: int, x_hi: int, height: int = 600, dark: int = 300, min_len: int = 15) -> np.ndarray:
    """Vertical centres of the six y-axis labels (100 % at the top, 0 % at the bottom) of the panel
    whose bars end at row `base`: dark text rows between columns x_lo and x_hi, the last six groups."""
    reg = rgb[base - height : base + 10, x_lo:x_hi].astype(int).sum(-1) < dark
    idx = np.flatnonzero(reg.sum(1) > 0)
    groups = [g for g in np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1) if len(g) >= min_len][-6:]
    if len(groups) != 6:
        raise RuntimeError(f"expected 6 axis labels, found {len(groups)}")
    return np.array([base - height + g.mean() for g in groups])


def bar_values(rows: np.ndarray, tops: list[int]) -> list[float]:
    """Percent value of each bar top from the label rows (100, 80, ..., 0)."""
    slope, icpt = np.polyfit(rows, [100, 80, 60, 40, 20, 0], 1)
    return [float(slope * t + icpt) for t in tops]


# ---- Sampling chunks -------------------------------------------------------------

def utm_zone(lon) -> np.ndarray:
    return (np.floor((np.asarray(lon, float) + 180.0) / 6.0).astype(int) % 60) + 1


def assign_chunks(lon, lat, max_n: int = 150, cell: float = 1.0) -> np.ndarray:
    """Chunk id per point: points sorted by UTM zone, then by 1-degree cell, then packed into
    chunks of at most max_n points that never mix zones and keep a cell together where possible."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    zone = utm_zone(lon)
    cx, cy = np.floor(lon / cell).astype(int), np.floor(lat / cell).astype(int)
    order = np.lexsort((cy, cx, zone))
    chunk = np.empty(len(lon), dtype=int)
    cur, n, prev = 0, 0, None
    for i in order:
        key = (zone[i], cx[i], cy[i])
        if n >= max_n or (prev is not None and zone[i] != prev[0]):
            cur, n = cur + 1, 0
        chunk[i] = cur
        n += 1
        prev = key
    return chunk


def combine_valid(df: pd.DataFrame, rule: str, n_windows: int = N_WINDOWS) -> np.ndarray:
    """(n, T) boolean validity matrix from the per-window count columns w<kk>_<rule>."""
    return np.column_stack([df[f"w{k:02d}_{rule}"].to_numpy() > 0 for k in range(n_windows)])
