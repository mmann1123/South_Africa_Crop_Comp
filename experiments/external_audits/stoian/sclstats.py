"""Pure functions for per-pixel clear-observation statistics from SCL stacks."""
from __future__ import annotations

import numpy as np

from experiments.external_audits.stoian.catalogue import SCL_CLEAR

WATER = 6


def clear_mask(scl: np.ndarray) -> np.ndarray:
    """Clear = Sen2Cor SCL in {2, 4, 5, 6, 7, 11} (snow counts as valid)."""
    return np.isin(scl, SCL_CLEAR)


def merge_same_date(granules: list[np.ndarray]) -> np.ndarray:
    """Merge granules of one date into one SCL layer.

    A pixel takes the first clear value among the granules that cover it; if none is
    clear, the first non-zero value (cloud, shadow, ...); 0 only if no granule covers it.
    """
    out = np.zeros_like(granules[0])
    for g in reversed(granules):  # earlier granules win
        out = np.where(g != 0, g, out)
    for g in reversed(granules):
        out = np.where(clear_mask(g), g, out)
    return out


def land_mask(scl_stack: np.ndarray) -> np.ndarray:
    """Land pixels: at least one clear observation and water (SCL 6) in under half of the clear ones."""
    clear = clear_mask(scl_stack)
    n_clear = clear.sum(0)
    n_water = (scl_stack == WATER).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n_clear > 0) & (n_water / np.maximum(n_clear, 1) < 0.5)


def window_days(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    d = np.asarray(dates, dtype="datetime64[D]")
    return (d >= np.datetime64(start)) & (d <= np.datetime64(end))


def per_pixel_longest_gap(clear: np.ndarray, dates: np.ndarray, start: str, end: str) -> np.ndarray:
    """Per-pixel longest run of days without a clear date in [start, end], edges as boundaries.

    Same definition as common.tables.longest_gap_days, vectorised over pixels. `clear`
    is (D, H, W) boolean with dates sorted ascending.
    """
    d = np.asarray(dates, dtype="datetime64[D]")
    s, e = np.datetime64(start), np.datetime64(end)
    last = np.full(clear.shape[1:], 0, dtype=np.int32)  # days since start of the last clear date
    best = np.zeros(clear.shape[1:], dtype=np.int32)
    for i in np.flatnonzero((d >= s) & (d <= e)):
        t = int((d[i] - s) / np.timedelta64(1, "D"))
        hit = clear[i]
        best = np.where(hit, np.maximum(best, t - last), best)
        last = np.where(hit, t, last)
    total = int((e - s) / np.timedelta64(1, "D"))
    return np.maximum(best, total - last)


def tile_stats(scl_stack: np.ndarray, dates: np.ndarray, season: tuple[str, str], region: np.ndarray | None = None) -> dict:
    """Per-tile summary over land pixels (optionally restricted to `region`, a boolean (H, W))."""
    clear = clear_mask(scl_stack)
    land = land_mask(scl_stack)
    if region is not None:
        land = land & region
    n_year = clear.sum(0)
    w = window_days(dates, *season)
    n_season = clear[w].sum(0)
    gap = per_pixel_longest_gap(clear, dates, *season)
    sel = lambda a: a[land].astype(float)  # noqa: E731
    return {
        "n_land_pixels": int(land.sum()),
        "n_dates": int(len(dates)),
        "median_clear_year": float(np.median(sel(n_year))),
        "mean_clear_year": float(sel(n_year).mean()),
        "p10_clear_year": float(np.percentile(sel(n_year), 10)),
        "median_clear_marOct": float(np.median(sel(n_season))),
        "p10_clear_marOct": float(np.percentile(sel(n_season), 10)),
        "median_longest_gap_marOct_days": float(np.median(sel(gap))),
    }
