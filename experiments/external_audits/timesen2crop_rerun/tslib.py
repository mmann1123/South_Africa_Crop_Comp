"""Monthly composites and gap fills for the TimeSen2Crop re-run.

Month index 0 = September ... 11 = August (the agronomic year). The paper's rule:
median of clear acquisitions per month, per band; a pixel-month with none is
all-zero. Composites are held with NaN for empty pixel-months and each fill turns
them into the model input."""
from __future__ import annotations

import numpy as np

TRAIN_TILES = ["32TNT", "32TPT", "32TQT", "33TUM", "33TUN", "33TVM", "33TVN", "33TWM", "33TXN", "33UUP", "33UWP", "33UWQ", "33UXP"]
VAL_TILE, TEST_TILE, YEAR_TILE = "33TWN", "33UVP", "2019_33UVP"
CLASSES = ["Legumes", "Grassland", "Maize", "Potato", "Sunflower", "Soy", "Winter Barley", "Winter Caraway", "Rye", "Rapeseed",
           "Beet", "Spring Cereals", "Winter Wheat", "Winter Triticale", "Permanent Plantation", "Other Crops"]
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]


def month_index(dates: np.ndarray) -> np.ndarray:
    """yyyymmdd -> 0 (September) .. 11 (August)."""
    return (((np.asarray(dates) // 100) % 100) - 9) % 12


def composite(X: np.ndarray, flag: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """(N, T, 9) int, (N, T) flags -> (N, 12, 9) float32 reflectance (x 1e-4), NaN where no clear acquisition."""
    m = month_index(dates)
    out = np.full((X.shape[0], 12, X.shape[2]), np.nan, np.float32)
    for k in range(12):
        idx = np.flatnonzero(m == k)
        if len(idx) == 0:
            continue
        x = X[:, idx].astype(np.float32)
        x[flag[:, idx] != 0] = np.nan
        for n0 in range(0, len(x), 20000):  # nanmedian is memory hungry
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[n0:n0 + 20000, k] = np.nanmedian(x[n0:n0 + 20000], axis=1)
    return out * 1e-4


def present_mask(C: np.ndarray) -> np.ndarray:
    """(N, 12, 9) -> (N, 12) bool: month holds a composite."""
    return ~np.isnan(C[..., 0])


def fill_zero(C: np.ndarray) -> np.ndarray:
    """The paper's rule."""
    return np.nan_to_num(C, nan=0.0)


def fill_median(C: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Empty pixel-months take a fixed (12, 9) table, the training tiles' per-month band medians."""
    return np.where(np.isnan(C), med[None], C)


def fill_linear(C: np.ndarray) -> np.ndarray:
    """Per pixel and band, linear in month between present months, nearest present month beyond the ends.
    A pixel with no present month at all stays zero."""
    N, M, B = C.shape
    p = present_mask(C)
    out = C.copy()
    idx = np.arange(M)
    # previous and next present month per (pixel, month)
    prev = np.where(p, idx, -1)
    prev = np.maximum.accumulate(prev, axis=1)
    nxt = np.where(p, idx, M)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    has_prev, has_next = prev >= 0, nxt < M
    r = np.arange(N)[:, None]
    lo = np.where(has_prev, prev, np.where(has_next, nxt, 0))
    hi = np.where(has_next, nxt, np.where(has_prev, prev, 0))
    a, b = C[r, lo], C[r, hi]
    den = (hi - lo).astype(np.float32)
    w = np.where(den > 0, (idx[None] - lo) / np.where(den > 0, den, 1), 0.0).astype(np.float32)[..., None]
    v = a + w * (b - a)
    out = np.where(np.isnan(C), v, C)
    return np.nan_to_num(out, nan=0.0)


def apply_pattern(C: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Blank pixel-months where keep (N, 12) is False."""
    out = C.copy()
    out[~keep] = np.nan
    return out
