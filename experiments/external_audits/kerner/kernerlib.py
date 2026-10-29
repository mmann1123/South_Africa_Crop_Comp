"""Pure functions for the Kerner et al. 2020 audit: pixel geometry, stratified
sampling, per-pixel HLS state vectors, window counts and bootstrap intervals.

State codes per acquisition and pixel: 0 = no data (pixel not observed that
day), 1 = clear (Fmask bits 0-3 all zero), 2 = flagged (any of bits 0-3 set).
"""
from __future__ import annotations

import re

import numpy as np

TILE = "16TBL"
TILE_CRS = "EPSG:32616"
TILE_ULX, TILE_ULY, TILE_RES = 199980.0, 4600020.0, 30.0  # HLS 16TBL grid (crs_transform read from Earth Engine)
FMASK_BITS = 0b1111  # bits 0 (cirrus), 1 (cloud), 2 (adjacent to cloud/shadow), 3 (cloud shadow)
HALF_WINDOW = 8
CLASS_NAMES = {0: "corn", 1: "soy", 2: "other"}
FIXED_DOYS = {2017: (153, 210, 274), 2018: (149, 209, 271), 2019: (152, 211, 274)}
FIXED_STAGES = ("early", "mid", "late")
GROWTH_STAGES = ("greenup", "peak", "senescence")
CLEAR, FLAGGED, NODATA = 1, 2, 0

_NAME_RE = re.compile(r"^(?:[SL]_)?T(\d\d[A-Z]{3})_(\d{4})(\d\d)(\d\d)T")


def pixel_to_xy(row, col):
    """UTM 16N coordinates of pixel centres on the HLS 16TBL grid."""
    row = np.asarray(row, dtype=float)
    col = np.asarray(col, dtype=float)
    return TILE_ULX + (col + 0.5) * TILE_RES, TILE_ULY - (row + 0.5) * TILE_RES


def parse_acquisition(name: str) -> tuple[str, str, int, int]:
    """'S_T16TBL_20180720T163839' -> ('S', '16TBL', year, calendar DOY)."""
    import datetime as dt

    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"unrecognised HLS image name: {name}")
    sensor = name[0] if name[1] == "_" else "?"
    tile, y, mo, d = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return sensor, tile, y, dt.date(y, mo, d).timetuple().tm_yday


def stratified_sample(labels: np.ndarray, valid: np.ndarray, n_per_class: int, seed: int = 42, classes=(0, 1, 2)):
    """Row/col indices of n_per_class random valid pixels per class (without replacement).

    Returns (rows, cols, cls) sorted by class then drawn order.
    """
    rng = np.random.default_rng(seed)
    rows, cols, cls = [], [], []
    for c in classes:
        r, k = np.nonzero((labels == c) & valid)
        n = min(n_per_class, len(r))
        pick = rng.choice(len(r), size=n, replace=False)
        rows.append(r[pick])
        cols.append(k[pick])
        cls.append(np.full(n, c))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(cls)


def window_counts(states: np.ndarray, acq_doy: np.ndarray, target_doy, half: int = HALF_WINDOW):
    """Clear counts and exact-day state for each pixel.

    states: (N, A) int codes; acq_doy: (A,) calendar DOY of each acquisition;
    target_doy: scalar or (N,) calendar DOY of the model input. Returns
    (n_clear, n_flagged, exact) with exact in {CLEAR, FLAGGED, NODATA}: the best
    state among acquisitions on exactly target_doy (clear beats flagged beats
    no data); NODATA when no acquisition falls on that day.
    """
    states = np.asarray(states)
    acq = np.asarray(acq_doy)[None, :]
    t = np.asarray(target_doy, dtype=float)
    t = t[:, None] if t.ndim else t
    inwin = np.abs(acq - t) <= half
    n_clear = ((states == CLEAR) & inwin).sum(axis=1)
    n_flag = ((states == FLAGGED) & inwin).sum(axis=1)
    onday = acq == t
    exact = np.zeros(states.shape[0], dtype=int)
    exact[((states == FLAGGED) & onday).any(axis=1)] = FLAGGED
    exact[((states == CLEAR) & onday).any(axis=1)] = CLEAR
    return n_clear, n_flag, exact


def best_offset(stage_doys: np.ndarray, acq_doy_set, offsets=(0, 1)) -> tuple[int, dict[int, float]]:
    """Which index-to-calendar offset makes the phenology DOYs land on HLS acquisition days.

    Returns the offset with the highest exact-match share and all shares.
    """
    s = set(int(a) for a in acq_doy_set)
    d = np.asarray(stage_doys).ravel()
    share = {o: float(np.mean([(int(v) + o) in s for v in d])) for o in offsets}
    return max(share, key=share.get), share


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Mean of values (n_pixels, k) with a percentile CI from resampling pixels (rows)."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    n = len(v)
    rng = np.random.default_rng(seed)
    per_pix = v.mean(axis=1)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = per_pix[idx].mean(axis=1)
    return float(per_pix.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
