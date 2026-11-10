"""Inputs for the Sen4AgriNet PAD re-run: a numpy re-implementation of the
authors' monthly median and gap fill (export_medians_multi.get_medians, S4A-Models
ae5a376), with hooks to delete acquisitions or mask pixels before the median.

Authors' rule, per band and pixel: acquisitions are binned into calendar months
with right-closed intervals (Jan 1, Feb 1], ..., labelled by the month start;
each bin takes the median over time; a month with no acquisition is NaN and is
then filled by xarray interpolate_na(method='linear', fill_value='extrapolate')
over the 12 month-start dates, which interpolates linearly in time between the
nearest valid months and extrapolates linearly from the two nearest valid months
at the ends of the year. Empty months stay in place (checked against xarray
0.19.0, the authors' pinned version, and the current release), so the
April to September window is always bins 3 to 8. Coarser bands are repeated to
10 m after the median. `check_against_authors` compares this implementation with
the authors' function on a real patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

MODEL_BANDS = ["B02", "B03", "B04", "B08"]  # sorted, as PADDataset sorts them
HOLLSTEIN_BANDS = ["B01", "B02", "B03", "B05", "B06", "B07", "B09", "B10", "B11", "B8A"]
RES = {"B02": 10, "B03": 10, "B04": 10, "B08": 10, "B05": 20, "B07": 20, "B06": 20, "B8A": 20, "B11": 20,
       "B12": 20, "B01": 60, "B09": 60, "B10": 60}
WINDOW = slice(3, 9)  # April to September
SUB = 61
NORM = 10000.0


@dataclass
class Patch:
    stem: str
    year: int
    times: pd.DatetimeIndex
    bands: dict  # band -> (T, h, w) float32 at native resolution
    labels: np.ndarray  # (366, 366) raw crop ids


def read_patch(path: Path, bands: list[str]) -> Patch:
    nc = netCDF4.Dataset(path, "r")
    try:
        year = int(nc.patch_year)
        tv = nc["B02"]["time"]
        times = pd.to_datetime(netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
        out = {}
        for b in bands:
            g = nc[b]
            t = g["time"]
            if not np.array_equal(t[:], tv[:]) or t.units != tv.units:
                raise ValueError(f"{path.name}: {b} time axis differs from B02")
            out[b] = np.asarray(g[b][:], dtype=np.float32)
        labels = np.asarray(nc["labels"]["labels"][:])
    finally:
        nc.close()
    return Patch(path.stem, year, pd.DatetimeIndex(times), out, labels)


def month_of(times: pd.DatetimeIndex, year: int) -> np.ndarray:
    """0-based month bin of each acquisition under right-closed bins; -1 outside the year."""
    edges = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="1MS")
    idx = np.searchsorted(edges.values, times.values, side="left") - 1  # t in (edges[k], edges[k+1]] -> k
    idx[(times.values <= edges.values[0]) | (times.values > edges.values[-1])] = -1
    return idx.astype(int)


def month_starts_days(year: int) -> np.ndarray:
    d = pd.date_range(f"{year}-01-01", periods=12, freq="1MS")
    return (d.values.astype("datetime64[ns]").astype(np.int64) / 86400e9).astype(np.float64)


def fill_weights(valid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """12 x 12 matrix W with filled = W @ medians(NaN->0) for one valid-month
    pattern, reproducing scipy interp1d(kind='linear', fill_value='extrapolate')
    on the valid months. One valid month gives a constant (interp1d would raise;
    the authors' code would fail there too, so such stacks are flagged)."""
    v = np.flatnonzero(valid)
    w = np.zeros((12, 12))
    if len(v) == 0:
        w[:] = np.nan
        return w
    if len(v) == 1:
        w[:, v[0]] = 1.0
        return w
    for m in range(12):
        if valid[m]:
            w[m, m] = 1.0
            continue
        j = np.searchsorted(x[v], x[m])
        lo, hi = (v[0], v[1]) if j == 0 else (v[-2], v[-1]) if j >= len(v) else (v[j - 1], v[j])
        t = (x[m] - x[lo]) / (x[hi] - x[lo])
        w[m, lo] += 1 - t
        w[m, hi] += t
    return w


def monthly_filled(data: np.ndarray, months: np.ndarray, year: int, keep: np.ndarray | None = None,
                   pixel_ok: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """data (T, h, w). keep (T,) drops whole acquisitions; pixel_ok (T, h, w)
    drops single pixel-acquisitions. Returns (12, h, w) filled medians and the
    (12, h, w) boolean of months that had data before the fill."""
    T, h, w = data.shape
    sel = months >= 0 if keep is None else (months >= 0) & keep
    med = np.full((12, h, w), np.nan, np.float32)
    for m in range(12):
        idx = np.flatnonzero(sel & (months == m))
        if len(idx) == 0:
            continue
        block = data[idx]
        if pixel_ok is not None:
            block = np.where(pixel_ok[idx], block, np.nan)
            with np.errstate(all="ignore"):
                med[m] = np.nanmedian(block, axis=0)
        else:
            med[m] = np.median(block, axis=0)
    has = ~np.isnan(med)
    x = month_starts_days(year)
    flat = med.reshape(12, -1)
    hflat = has.reshape(12, -1)
    out = np.empty_like(flat)
    codes = np.packbits(hflat, axis=0, bitorder="little")
    codes = codes[0].astype(np.int32) | (codes[1].astype(np.int32) << 8)
    for c in np.unique(codes):
        cols = np.flatnonzero(codes == c)
        valid = hflat[:, cols[0]]
        W = fill_weights(valid, x)
        out[:, cols] = (W @ np.nan_to_num(flat[:, cols])).astype(np.float32)
    return out.reshape(12, h, w), has


def upsample(a: np.ndarray, band: str) -> np.ndarray:
    r = RES[band] // 10
    return a if r == 1 else np.repeat(np.repeat(a, r, axis=-2), r, axis=-1)


def encode_labels(labels: np.ndarray, linear_encoder: dict) -> tuple[np.ndarray, np.ndarray]:
    """As PADDataset.__getitem__: parcels = raw label != 0; labels mapped by the
    linear encoder, everything else 0."""
    parcels = labels != 0
    enc = np.zeros(labels.shape, np.int64)
    for crop_id, lin in linear_encoder.items():
        enc[labels == crop_id] = lin
    enc[~np.isin(enc, list(linear_encoder.values()))] = 0
    return enc, parcels


def to_subpatches(a: np.ndarray) -> np.ndarray:
    """(..., 366, 366) -> (36, ..., 61, 61), sub index i*6+j as in the export."""
    lead = a.shape[:-2]
    n = a.shape[-1] // SUB
    b = a.reshape(*lead, n, SUB, n, SUB)
    nd = len(lead)
    perm = (nd, nd + 2) + tuple(range(nd)) + (nd + 1, nd + 3)
    return b.transpose(perm).reshape(n * n, *lead, SUB, SUB)


def export_slots() -> list[list[tuple[int, int]]]:
    """(band index, month index) held by each of the 6 x 4 model input slots
    when medians come from the authors' export_medians_multi.py.

    get_medians returns medians.transpose(1, 0, 2, 3), a non-contiguous view of
    a (band, month, H, W) buffer, and the gist sliding_window_view computes its
    strides as if the array were C-contiguous (month, band, H, W), so slot
    (t, b) reads flat index 4t + b of the (band, month) buffer: band
    (4t + b) // 12, month (4t + b) % 12. The fixed window keeps t = 3..8, which
    is B03 for months 1 to 12 followed by B04 for months 1 to 12 (verified
    against the authors' functions on a real patch)."""
    return [[((4 * t + b) // 12, (4 * t + b) % 12) for b in range(4)] for t in range(3, 9)]


def model_inputs(p: Patch, linear_encoder: dict, keep: np.ndarray | None = None,
                 pixel_ok: np.ndarray | None = None, layout: str = "export") -> dict:
    """Model inputs (36, 6, 4, 61, 61) normalised, encoded labels and parcels
    (36, 61, 61), and the per-month data-presence share of the patch before the
    fill (12,), for one patch under one intervention.

    layout 'export' reproduces what the released checkpoints were trained and
    tested on (the authors' saved medians, see export_slots); layout 'intended'
    is the April to September window of all four bands that the paper and
    README describe."""
    months = month_of(p.times, p.year)
    full, has_all = [], None
    for b in MODEL_BANDS:
        f, has = monthly_filled(p.bands[b], months, p.year, keep=keep, pixel_ok=pixel_ok)
        full.append(f)
        has_all = has if has_all is None else has_all & has
    full = np.stack(full, axis=0)  # (4 bands, 12 months, H, W)
    if layout == "export":
        x = np.stack([np.stack([full[bi, mi] for bi, mi in row]) for row in export_slots()]) / NORM
    elif layout == "intended":
        x = full[:, WINDOW].transpose(1, 0, 2, 3) / NORM  # (6, 4, 366, 366)
    else:
        raise ValueError(layout)
    enc, parcels = encode_labels(p.labels, linear_encoder)
    return {
        "medians": to_subpatches(x.astype(np.float32)),
        "labels": to_subpatches(enc),
        "parcels": to_subpatches(parcels),
        "present_share": has_all.reshape(12, -1).mean(axis=1),
    }


def hollstein_ok(p: Patch, clouds: bool = True, cirrus: bool = False, shadows: bool = False) -> np.ndarray:
    """(T, 366, 366) True where the authors' own hollstein_mask (utils/tools.py)
    does not flag the pixel-acquisition, applied per acquisition (the authors'
    loader only computes it on medians and never uses it)."""
    from experiments.external_audits.sen4agrinet_rerun import evalcore  # noqa: PLC0415

    with evalcore._in_repo():
        from utils.tools import hollstein_mask  # noqa: PLC0415
    bands = {b: upsample(p.bands[b], b) / NORM for b in HOLLSTEIN_BANDS}
    with np.errstate(all="ignore"):
        m = hollstein_mask(bands, clouds=clouds, cirrus=cirrus, shadows=shadows, snow=False, requires_norm=True)
    return ~np.asarray(m, dtype=bool)


def check_against_authors(path: Path) -> float:
    """Max absolute difference between this module's window medians and the
    authors' export_medians_multi.get_medians on one patch (raw reflectance)."""
    from experiments.external_audits.sen4agrinet_rerun import evalcore  # noqa: PLC0415

    with evalcore._in_repo():
        import export_medians_multi as emm  # noqa: PLC0415
    nc = netCDF4.Dataset(path, "r")
    try:
        ref = emm.get_medians(nc, 0, 12, "1MS", MODEL_BANDS, 366, 366, (SUB, SUB), 0, 0, 0, 0, np.float32)
    finally:
        nc.close()
    p = read_patch(path, MODEL_BANDS)
    months = month_of(p.times, p.year)
    mine = np.stack([monthly_filled(p.bands[b], months, p.year)[0] for b in MODEL_BANDS], axis=1)
    return float(np.nanmax(np.abs(mine - ref)))
