"""Pure re-implementation of which calendar months the S4A-Models loader fills.

The loader (utils/PAD_dataset.py lines 293 to 307 at ae5a376) bins a patch-year's
acquisitions into calendar months (intervals (start, end], right-closed), takes
the median per month, and calls interpolate_na(method='linear',
fill_value='extrapolate') over the 12 monthly bins. A month with no acquisition
is therefore filled; if it lies between two months that have data it is
interpolated, and if it lies before the first or after the last month with data
it is extrapolated. Stored patch stacks have no NaN pixels, so the fill is
patch-wide. This module never imports or runs the authors' code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAD_MONTHS = tuple(range(4, 10))  # Apr to Sep, --fixed_window bins 3 to 8
OAD_MONTHS = tuple(range(5, 12))  # month indices 4 to 10 of 0..11 = May to Nov


def month_bins(times: pd.DatetimeIndex, year: int) -> np.ndarray:
    """Month number (1..12) for each acquisition under pd.cut-style bins
    (start, end] with date_range(year-01-01, year+1-01-01, freq='1MS'); -1 for
    acquisitions outside the year. An acquisition at exactly 00:00 on the 1st
    belongs to the previous month, which this reproduces by shifting 1 ns back."""
    t = pd.DatetimeIndex(times) - pd.Timedelta(1, "ns")
    return np.where(t.year == year, t.month, -1)


def present_months(times, year: int) -> np.ndarray:
    """Boolean array of 12: month has at least one acquisition."""
    m = month_bins(pd.DatetimeIndex(times), year)
    out = np.zeros(12, dtype=bool)
    for k in m[m > 0]:
        out[k - 1] = True
    return out


def fill_masks(present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(interpolated, extrapolated) boolean arrays of 12 for a presence mask.
    Months before the first or after the last present month are extrapolated;
    absent months between present ones are interpolated. With no present month
    nothing can be filled (the loader would return NaN); both masks are all
    False and callers should drop such stacks."""
    present = np.asarray(present, dtype=bool)
    interp = np.zeros(12, dtype=bool)
    extrap = np.zeros(12, dtype=bool)
    if not present.any():
        return interp, extrap
    idx = np.flatnonzero(present)
    first, last = idx[0], idx[-1]
    for k in range(12):
        if present[k]:
            continue
        if k < first or k > last:
            extrap[k] = True
        else:
            interp[k] = True
    return interp, extrap


def window_stats(present: np.ndarray, months: tuple[int, ...]) -> dict:
    """Fill counts inside a window of 1-based months for one patch-year."""
    interp, extrap = fill_masks(present)
    sel = np.array([m - 1 for m in months])
    return {
        "months": len(months),
        "interpolated": int(interp[sel].sum()),
        "extrapolated": int(extrap[sel].sum()),
        "filled": int((interp | extrap)[sel].sum()),
    }


def linear_fill(values: np.ndarray, present: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference for what interpolate_na(linear, extrapolate) does to one pixel
    series over x (days): np.interp inside the span, straight-line extension from
    the two nearest valid points outside it. Used to test the filled mask."""
    v = np.asarray(values, float).copy()
    p = np.asarray(present, bool)
    xs, vs = np.asarray(x, float)[p], v[p]
    for k in np.flatnonzero(~p):
        if x[k] < xs[0] or x[k] > xs[-1]:
            a, b = (0, 1) if x[k] < xs[0] else (len(xs) - 2, len(xs) - 1)
            if len(xs) == 1:
                v[k] = vs[0]
            else:
                v[k] = vs[a] + (vs[b] - vs[a]) * (x[k] - xs[a]) / (xs[b] - xs[a])
        else:
            v[k] = np.interp(x[k], xs, vs)
    return v


def month_starts(year: int) -> np.ndarray:
    """Month-start timestamps in days since the year start (the xarray time_bins
    coordinate that the interpolation is linear in)."""
    d = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    return ((d - d[0]) / pd.Timedelta(days=1)).to_numpy()


def fill_weights(present: np.ndarray, x: np.ndarray) -> np.ndarray:
    """12 x 12 matrix W with filled_series = W @ series, the linear operator that
    interpolate_na(linear, extrapolate) applies for a presence mask (identity rows
    for present months)."""
    present = np.asarray(present, bool)
    W = np.zeros((12, 12))
    idx = np.flatnonzero(present)
    for k in range(12):
        if present[k]:
            W[k, k] = 1.0
        elif len(idx) == 1:
            W[k, idx[0]] = 1.0
        elif x[k] < x[idx[0]]:
            a, b = idx[0], idx[1]
            w = (x[k] - x[a]) / (x[b] - x[a])
            W[k, a], W[k, b] = 1 - w, w
        elif x[k] > x[idx[-1]]:
            a, b = idx[-2], idx[-1]
            w = (x[k] - x[a]) / (x[b] - x[a])
            W[k, a], W[k, b] = 1 - w, w
        else:
            j = np.searchsorted(idx, k)
            a, b = idx[j - 1], idx[j]
            w = (x[k] - x[a]) / (x[b] - x[a])
            W[k, a], W[k, b] = 1 - w, w
    return W


def raw_collinear(means: np.ndarray, year: int, rel_tol: float = 1e-6) -> np.ndarray:
    """Vectorised first pass over N parcels (means shape N x 12 x B): flag months
    whose B band means lie on the time-weighted line through both neighbours, the
    two previous months or the two following months. All-zero or non-finite
    months are never flagged. This over-flags the valid month next to an edge run;
    infer_filled repairs that."""
    x = month_starts(year)
    m = np.asarray(means, float)
    N = m.shape[0]
    ok = np.isfinite(m).all(axis=2) & (m != 0).any(axis=2)
    out = np.zeros((N, 12), bool)

    def on_line(k, a, b):
        pred = m[:, a] + (m[:, b] - m[:, a]) * (x[k] - x[a]) / (x[b] - x[a])
        scale = np.maximum(np.abs(pred), 1e-12)
        return (np.abs(m[:, k] - pred) / scale < rel_tol).all(axis=1) & ok[:, a] & ok[:, b]

    for k in range(12):
        c = np.zeros(N, bool)
        if 0 < k < 11:
            c |= on_line(k, k - 1, k + 1)
        if k >= 2:
            c |= on_line(k, k - 2, k - 1)
        if k <= 9:
            c |= on_line(k, k + 1, k + 2)
        out[:, k] = c & ok[:, k]
    return out


def _candidate_sets(flag: np.ndarray):
    """All subsets of the raw-flagged months that leave at least one valid month,
    largest first, ties broken towards more extrapolated months. A truly filled
    month is always raw-flagged, so the filled set is one of these. Some patterns
    are genuinely ambiguous (a valid last month lying on the line through the two
    before it is indistinguishable from an extrapolated one); the tie-break picks
    the extrapolated reading, which is also the likelier one for trailing months."""
    idx = np.flatnonzero(flag)
    out = []
    for bits in range(1 << len(idx)):
        f = np.zeros(12, bool)
        for j, k in enumerate(idx):
            if bits >> j & 1:
                f[k] = True
        if not f.all():
            out.append(f)
    out.sort(key=lambda f: (-int(f.sum()), -int(fill_masks(~f)[1].sum())))
    return out


def infer_filled(means: np.ndarray, year: int, rel_tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Infer, for N parcels (means N x 12 x B), which months were filled by the
    loader. A filled set is accepted when rebuilding those months from the
    remaining ones with the linear fill operator reproduces the data (all bands,
    relative error < rel_tol); among accepted sets the largest wins, so a month
    that merely happens to lie on a line is counted as filled only when the whole
    pattern is consistent. Returns (filled, extrapolated) boolean N x 12."""
    x = month_starts(year)
    m = np.asarray(means, float)
    raw = raw_collinear(m, year, rel_tol)
    filled = np.zeros_like(raw)
    wcache: dict[bytes, np.ndarray] = {}
    for pat in {tuple(r) for r in raw}:
        rows = np.flatnonzero((raw == np.array(pat)).all(axis=1))
        todo = np.ones(len(rows), bool)
        for f in _candidate_sets(np.array(pat)):
            if not todo.any():
                break
            W = wcache.get(f.tobytes())
            if W is None:
                W = wcache[f.tobytes()] = fill_weights(~f, x)
            sub = m[rows[todo]]
            rebuilt = np.einsum("kj,njb->nkb", W, sub)
            err = np.abs(rebuilt - sub) / np.maximum(np.abs(sub), 1e-12)
            good = (err < rel_tol).all(axis=(1, 2))
            idx = rows[todo][good]
            filled[idx] = f
            t = np.flatnonzero(todo)
            todo[t[good]] = False
    extrap = np.zeros_like(filled)
    for i in np.unique(np.flatnonzero(filled.any(axis=1))):
        _, e = fill_masks(~filled[i])
        extrap[i] = e
    return filled, extrap
