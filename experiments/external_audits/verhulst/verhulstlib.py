"""Pure functions for the Verhulst audit: dekad windows, gap structure of a dekadal series,
the Gaussian dilation kernel, and Figure 7 digitising helpers."""
from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

from experiments.external_audits.verhulst.catalogue import JAN_JUN, N_DEKADS


def dekad_bounds(year: int, k: int) -> tuple[dt.date, dt.date]:
    """[start, end) of dekad k (0..35): days 1-10, 11-20, and the rest of the month."""
    month, part = divmod(k, 3)
    start = dt.date(year, month + 1, (1, 11, 21)[part])
    if part < 2:
        end = dt.date(year, month + 1, (11, 21)[part])
    else:
        end = dt.date(year + (month == 11), (month + 1) % 12 + 1, 1)
    return start, end


def dekad_of(d: dt.date) -> int:
    return (d.month - 1) * 3 + (0 if d.day <= 10 else 1 if d.day <= 20 else 2)


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Normalised size x size Gaussian weights (the paper's dilation kernel)."""
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return k / k.sum()


def dilate_unusable(usable: np.ndarray, threshold: float | None, size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Usable mask after dilating the unusable pixels with the Gaussian kernel.

    A pixel stays usable if it is usable itself and the kernel-weighted share of unusable
    pixels around it is below `threshold`. `threshold=None` returns the mask unchanged.
    Pixels outside the array count as usable (no information), as do scene footprints.
    """
    usable = np.asarray(usable, bool)
    if threshold is None:
        return usable
    share = ndi.convolve((~usable).astype(float), gaussian_kernel(size, sigma), mode="constant", cval=0.0)
    return usable & (share < threshold)


def longest_run(flags: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = cur = 0
    for f in np.asarray(flags, bool):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def gap_metrics(valid: np.ndarray, window: int = JAN_JUN) -> dict[str, np.ndarray]:
    """Per-pixel gap structure of a (n_pixels, 36) boolean matrix of dekads with a valid median.

    n_valid: dekads with an observation. n_interp: empty dekads between the first and last
    valid one (linear interpolation). n_edge: empty dekads before the first or after the last
    valid one (forward/backward fill). n_no_data: pixels without any valid dekad (no series).
    longest_gap and longest_gap_jj: longest run of empty dekads in the year and in the first
    `window` dekads (a run that starts at the edge counts), longest_interp: longest interior run.
    """
    v = np.asarray(valid, bool)
    n = v.shape[0]
    n_valid = v.sum(1)
    any_ = n_valid > 0
    first = np.where(any_, v.argmax(1), v.shape[1])
    last = np.where(any_, v.shape[1] - 1 - v[:, ::-1].argmax(1), -1)
    empty = ~v
    n_edge = np.where(any_, first + (v.shape[1] - 1 - last), v.shape[1])
    n_interp = np.where(any_, empty.sum(1) - n_edge, 0)
    lg = np.array([longest_run(empty[i]) for i in range(n)])
    lg_jj = np.array([longest_run(empty[i, :window]) for i in range(n)])
    lg_int = np.array([longest_run(empty[i, first[i] : last[i] + 1]) if any_[i] else 0 for i in range(n)])
    return {
        "n_valid": n_valid,
        "n_interp": n_interp,
        "n_edge": n_edge,
        "n_no_data": (~any_).astype(int),
        "longest_gap": lg,
        "longest_gap_jj": lg_jj,
        "longest_interp": lg_int,
        "n_valid_jj": v[:, :window].sum(1),
    }


def year_summary(valid: np.ndarray) -> dict[str, float]:
    """Pixel-level means and shares of the gap metrics for one year (one row of the summary table)."""
    m = gap_metrics(valid)
    nd = valid.shape[1]
    ok = m["n_no_data"] == 0
    return {
        "n_pixels": int(valid.shape[0]),
        "valid_dekads_mean": float(m["n_valid"].mean()),
        "valid_dekads_median": float(np.median(m["n_valid"])),
        "valid_dekads_jan_jun_mean": float(m["n_valid_jj"].mean()),
        "valid_share": float(m["n_valid"].mean() / nd),
        "interp_share": float(m["n_interp"].mean() / nd),
        "edge_share": float(m["n_edge"].mean() / nd),
        "filled_share": float(1 - m["n_valid"].mean() / nd),
        "longest_gap_mean": float(m["longest_gap"].mean()),
        "longest_gap_jan_jun_mean": float(m["longest_gap_jj"].mean()),
        "longest_interp_mean": float(m["longest_interp"][ok].mean()) if ok.any() else float("nan"),
        "share_pixels_gap_ge_6_jan_jun": float((m["longest_gap_jj"] >= 6).mean()),
        "share_pixels_gap_ge_9_year": float((m["longest_gap"] >= 9).mean()),
        "share_pixels_no_valid_dekad": float(1 - ok.mean()),
    }


def monthly_valid_share(valid: np.ndarray) -> np.ndarray:
    """Share of (pixel, dekad) cells with a valid observation per month, shape (12,)."""
    v = np.asarray(valid, float)
    return v.reshape(v.shape[0], 12, 3).mean(axis=(0, 2))


def spearman(x, y) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = stats.spearmanr(x, y)
    return {"n": int(len(x)), "spearman_rho": float(rho), "spearman_p": float(p)}


def scenarios(oa: dict[tuple[int, int], float], years: list[int]) -> list[dict]:
    """Cross-year scenarios (train, test) with OA loss = same-year OA of the test year minus cross OA."""
    out = []
    for te in years:
        for tr in years:
            if tr != te:
                out.append({"train_year": tr, "test_year": te, "oa_same_year": oa[(te, te)], "oa_cross": oa[(tr, te)], "oa_loss": oa[(te, te)] - oa[(tr, te)]})
    return out


# ---- Figure 7 digitising ------------------------------------------------------------

def calibrate_linear(px: list[float], values: list[float]) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.asarray(px, float), np.asarray(values, float), 1)
    return float(slope), float(intercept)


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of consecutive True runs in a 1-D boolean array."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cuts + 1]].tolist(), np.r_[idx[cuts], idx[-1]].tolist()))


def colour_mask(rgb: np.ndarray, colour: tuple[int, int, int], tol: int = 14) -> np.ndarray:
    return (np.abs(rgb.astype(int) - np.asarray(colour)) <= tol).all(-1)


def box_from_mask(mask: np.ndarray) -> tuple[float, float, float] | None:
    """(top_row, median_row, bottom_row) of a box drawn in a fill colour whose median line is
    darker, so the median shows as the row with the least fill. Uses the largest vertical run
    of rows holding fill; None if no fill is present."""
    rows = mask.sum(1)
    rr = runs(rows >= max(3, 0.4 * rows.max())) if rows.max() > 0 else []
    if not rr:
        return None
    merged = [rr[0]]
    for r in rr[1:]:  # bridge the median line (and anti-aliased edges) between fill rows
        if r[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    a, b = max(merged, key=lambda r: r[1] - r[0])
    if b - a < 4:
        return float(a), (a + b) / 2.0, float(b)
    inner = rows[a + 1 : b]
    median = float(a + 1 + np.argmin(inner)) if inner.min() < 0.6 * rows[a : b + 1].max() else (a + b) / 2.0
    return float(a), median, float(b)


def blob_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(row, col) centroid of the largest 8-connected component."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    cy, cx = ndi.center_of_mass(lab == 1 + int(np.argmax(sizes)))
    return float(cy), float(cx)
