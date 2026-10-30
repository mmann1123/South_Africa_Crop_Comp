"""Pure functions: valid-observation counts, grid interpolation share and gap lengths per pixel.

Inputs are boolean matrices V of shape (n_pixels, n_dates) with the matching day-of-year
vector `doy` (1-based, ascending). The paper resamples each pixel to a 6-day grid of 60 steps
and fills masked values by linear interpolation; a grid step is "observed" when a valid
acquisition lies within +-half_window days, otherwise it is interpolated (between two valid
dates) or extrapolated (before the first or after the last valid date).
"""
from __future__ import annotations

import numpy as np


def grid_days(step: int = 6, n: int = 60, first: int = 1) -> np.ndarray:
    return first + step * np.arange(n)


def valid_counts(v: np.ndarray) -> np.ndarray:
    return v.sum(axis=1).astype(int)


def monthly_counts(v: np.ndarray, doy: np.ndarray, year: int = 2021) -> np.ndarray:
    """(n_pixels, 12) valid acquisitions per calendar month."""
    month = (np.datetime64(f"{year}-01-01") + (doy.astype(int) - 1).astype("timedelta64[D]")).astype("datetime64[M]").astype(int) % 12
    return np.stack([v[:, month == m].sum(axis=1) for m in range(12)], axis=1).astype(int)


def observed_steps(v: np.ndarray, doy: np.ndarray, grid: np.ndarray, half_window: int = 3) -> np.ndarray:
    """(n_pixels, n_steps) True where a valid date lies within +-half_window days of the grid day."""
    near = np.abs(doy[None, :] - grid[:, None]) <= half_window  # (steps, dates)
    return (v.astype(int) @ near.T.astype(int)) > 0


def step_kinds(v: np.ndarray, doy: np.ndarray, grid: np.ndarray, half_window: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean (n_pixels, n_steps) arrays: observed, interpolated (valid dates on both sides), extrapolated."""
    obs = observed_steps(v, doy, grid, half_window)
    n = v.shape[0]
    first = np.where(v.any(axis=1), np.argmax(v, axis=1), len(doy))
    last = np.where(v.any(axis=1), v.shape[1] - 1 - np.argmax(v[:, ::-1], axis=1), -1)
    first_doy = np.where(first < len(doy), doy[np.minimum(first, len(doy) - 1)], 10**6)
    last_doy = np.where(last >= 0, doy[np.maximum(last, 0)], -(10**6))
    outside = (grid[None, :] < first_doy[:, None]) | (grid[None, :] > last_doy[:, None])
    extra = ~obs & outside
    interp = ~obs & ~outside
    assert obs.shape == (n, len(grid))
    return obs, interp, extra


def longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of consecutive True along axis 1, per row."""
    out = np.zeros(mask.shape[0], dtype=int)
    cur = np.zeros(mask.shape[0], dtype=int)
    for j in range(mask.shape[1]):
        cur = np.where(mask[:, j], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def longest_gap_days(v: np.ndarray, doy: np.ndarray, start: int = 1, end: int = 365) -> np.ndarray:
    """Longest interval in days between consecutive valid dates per pixel; the window edges count as
    boundaries (as in common.tables.longest_gap_days). A pixel with no valid date gets end - start."""
    out = np.empty(v.shape[0], dtype=int)
    for i in range(v.shape[0]):
        pts = np.concatenate([[start], doy[v[i]], [end]])
        out[i] = int(np.diff(pts).max())
    return out


def pixel_metrics(v: np.ndarray, doy: np.ndarray, grid: np.ndarray, half_window: int = 3) -> dict[str, np.ndarray]:
    obs, interp, extra = step_kinds(v, doy, grid, half_window)
    n_steps = len(grid)
    return {
        "n_valid": valid_counts(v),
        "obs_steps": obs.sum(axis=1),
        "interp_steps": interp.sum(axis=1),
        "extrap_steps": extra.sum(axis=1),
        "share_not_observed": 1.0 - obs.sum(axis=1) / n_steps,
        "longest_unobserved_steps": longest_run(~obs),
        "longest_gap_days": longest_gap_days(v, doy),
    }


def summarise(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [10, 25, 50, 75, 90])
    return {"mean": float(values.mean()), "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}


def drops(transfer: dict[str, list[float]], within: dict[str, list[float]]) -> list[float]:
    """Within-site baseline OA minus transfer OA, flattened over algorithm and feature set."""
    return [round(w - t, 1) for a in transfer for t, w in zip(transfer[a], within[a])]
