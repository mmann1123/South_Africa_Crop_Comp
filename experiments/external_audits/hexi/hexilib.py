"""Pure functions for the Hexi audit: weekly bins, QA60 bit tests, gap structure of weekly composites."""
from __future__ import annotations

import datetime as dt

import numpy as np

from experiments.external_audits.hexi.catalogue import N_WEEKS, QA60_CLOUD_BITS, SEASON_WEEKS


def week_bounds(year: int, k: int) -> tuple[dt.date, dt.date]:
    """[start, end) of seven-day bin k (0..51) counted from 1 January."""
    start = dt.date(year, 1, 1) + dt.timedelta(days=7 * k)
    return start, start + dt.timedelta(days=7)


def week_of(d: dt.date) -> int | None:
    """Bin index of a date, None for 31 December (day 365/366 of the year, outside the 52 bins)."""
    k = (d - dt.date(d.year, 1, 1)).days // 7
    return k if k < N_WEEKS else None


def qa60_cloud(qa60: np.ndarray) -> np.ndarray:
    """True where QA60 flags opaque cloud (bit 10) or cirrus (bit 11)."""
    mask = sum(1 << b for b in QA60_CLOUD_BITS)
    return (np.asarray(qa60).astype(np.int64) & mask) != 0


def longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in np.asarray(flags, bool):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def gap_metrics(valid: np.ndarray, season: tuple[int, int] = SEASON_WEEKS) -> dict[str, np.ndarray]:
    """Per-point gap structure of a (n_points, 52) boolean matrix of weeks with a usable composite.

    n_valid: weeks with an observation. n_interp: empty weeks between the first and last valid one
    (linear interpolation). n_edge: empty weeks before the first or after the last valid one (no
    interpolation possible; extrapolated or left empty). n_no_data: points without any valid week.
    longest_gap: longest run of empty weeks in the year (edge runs count); longest_interp: longest
    interior run; season_*: the same counts inside the inclusive week range `season`.
    """
    v = np.asarray(valid, bool)
    n, nw = v.shape
    any_ = v.any(1)
    first = np.where(any_, v.argmax(1), nw)
    last = np.where(any_, nw - 1 - v[:, ::-1].argmax(1), -1)
    empty = ~v
    n_edge = np.where(any_, first + (nw - 1 - last), nw)
    n_interp = np.where(any_, empty.sum(1) - n_edge, 0)
    a, b = season
    s_empty = empty[:, a : b + 1]
    idx = np.arange(nw)[None, :]
    interior = (idx > first[:, None]) & (idx < last[:, None]) & empty
    return {
        "n_valid": v.sum(1),
        "n_interp": n_interp,
        "n_edge": n_edge,
        "n_no_data": (~any_).astype(int),
        "longest_gap": np.array([longest_run(empty[i]) for i in range(n)]),
        "longest_interp": np.array([longest_run(interior[i, :]) for i in range(n)]),
        "season_valid": v[:, a : b + 1].sum(1),
        "season_interp": (interior[:, a : b + 1]).sum(1),
        "season_longest_gap": np.array([longest_run(s_empty[i]) for i in range(n)]),
    }


def summary(valid: np.ndarray, season: tuple[int, int] = SEASON_WEEKS) -> dict[str, float]:
    """Point-level means for one domain, scenario and year (one row of the summary table)."""
    m = gap_metrics(valid, season)
    nw = valid.shape[1]
    ns = season[1] - season[0] + 1
    ok = m["n_no_data"] == 0
    return {
        "n_points": int(valid.shape[0]),
        "valid_weeks_mean": float(m["n_valid"].mean()),
        "valid_weeks_median": float(np.median(m["n_valid"])),
        "empty_share": float(1 - m["n_valid"].mean() / nw),
        "interp_share": float(m["n_interp"].mean() / nw),
        "edge_share": float(m["n_edge"].mean() / nw),
        "longest_gap_mean": float(m["longest_gap"].mean()),
        "longest_interp_mean": float(m["longest_interp"][ok].mean()) if ok.any() else float("nan"),
        "season_valid_weeks_mean": float(m["season_valid"].mean()),
        "season_empty_share": float(1 - m["season_valid"].mean() / ns),
        "season_interp_share": float(m["season_interp"].mean() / ns),
        "season_longest_gap_mean": float(m["season_longest_gap"].mean()),
        "share_points_season_gap_ge_4": float((m["season_longest_gap"] >= 4).mean()),
        "share_points_no_valid_week": float(1 - ok.mean()),
    }


def residual_cloud(n_qa: np.ndarray, n_qa_cs: np.ndarray) -> dict[str, float]:
    """Cloud that QA60 lets through, from per-point-week counts of QA60-clear observations (n_qa)
    and QA60-clear observations that Cloud Score+ also calls clear (n_qa_cs).

    obs_share: share of QA60-clear observations that Cloud Score+ flags. week_any: share of
    QA60-valid weeks with at least one such observation (a maximum composite can pick it up).
    week_all: share of QA60-valid weeks in which every QA60-clear observation is flagged.
    """
    q = np.asarray(n_qa, float)
    c = np.asarray(n_qa_cs, float)
    wk = q > 0
    return {
        "obs_share": float(1 - c.sum() / q.sum()) if q.sum() > 0 else float("nan"),
        "week_any": float((c[wk] < q[wk]).mean()) if wk.any() else float("nan"),
        "week_all": float((c[wk] == 0).mean()) if wk.any() else float("nan"),
    }
