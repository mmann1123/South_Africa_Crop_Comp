"""Pure functions for the Masolele audit: confluence grid, half-year windows, empty-composite flags, summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_ORDER = ("L5", "L7")


def confluence_points(lon_range: tuple[int, int], lat_max: int) -> list[tuple[int, int]]:
    """Integer (lon, lat) confluence points with lon in lon_range (inclusive) and |lat| <= lat_max."""
    lo, hi = lon_range
    return [(lon, lat) for lon in range(lo, hi + 1) for lat in range(-lat_max, lat_max + 1)]


def unit_box(lon: float, lat: float, half: float) -> list[float]:
    """[west, south, east, north] of the sample unit centred on a confluence point."""
    return [lon - half, lat - half, lon + half, lat + half]


def half_window(year: int, half: int) -> tuple[str, str]:
    """[start, end) of the January-June (half 1) or July-December (half 2) composite window."""
    if half == 1:
        return f"{year}-01-01", f"{year}-07-01"
    if half == 2:
        return f"{year}-07-01", f"{year + 1}-01-01"
    raise ValueError(half)


def window_ids(years: list[int], halves=(1, 2)) -> list[str]:
    return [f"{y}h{h}" for y in years for h in halves]


def count_columns(windows: list[str], sensors=SENSOR_ORDER) -> list[str]:
    """n_<sensor>_<window> and n_all_<window> in a fixed order."""
    return [f"n_{s}_{w}" for w in windows for s in (*sensors, "all")]


def long_form(df: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    """One row per point and window with n_all, n_L5, n_L7 and empty (no valid observation, so the median is masked)."""
    parts = []
    for w in windows:
        p = df[["continent", "point_id"]].copy()
        p["window"] = w
        p["year"] = int(w[:4])
        p["half"] = int(w[-1])
        for s in (*SENSOR_ORDER, "all"):
            p[f"n_{s}"] = df[f"n_{s}_{w}"].to_numpy()
        p["empty"] = p["n_all"] <= 0
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def continent_summary(lf: pd.DataFrame) -> pd.DataFrame:
    """Per continent: points, empty share of point-windows, share of points with any empty window,
    mean empty windows per point (of 12), mean and median valid observations per window."""
    rows = []
    for c, g in lf.groupby("continent"):
        per_point = g.groupby("point_id")["empty"].sum()
        rows.append(
            {
                "continent": c,
                "n_points": int(per_point.size),
                "empty_share": float(g["empty"].mean()),
                "points_any_empty": float((per_point > 0).mean()),
                "mean_empty_windows": float(per_point.mean()),
                "mean_obs_per_window": float(g["n_all"].mean()),
                "median_obs_per_window": float(g["n_all"].median()),
                "share_windows_le1_obs": float((g["n_all"] <= 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def window_summary(lf: pd.DataFrame) -> pd.DataFrame:
    """Per continent and window: empty share and mean valid observations."""
    return (
        lf.groupby(["continent", "window", "year", "half"])
        .agg(n_points=("point_id", "size"), empty_share=("empty", "mean"), mean_obs=("n_all", "mean"),
             mean_L5=("n_L5", "mean"), mean_L7=("n_L7", "mean"))
        .reset_index()
    )


def max_pairwise_ratio(values: dict[str, float]) -> float:
    """Largest ratio between any two positive values (inf if one is zero and another positive)."""
    v = np.array(list(values.values()), dtype=float)
    if (v <= 0).any() and (v > 0).any():
        return float("inf")
    return float(v.max() / v.min()) if v.min() > 0 else float("nan")
