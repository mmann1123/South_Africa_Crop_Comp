"""Pure functions for the Wijesingha audit: bimonth windows, dropna flags, summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_ORDER = ("L5", "L7", "L8")


def bimonth_window(year: int, start_month: int) -> tuple[str, str]:
    """[start, end) dates of the two-month window beginning in start_month (as ee.Date.advance(2, 'month'))."""
    end_year, end_month = year, start_month + 2
    if end_month > 12:
        end_year, end_month = year + 1, end_month - 12
    return f"{year}-{start_month:02d}-01", f"{end_year}-{end_month:02d}-01"


def count_columns(bimonths: list[int], sensors=SENSOR_ORDER) -> list[str]:
    """Column names n_<sensor>_m<mm> and n_all_m<mm> in a fixed order."""
    return [f"n_{s}_m{m:02d}" for m in bimonths for s in (*sensors, "all")]


def dropna_flags(df: pd.DataFrame, bimonths: list[int]) -> pd.DataFrame:
    """Add empty_m<mm> (median composite undefined: no valid observation) and dropped (any bimonth
    empty, i.e. the row is removed by df.dropna())."""
    out = df.copy()
    empties = []
    for m in bimonths:
        out[f"empty_m{m:02d}"] = out[f"n_all_m{m:02d}"] <= 0
        empties.append(f"empty_m{m:02d}")
    out["dropped"] = out[empties].any(axis=1)
    return out


def summarise_group(g: pd.DataFrame, bimonths: list[int]) -> dict:
    """One row per region-year: dropped share, per-bimonth empty share, mean and median valid obs, sensor mix."""
    row = {"n_points": int(len(g)), "dropped_share": float(g["dropped"].mean())}
    for m in bimonths:
        a = g[f"n_all_m{m:02d}"]
        row[f"empty_share_m{m:02d}"] = float((a <= 0).mean())
        row[f"mean_obs_m{m:02d}"] = float(a.mean())
        row[f"median_obs_m{m:02d}"] = float(a.median())
        for s in SENSOR_ORDER:
            row[f"mean_{s}_m{m:02d}"] = float(g[f"n_{s}_m{m:02d}"].mean())
    row["mean_obs_all_bimonths"] = float(np.mean([row[f"mean_obs_m{m:02d}"] for m in bimonths]))
    return row


def period_contrast(tab: pd.DataFrame, train_years: list[int], test_years: list[int]) -> pd.DataFrame:
    """Per region: mean dropped share over train years and over test years (year rows weighted equally)."""
    rows = []
    for region, g in tab.groupby("region"):
        tr = g[g["year"].isin(train_years)]
        te = g[g["year"].isin(test_years)]
        rows.append(
            {
                "region": region,
                "dropped_train_mean": float(tr["dropped_share"].mean()),
                "dropped_test_mean": float(te["dropped_share"].mean()),
                "obs_train_mean": float(tr["mean_obs_all_bimonths"].mean()),
                "obs_test_mean": float(te["mean_obs_all_bimonths"].mean()),
            }
        )
    return pd.DataFrame(rows)
