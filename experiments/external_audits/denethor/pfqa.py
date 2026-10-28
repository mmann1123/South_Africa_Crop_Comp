"""Planet Fusion QA (PF-QA) readers and per-field synthetic-fraction statistics.

PF-QA layer 1 = percent synthetic (1 = observed, 100 = fully gap-filled),
layer 2 = days to nearest real observation (signed). Nodata is -1000.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from experiments.external_audits.common import zonal

NODATA = -1000
SYNTHETIC_THRESHOLD = 50


def read_day(url_or_path: str, window: Window) -> tuple[np.ndarray, np.ndarray] | None:
    """(percent_synthetic, days_to_real) as float32 with NaN at nodata, or None if unreadable."""
    src = url_or_path
    if src.startswith("http"):
        src = "/vsicurl/" + src
    elif not Path(src).exists():
        return None
    try:
        with rasterio.open(src) as ds:
            arr = ds.read([1, 2], window=window).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None
    arr[arr == NODATA] = np.nan
    return arr[0], arr[1]


def field_day_stats(
    percent_synthetic: np.ndarray,
    days_to_real: np.ndarray,
    index_map: np.ndarray,
    n_fields: int,
    synthetic_threshold: int = SYNTHETIC_THRESHOLD,
) -> pd.DataFrame:
    pct = np.asarray(percent_synthetic, dtype=np.float64)
    days = np.asarray(days_to_real, dtype=np.float64)
    valid = ~np.isnan(pct)
    idx = np.where(valid, index_map, -1)
    frac_syn = zonal.field_fraction((pct > synthetic_threshold)[None], idx, n_fields)[0]
    mean_pct = zonal.field_means(np.nan_to_num(pct)[None], idx, n_fields)[0]
    mean_days = zonal.field_means(np.nan_to_num(np.abs(days))[None], idx, n_fields)[0]
    n_valid = np.bincount(idx[idx >= 0], minlength=n_fields)
    out = pd.DataFrame(
        {
            "field_pos": np.arange(n_fields),
            "n_valid_pixels": n_valid,
            "frac_synthetic": frac_syn,
            "mean_percent_synthetic": mean_pct,
            "mean_days_to_real": mean_days,
        }
    )
    out.loc[out["n_valid_pixels"] == 0, ["frac_synthetic", "mean_percent_synthetic", "mean_days_to_real"]] = np.nan
    return out


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    """Per field and month: QA-day counts, share of mostly-synthetic days, means.
    Days with NaN frac_synthetic (no QA raster) are excluded from denominators."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.month
    d["has_qa"] = d["frac_synthetic"].notna()
    d["mostly_synthetic"] = d["frac_synthetic"] > 0.5
    g = d.groupby(["field_pos", "month"], sort=True)
    out = pd.DataFrame(
        {
            "n_days_with_qa": g["has_qa"].sum().astype(int),
            "n_days_missing": (g["has_qa"].count() - g["has_qa"].sum()).astype(int),
            "share_days_mostly_synthetic": d[d["has_qa"]].groupby(["field_pos", "month"], sort=True)["mostly_synthetic"].mean(),
            "mean_percent_synthetic": g["mean_percent_synthetic"].mean(),
            "mean_days_to_real": g["mean_days_to_real"].mean(),
        }
    ).reset_index()
    return out
