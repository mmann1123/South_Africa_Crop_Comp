"""Pure functions for the Gallo et al. 2023 audit: series length harmonisation, probe helpers, cloud flags."""
from __future__ import annotations

import datetime as dt

import numpy as np

PAPER = "gallo"
SEQ_LEN = 71  # sample_duration in the authors' training command and predict_daily_annual.py
BASE = "https://www.kaggle.com/api/v1/datasets/download/ignazio/sentinel2-crop-mapping"
AREAS = {"lombardia": "L1", "lombardia2": "L2", "lombardia3": "L3"}
YEARS = (2016, 2017, 2018, 2019)
# Reflectance x 10000 of the first band (blue) above which a pixel is called cloud or snow.
CLOUD_BLUE = 2000


def candidate_days(year: int) -> list[str]:
    """Every calendar day of the year as YYYYMMDD, for probing which dates a patch folder holds."""
    d = dt.date(year, 1, 1)
    out = []
    while d.year == year:
        out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


def file_url(area_dir: str, year: int, patch: int, name: str) -> str:
    return f"{BASE}?fileName={area_dir}/data{year}/{patch}/{name}.tif"


def harmonise_counts(n_dates: int, seq_len: int = SEQ_LEN) -> dict:
    """What the authors' length harmonisation does to a year with n_dates images.

    Fewer than seq_len: zero images are appended (pad). More: random subsample down to seq_len.
    """
    pad = max(0, seq_len - n_dates)
    kept = min(n_dates, seq_len)
    return {"n_dates": n_dates, "pad_images": pad, "pad_share": pad / seq_len, "subsample_keep": kept / n_dates if n_dates else float("nan")}


def image_state(arr: np.ndarray, cloud_blue: int = CLOUD_BLUE) -> dict:
    """Per-image fractions from a (bands, H, W) array: no-data (all bands zero), cloud (blue above threshold)."""
    nodata = (arr == 0).all(axis=0)
    cloud = (arr[0] > cloud_blue) & ~nodata
    n = nodata.size
    return {"nodata": float(nodata.sum() / n), "cloud": float(cloud.sum() / n), "clear": float((~nodata & ~cloud).sum() / n)}


def effective_clear(clear_frac: np.ndarray, thresh: float = 0.5) -> int:
    """Number of images in a series whose clear fraction is at least thresh."""
    return int((np.asarray(clear_frac) >= thresh).sum())


def zero_image_share(n_dates: int, n_zero_images: float, seq_len: int = SEQ_LEN) -> float:
    """Share of the seq_len slots that hold no usable signal: padding plus all-zero (no-data) real images."""
    pad = max(0, seq_len - n_dates)
    kept = min(n_dates, seq_len)
    # subsampling is uniform, so the expected no-data images kept scale with kept / n_dates
    z = n_zero_images * (kept / n_dates if n_dates else 0)
    return (pad + z) / seq_len


def parse_name(name: str) -> dict | None:
    """Split a Kaggle path like lombardia2/data2018/17/20180105.tif; None for non-patch files and MSAVI index rasters."""
    parts = name.split("/")
    if len(parts) != 4 or not parts[1].startswith("data") or not parts[3].endswith(".tif"):
        return None
    stem = parts[3][:-4]
    if parts[0] not in AREAS or not parts[2].isdigit():
        return None
    if stem == "y":
        return {"area": AREAS[parts[0]], "year": int(parts[1][4:]), "patch": int(parts[2]), "kind": "label", "date": None}
    if len(stem) == 8 and stem.isdigit():
        return {"area": AREAS[parts[0]], "year": int(parts[1][4:]), "patch": int(parts[2]), "kind": "image", "date": stem}
    return None


def date_stats(dates_per_patch: dict) -> dict:
    """Summary of images per patch for one area-year, and the pad/subsample rule applied to the median patch."""
    n = np.array([len(v) for v in dates_per_patch.values()])
    med = int(np.median(n))
    out = {"n_patches": int(len(n)), "dates_min": int(n.min()), "dates_median": med, "dates_max": int(n.max())}
    out.update({k: v for k, v in harmonise_counts(med).items() if k != "n_dates"})
    return out


def doy(date: str) -> int:
    return dt.datetime.strptime(date, "%Y%m%d").timetuple().tm_yday


def slot_shift(dates_a: list[str], dates_b: list[str], seq_len: int = SEQ_LEN) -> dict:
    """Calendar offset between the same slot index of two series (sorted dates, zero-padded at the end).

    The loader gives the network slot k = k-th image of the year, so a year with fewer or differently
    spaced images puts a different time of year in the same slot. Compared over slots both series fill.
    """
    a = sorted(dates_a)[:seq_len]
    b = sorted(dates_b)[:seq_len]
    k = min(len(a), len(b))
    diff = np.array([doy(a[i]) - doy(b[i]) for i in range(k)])
    return {"slots_compared": k, "median_abs_doy_shift": float(np.median(np.abs(diff))) if k else float("nan"), "max_abs_doy_shift": float(np.abs(diff).max()) if k else float("nan"), "slots_beyond_other": max(0, len(a) - len(b))}


def classify_dates(crop_cloudy: list, n_images: list, raster_nodata: list | None = None, cloudy_thresh: float = 0.5, nodata_thresh: float = 0.99) -> list:
    """Label each deposit date: 'nodata' (no Sentinel-2 granule found for it, or the deposited raster is
    all zero), 'cloudy' (cloudy share of cropland pixels at least cloudy_thresh) or 'clear'."""
    out = []
    rn = raster_nodata if raster_nodata is not None else [0.0] * len(n_images)
    for c, n, z in zip(crop_cloudy, n_images, rn):
        if n == 0 or z >= nodata_thresh or c is None or (isinstance(c, float) and np.isnan(c)):
            out.append("nodata")
        elif c >= cloudy_thresh:
            out.append("cloudy")
        else:
            out.append("clear")
    return out


def series_summary(labels: list, seq_len: int = SEQ_LEN) -> dict:
    """Composition of the 71 input slots for one area-year series given per-date labels.

    Slots hold the labelled images (or a random subsample of them when there are more than seq_len),
    followed by zero padding. clear_slots is the expected number of clear images among the slots.
    """
    n = len(labels)
    keep = min(1.0, seq_len / n) if n else 0.0
    counts = {k: sum(1 for x in labels if x == k) for k in ("clear", "cloudy", "nodata")}
    pad = max(0, seq_len - n)
    slots = {f"{k}_slots": v * keep for k, v in counts.items()}
    empty = pad + slots["nodata_slots"]
    return {"n_dates": n, "pad_slots": pad, **slots, "zero_slots": empty, "zero_share": empty / seq_len,
            "clear_share": slots["clear_slots"] / seq_len, "cloudy_share": slots["cloudy_slots"] / seq_len}


def pool_mean(df, cols: list, weights=None) -> dict:
    """Mean of cols over the rows of df (one row per area-year series), optionally weighted by patch count."""
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)
    return {c: float(np.average(df[c].to_numpy(dtype=float), weights=w)) for c in cols}
