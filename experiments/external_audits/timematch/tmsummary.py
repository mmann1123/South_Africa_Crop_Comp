"""Summary arithmetic for the TimeMatch per-date cloud fractions."""
from __future__ import annotations

import pandas as pd

RULES = ("csplus", "s2cp40", "s2cp16")
LABELS = {"33UVP": "AT1", "32VNH": "DK1", "30TXT": "FR1", "31TCJ": "FR2"}


def tile_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, tile), g in df.groupby(["country", "tile"], sort=True):
        with_img = g[g["n_images"] > 0]
        row = {
            "country": country,
            "tile": tile,
            "label": LABELS.get(tile, tile),
            "n_dates": int(len(g)),
            "n_dates_with_images": int(len(with_img)),
            "mean_tile_cloudy_pct": float(g["tile_cloudy_pct"].mean()),
            "median_tile_cloudy_pct": float(g["tile_cloudy_pct"].median()),
            "dates_tile_over50": int((g["tile_cloudy_pct"] > 50).sum()),
            "mean_frac_nodata": float(with_img["frac_nodata"].mean()),
        }
        for rule in RULES:
            f = with_img[f"frac_cloudy_{rule}"].dropna()
            row[f"mean_field_cloudy_{rule}"] = float(f.mean())
            row[f"median_field_cloudy_{rule}"] = float(f.median())
            row[f"dates_field_over50_{rule}"] = int((f > 0.5).sum())
            row[f"dates_field_over80_{rule}"] = int((f > 0.8).sum())
            row[f"expected_cloudy_obs_per_series_{rule}"] = float(f.sum())
            row[f"expected_clear_obs_per_series_{rule}"] = float((1 - f).sum())
            row[f"expected_cloudy_obs_{rule}"] = row[f"expected_cloudy_obs_per_series_{rule}"]
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.month
    rows = []
    for (country, tile, month), g in d.groupby(["country", "tile", "month"], sort=True):
        row = {"country": country, "tile": tile, "label": LABELS.get(tile, tile), "month": int(month), "n_dates": int(len(g))}
        for rule in RULES:
            f = g.loc[g["n_images"] > 0, f"frac_cloudy_{rule}"].dropna()
            row[f"expected_cloudy_obs_{rule}"] = float(f.sum())
            row[f"expected_clear_obs_{rule}"] = float((1 - f).sum())
        rows.append(row)
    return pd.DataFrame(rows)
