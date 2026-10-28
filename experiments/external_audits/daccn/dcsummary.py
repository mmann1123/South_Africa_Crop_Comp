"""Summary arithmetic for the DACCN zero-fill counters."""
from __future__ import annotations

import numpy as np
import pandas as pd

ZERO_COLS = [f"zero_{k:02d}" for k in range(22)]


def region_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, year, variant), g in df.groupby(["region", "year", "variant"], sort=True):
        z = g[ZERO_COLS].mean().values
        cell_shares = g.groupby("cell")["n_zero"].apply(lambda x: (x > 0).mean()) if "cell" in g.columns else pd.Series(dtype=float)
        rows.append(
            {
                "region": region,
                "country": g["country"].iloc[0],
                "year": int(year),
                "variant": variant,
                "frame": g["frame"].iloc[0],
                "n_points": int(len(g)),
                "median_n_valid": float(g["n_valid"].median()),
                "mean_n_valid": float(g["n_valid"].mean()),
                "share_any_zero": float((g["n_zero"] > 0).mean()),
                "mean_n_zero": float(g["n_zero"].mean()),
                "share_zero_step01": float(z[0]),
                "share_zero_step22": float(z[21]),
                "share_zero_apr": float(z[0:3].mean()),
                "share_zero_oct": float(z[19:22].mean()),
                "share_zero_by_step": ";".join(f"{v:.4f}" for v in z),
                "n_cells": int(len(cell_shares)),
                "cell_sd_share_any_zero": float(cell_shares.std(ddof=1)) if len(cell_shares) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pairs_table(df: pd.DataFrame, spatial_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    s = region_year_summary(df).set_index(["region", "year", "variant"])
    rows = []
    for variant in sorted(df["variant"].unique()):
        for src, tgt in spatial_pairs:
            for year in sorted(df["year"].unique()):
                if (src, year, variant) in s.index and (tgt, year, variant) in s.index:
                    a, b = s.loc[(src, year, variant)], s.loc[(tgt, year, variant)]
                    rows.append(_row("spatial", src, tgt, year, variant, a, b))
        years = sorted(df["year"].unique())
        if len(years) >= 2:
            y0, y1 = years[0], years[1]
            for region in sorted(df["region"].unique()):
                if (region, y0, variant) in s.index and (region, y1, variant) in s.index:
                    a, b = s.loc[(region, y0, variant)], s.loc[(region, y1, variant)]
                    rows.append(_row("temporal", f"{region} {y0}", f"{region} {y1}", y1, variant, a, b))
    return pd.DataFrame(rows)


def _row(kind, src, tgt, year, variant, a, b) -> dict:
    return {
        "kind": kind,
        "source": src,
        "target": tgt,
        "year": int(year),
        "variant": variant,
        "source_share_any_zero": float(a["share_any_zero"]),
        "target_share_any_zero": float(b["share_any_zero"]),
        "difference": float(b["share_any_zero"] - a["share_any_zero"]),
        "source_mean_n_zero": float(a["mean_n_zero"]),
        "target_mean_n_zero": float(b["mean_n_zero"]),
        "source_mean_n_valid": float(a["mean_n_valid"]),
        "target_mean_n_valid": float(b["mean_n_valid"]),
    }
