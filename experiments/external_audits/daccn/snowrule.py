"""Pure snow rule and summary arithmetic for the DACCN snow check (no Earth Engine import)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Independent snow test applied to the 10-day median composite (TOA reflectance, 0-1).
# NDSI = (B3 - B11) / (B3 + B11) >= 0.40 and NIR > 0.11 is the MODIS MOD10 snow-mapping rule
# (Hall et al. 2002, IEEE TGRS 40:1002; Riggs, Hall & Salomonson 2006, MOD10 ATBD), whose
# Sentinel-2 adaptations (Sen2Cor L2A snow test; MAJA, Hagolle et al. 2010, RSE 114:1747)
# use the same NDSI and NIR conditions. A green > 0.10 floor (MOD10: band 4 > 0.10) removes dark
# targets with a spurious high NDSI. We use 0.42 as pre-specified.
NDSI_MIN = 0.42
NIR_MIN = 0.11
GREEN_MIN = 0.10
MODIS_MIN = 40  # MOD10A1 NDSI_Snow_Cover (0-100) at or above which a MODIS pixel-day counts as snow
N_STEPS = 22
VALID = [f"valid_{k:02d}" for k in range(N_STEPS)]
SNOW = [f"snow_{k:02d}" for k in range(N_STEPS)]
SCL = [f"scl_{k:02d}" for k in range(N_STEPS)]
MODIS = [f"modis_{k:02d}" for k in range(N_STEPS)]


def snow_flag(green, swir1, nir, ndsi_min: float = NDSI_MIN, nir_min: float = NIR_MIN, green_min: float = GREEN_MIN):
    """Boolean snow flag from composite reflectances (arrays or scalars)."""
    green, swir1, nir = (np.asarray(x, dtype=float) for x in (green, swir1, nir))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = (green - swir1) / (green + swir1)
    return (ndsi > ndsi_min) & (nir > nir_min) & (green > green_min)


def region_year_snow(df: pd.DataFrame) -> pd.DataFrame:
    """Per region-year-variant snow summary of valid composites."""
    rows = []
    for (region, year, variant), g in df.groupby(["region", "year", "variant"], sort=True):
        v = g[VALID].values
        s = g[SNOW].values * v  # snow only counts on a valid composite
        n_snow = s.sum(axis=1)
        n_valid = v.sum(axis=1)
        cell_any = pd.Series(n_snow > 0, index=g.index).groupby(g["cell"]).mean() if "cell" in g.columns else pd.Series(dtype=float)
        by_step = s.sum(axis=0) / np.maximum(v.sum(axis=0), 1)
        rows.append(
            {
                "region": region,
                "year": int(year),
                "variant": variant,
                "n_points": int(len(g)),
                "mean_n_valid": float(n_valid.mean()),
                "share_any_snow": float((n_snow > 0).mean()),
                "mean_n_snow": float(n_snow.mean()),
                "share_valid_steps_snow": float(s.sum() / max(v.sum(), 1)),
                "share_any_snow_apr": float((s[:, 0:3].sum(axis=1) > 0).mean()),
                "share_any_snow_oct": float((s[:, 19:22].sum(axis=1) > 0).mean()),
                "share_any_snow_midseason": float((s[:, 3:19].sum(axis=1) > 0).mean()),
                "share_snow_by_step": ";".join(f"{x:.4f}" for x in by_step),
                "n_cells": int(len(cell_any)),
                "cell_sd_share_any_snow": float(cell_any.std(ddof=1)) if len(cell_any) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def crosscheck(df: pd.DataFrame, group_cols=("region", "year")) -> pd.DataFrame:
    """Agreement of the composite snow flag with SCL == 11 and MODIS NDSI_Snow_Cover over the same window.

    Only valid composites are used. SCL is available where scl_k >= 0 (-1 = no Sentinel-2 L2A scene in the
    window); MODIS where modis_k >= 0 (-1 = no valid MOD10A1 observation). Reports P(other says snow | flag) and
    P(other says snow | no flag), the L2A/MODIS coverage of valid composites, and the flagged-composite count.
    """
    rows = []
    for key, g in df.groupby(list(group_cols), sort=True):
        v = g[VALID].values.astype(bool)
        flag = g[SNOW].values.astype(bool) & v
        scl = g[SCL].values
        mod = g[MODIS].values
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["n_valid_composites"] = int(v.sum())
        row["n_flagged"] = int(flag.sum())
        for name, arr, snow_arr in (("scl", scl, scl == 1), ("modis", mod, mod >= MODIS_MIN)):
            avail = (arr >= 0) & v
            row[f"{name}_coverage"] = float(avail.sum() / max(v.sum(), 1))
            fa = flag & avail
            na = (~flag) & avail
            row[f"{name}_n_flagged_with_data"] = int(fa.sum())
            row[f"{name}_snow_given_flag"] = float((snow_arr & fa).sum() / fa.sum()) if fa.sum() else np.nan
            row[f"{name}_snow_given_noflag"] = float((snow_arr & na).sum() / na.sum()) if na.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pairs_snow(summary: pd.DataFrame, spatial_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    s = summary.set_index(["region", "year", "variant"])
    years = sorted(summary["year"].unique())
    rows = []
    for variant in sorted(summary["variant"].unique()):
        for src, tgt in spatial_pairs:
            for year in years:
                if (src, year, variant) in s.index and (tgt, year, variant) in s.index:
                    rows.append(_row("spatial", src, tgt, year, variant, s.loc[(src, year, variant)], s.loc[(tgt, year, variant)]))
        if len(years) >= 2:
            for region in sorted(summary["region"].unique()):
                if (region, years[0], variant) in s.index and (region, years[1], variant) in s.index:
                    rows.append(_row("temporal", f"{region} {years[0]}", f"{region} {years[1]}", years[1], variant, s.loc[(region, years[0], variant)], s.loc[(region, years[1], variant)]))
    return pd.DataFrame(rows)


def _row(kind, src, tgt, year, variant, a, b) -> dict:
    return {
        "kind": kind,
        "source": src,
        "target": tgt,
        "year": int(year),
        "variant": variant,
        "source_share_any_snow": float(a["share_any_snow"]),
        "target_share_any_snow": float(b["share_any_snow"]),
        "difference": float(b["share_any_snow"] - a["share_any_snow"]),
        "source_mean_n_snow": float(a["mean_n_snow"]),
        "target_mean_n_snow": float(b["mean_n_snow"]),
    }
