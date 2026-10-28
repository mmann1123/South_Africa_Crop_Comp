"""Summary arithmetic for the DENETHOR per-field clear counts."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN = ("18E-242N", 2018)
TEST = ("17E-243N", 2019)


def tile_year_summary(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ty, g in s.groupby("tile_year", sort=True):
        c = g["n_clear_s2cp40"]
        months = np.array([list(map(int, x.split(";"))) for x in g["clear_by_month_s2cp40"]])
        rows.append(
            {
                "tile_year": ty,
                "tile": g["tile"].iloc[0],
                "year": int(g["year"].iloc[0]),
                "n_fields": int(len(g)),
                "median_n_clear_s2cp40": float(c.median()),
                "iqr_n_clear_s2cp40": float(c.quantile(0.75) - c.quantile(0.25)),
                "median_n_clear_literal40": float(g["n_clear_literal40"].median()),
                "median_n_clear_scl": float(g["n_clear_scl"].median()),
                "median_n_clear_s2cp40_aprOct": float(g["n_clear_s2cp40_aprOct"].median()),
                "share_fields_under_20_clear_aprOct_s2cp40": float((g["n_clear_s2cp40_aprOct"] < 20).mean()),
                "median_longest_gap_days": float(g["longest_gap_days_s2cp40_aprOct"].median()),
                "mean_clear_by_month": ";".join(f"{v:.1f}" for v in months.mean(axis=0)),
            }
        )
    return pd.DataFrame(rows)


def two_by_two(s: pd.DataFrame) -> pd.DataFrame:
    t = tile_year_summary(s)[["tile", "year", "median_n_clear_s2cp40", "median_n_clear_literal40", "median_n_clear_s2cp40_aprOct", "n_fields"]]
    return t.sort_values(["year", "tile"]).reset_index(drop=True)


def margins(t: pd.DataFrame, col: str = "median_n_clear_s2cp40") -> dict[str, float]:
    cell = t.set_index(["tile", "year"])[col]
    tiles = sorted({x[0] for x in cell.index})
    years = sorted({x[1] for x in cell.index})
    out = {}
    for tile in tiles:
        out[f"year_effect_{tile}"] = float(cell[(tile, years[1])] - cell[(tile, years[0])])
    for year in years:
        out[f"tile_effect_{year}"] = float(cell[(tiles[0], year)] - cell[(tiles[1], year)])
    out["benchmark_difference"] = float(cell[TEST] - cell[TRAIN])
    return out
