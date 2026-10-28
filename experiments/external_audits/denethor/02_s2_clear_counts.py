#!/usr/bin/env python
"""Per-field clear Sentinel-2 acquisition counts for the DENETHOR tile-years
from the shipped CLP, SCL and IS_DATA layers (no BANDS.npy needed)."""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, zonal  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import longest_gap_days, write_csv  # noqa: E402
from experiments.external_audits.denethor import catalogue, s2clear  # noqa: E402

PAPER = "denethor"
T_CHUNK = 8
RES = paths.results_dir(PAPER)


def process_tile_year(tile_year: str, log: RunLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = paths.paper_root(PAPER) / tile_year
    tile, year, zone = catalogue.tile_of(tile_year), catalogue.year_of(tile_year), catalogue.utm_zone_of(tile_year)
    minx, miny, maxx, maxy, crs = s2clear.load_bbox(root / "bbox.pkl", tile, zone)
    with open(root / "timestamp.pkl", "rb") as fh:
        ts = pd.to_datetime(pd.Series(pickle.load(fh)), utc=True).dt.tz_localize(None)  # naive UTC
    clp = np.load(root / "CLP.npy", mmap_mode="r")
    scl = np.load(root / "SCL.npy", mmap_mode="r")
    isd = np.load(root / "IS_DATA.npy", mmap_mode="r")
    T, H, W = clp.shape[:3]
    scale = s2clear.detect_clp_scale(np.asarray(clp[::max(1, T // 6), ::50, ::50]))
    labels = gpd.read_file(paths.paper_root(PAPER) / "labels" / f"{tile}.geojson")
    transform = zonal.grid_from_bbox(minx, miny, maxx, maxy, size=H)
    idx, fields = zonal.field_index_map(labels, transform, (H, W), crs)
    n = len(fields)
    clp_mean = np.empty((T, n))
    scl_clear = np.empty((T, n))
    isd_frac = np.empty((T, n))
    for t0 in range(0, T, T_CHUNK):
        t1 = min(T, t0 + T_CHUNK)
        c = np.asarray(clp[t0:t1, :, :, 0]).astype(np.float32)
        s = np.asarray(scl[t0:t1, :, :, 0])
        d = np.asarray(isd[t0:t1, :, :, 0]).astype(bool)
        clp_mean[t0:t1] = zonal.field_means(c, idx, n)
        scl_clear[t0:t1] = zonal.field_fraction(np.isin(s, s2clear.SCL_CLEAR), idx, n)
        isd_frac[t0:t1] = zonal.field_fraction(d, idx, n)
    rules = s2clear.clear_rules(clp_mean, scl_clear, isd_frac, scale)
    crop_col = next((c for c in ("crop_name", "crop", "label", "crop_id") if c in labels.columns), None)
    crops = labels.loc[fields["field_id"], crop_col].values if crop_col else [None] * n
    dates_long = pd.DataFrame(
        {
            "tile_year": tile_year,
            "tile": tile,
            "year": year,
            "field_id": np.repeat(fields["field_id"].values, T),
            "date": np.tile(ts.dt.strftime("%Y-%m-%d").values, n),
            "clp_mean": clp_mean.T.ravel(),
            "scl_clear_frac": scl_clear.T.ravel(),
            "isdata_frac": isd_frac.T.ravel(),
            "clear_literal40": rules["literal40"].T.ravel(),
            "clear_s2cp40": rules["s2cp40"].T.ravel(),
            "clear_scl": rules["scl"].T.ravel(),
        }
    )
    season = (ts >= f"{year}-04-01") & (ts <= f"{year}-10-31")
    months = ts.dt.month.values
    rows = []
    for j in range(n):
        c40 = rules["s2cp40"][:, j]
        by_month = [int(c40[months == m].sum()) for m in range(1, 13)]
        rows.append(
            {
                "tile_year": tile_year,
                "tile": tile,
                "year": year,
                "labels_from_year": catalogue.LABEL_YEAR[tile],
                "field_id": fields["field_id"].iloc[j],
                "crop": crops[j],
                "n_pixels": int(fields["n_pixels"].iloc[j]),
                "n_clear_literal40": int(rules["literal40"][:, j].sum()),
                "n_clear_s2cp40": int(c40.sum()),
                "n_clear_scl": int(rules["scl"][:, j].sum()),
                "n_clear_s2cp40_aprOct": int(c40[season.values].sum()),
                "longest_gap_days_s2cp40_aprOct": longest_gap_days(ts, pd.Series(c40), f"{year}-04-01", f"{year}-10-31"),
                "clear_by_month_s2cp40": ";".join(map(str, by_month)),
            }
        )
    log.add(
        tile_year,
        {
            "T": int(T),
            "grid": [H, W],
            "bbox": [minx, miny, maxx, maxy],
            "crs": crs,
            "clp_scale": scale,
            "n_label_polygons": int(len(labels)),
            "n_fields_rasterised": int(n),
            "n_fields_dropped_no_pixels": int(fields.attrs["n_dropped"]),
            "crop_column": crop_col,
            "first_date": str(ts.iloc[0].date()),
            "last_date": str(ts.iloc[-1].date()),
            "n_dates_aprOct": int(season.sum()),
        },
    )
    return dates_long, pd.DataFrame(rows)


def main(tile_years: list[str] | None = None, force: bool = False) -> None:
    tile_years = tile_years or catalogue.S2_TILE_YEARS
    out_summary = RES / "s2_field_summary.csv"
    out_dates = RES / "s2_field_dates.csv"
    if out_summary.exists() and not force:
        print(f"{out_summary} exists; use --force")
        return
    with RunLog("02_s2_clear_counts", {"tile_years": tile_years, "force": force}, paper=PAPER) as log:
        longs, sums = [], []
        for ty in tile_years:
            d, s = process_tile_year(ty, log)
            longs.append(d)
            sums.append(s)
            print(f"{ty}: {len(s)} fields, median clear (s2cp40) {s['n_clear_s2cp40'].median()}", flush=True)
        summary = pd.concat(sums, ignore_index=True)
        dates = pd.concat(longs, ignore_index=True)
        write_csv(summary, out_summary, "02_s2_clear_counts", [str(paths.manifest_path(PAPER))])
        dates_path = out_dates
        dates.to_csv(dates_path, index=False)
        size_mb = dates_path.stat().st_size / 1e6
        if size_mb > 20:
            ext_path = paths.paper_root(PAPER) / "s2_field_dates.csv"
            shutil.move(str(dates_path), str(ext_path))  # results dir and external root are different filesystems
            dates.to_csv(RES / "s2_field_dates.csv.gz", index=False, compression="gzip")
            log.add("s2_field_dates", {"size_mb_uncompressed": round(size_mb, 1), "committed_as": "s2_field_dates.csv.gz", "uncompressed_copy": str(ext_path)})
        else:
            log.add("s2_field_dates", {"size_mb": round(size_mb, 1), "committed_as": "s2_field_dates.csv"})
        log.count("fields_total", int(len(summary)))
        log.count("field_dates_rows", int(len(dates)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-years", nargs="*", default=None, dest="tile_years")
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
