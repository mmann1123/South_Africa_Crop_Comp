#!/usr/bin/env python
"""Per-field fraction of Planet Fusion daily inputs that were synthesised
(PF-QA layer 1 > 50) and mean days to the nearest real observation, April to
October, for the two DENETHOR benchmark tile-years."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths, zonal  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.denethor import catalogue, pfqa  # noqa: E402

PAPER = "denethor"
RES = paths.results_dir(PAPER)
BENCHMARK = [("18E-242N", 2018), ("17E-243N", 2019)]
SEASON = ("04-01", "10-31")


def season_dates(year: int, smoke: bool) -> list[str]:
    d = pd.date_range(f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}", freq="D")
    d = d[:3] if smoke else d
    return [x.strftime("%Y-%m-%d") for x in d]


def process(tile: str, year: int, smoke: bool, log: RunLog) -> pd.DataFrame:
    root = paths.paper_root(PAPER)
    manifest = paths.manifest_path(PAPER)
    cache_dir = root / "pfqa_cache" / tile
    cache_dir.mkdir(parents=True, exist_ok=True)
    cog_dir = root / "pfqa_cogs" / tile
    labels = gpd.read_file(root / "labels" / f"{tile}.geojson")
    dates = season_dates(year, smoke)
    first_key = catalogue.pfqa_key(tile, dates[0])
    first = fetch.download(catalogue.url(first_key), cog_dir / f"{dates[0]}.tif", manifest, first_key, expected_bytes=catalogue.listing_size(first_key))
    with rasterio.open(first) as ds:
        crs, full_transform = ds.crs, ds.transform
        lab = labels.to_crs(crs)
        b = lab.total_bounds
        win = from_bounds(*b, transform=full_transform).round_offsets().round_lengths()
        win = Window(max(0, win.col_off), max(0, win.row_off), min(ds.width - max(0, win.col_off), win.width), min(ds.height - max(0, win.row_off), win.height))
        transform = ds.window_transform(win)
    idx, fields = zonal.field_index_map(lab, transform, (int(win.height), int(win.width)), crs)
    n = len(fields)
    log.add(f"{tile}_{year}", {"n_label_polygons": int(len(labels)), "n_fields_rasterised": n, "n_dropped": int(fields.attrs["n_dropped"]), "window": [int(win.col_off), int(win.row_off), int(win.width), int(win.height)], "n_days": len(dates)})
    frames, missing = [], []
    for date in dates:
        cache = cache_dir / f"{date}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            key = catalogue.pfqa_key(tile, date)
            try:
                path = fetch.download(catalogue.url(key), cog_dir / f"{date}.tif", manifest, key, expected_bytes=catalogue.listing_size(key))
            except Exception as e:  # noqa: BLE001
                path = None
                log.add(f"missing_{tile}_{date}", f"{type(e).__name__}: {e}")
            layers = pfqa.read_day(str(path), win) if path else None
            if layers is None:
                missing.append(date)
                df = pd.DataFrame({"field_pos": np.arange(n), "n_valid_pixels": 0, "frac_synthetic": np.nan, "mean_percent_synthetic": np.nan, "mean_days_to_real": np.nan})
            else:
                df = pfqa.field_day_stats(layers[0], layers[1], idx, n)
            df["date"] = date
            df.to_parquet(cache, index=False)
            if path and Path(path).exists():
                Path(path).unlink()  # keep disk low; the manifest records the fetch
        frames.append(df)
        if len(frames) % 20 == 0:
            print(f"{tile} {year}: {len(frames)}/{len(dates)} days", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    daily["tile"] = tile
    daily["year"] = year
    daily["field_id"] = fields["field_id"].values[daily["field_pos"].values]
    log.add(f"{tile}_{year}_missing_days", missing)
    return daily


def main(smoke: bool = False, force: bool = False) -> None:
    out_month = RES / "pfqa_field_month.csv"
    if out_month.exists() and not force and not smoke:
        print(f"{out_month} exists; use --force")
        return
    with RunLog("03_pf_qa_synthetic", {"smoke": smoke, "force": force}, paper=PAPER) as log:
        dailies = [process(tile, year, smoke, log) for tile, year in BENCHMARK]
        daily = pd.concat(dailies, ignore_index=True)
        monthly = pd.concat(
            [pfqa.summarise(d).assign(tile=d["tile"].iloc[0], year=d["year"].iloc[0]).merge(d[["field_pos", "field_id"]].drop_duplicates(), on="field_pos") for d in dailies],
            ignore_index=True,
        )
        if smoke:
            print(daily.head(), monthly.head(), sep="\n")
            return
        write_csv(monthly, out_month, "03_pf_qa_synthetic", [str(paths.manifest_path(PAPER))])
        daily_path = RES / "pfqa_field_day.csv.gz"
        daily.to_csv(daily_path, index=False, compression="gzip")
        log.count("field_days", int(len(daily)))
        log.count("field_months", int(len(monthly)))
        log.add("pfqa_field_day", str(daily_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
