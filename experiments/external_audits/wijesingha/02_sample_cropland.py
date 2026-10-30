#!/usr/bin/env python
"""Draw N random cropland pixels (ESA WorldCover class 40) inside North Hesse (geoBoundaries
'Kassel') and Weser-Ems, seed 42, on a 30 m grid; saved so every year uses the same points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.wijesingha import catalogue  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
OVERSAMPLE = 6  # cropland is roughly a quarter to a third of each region
SIMPLIFY_DEG = 0.002  # about 150 m; keeps the EE request small


def region_geometries() -> dict:
    p = fetch.download(catalogue.GEOB_URL, ROOT / "boundaries" / "geoBoundaries-DEU-ADM2.geojson", paths.manifest_path(PAPER), "geoBoundaries-DEU-ADM2.geojson")
    g = gpd.read_file(p).to_crs(4326)
    out = {}
    for code, name in catalogue.REGION_NAME.items():
        geom = g.loc[g["shapeName"] == name, "geometry"].union_all().simplify(SIMPLIFY_DEG)
        out[code] = geom
    return out


def main(force: bool = False) -> None:
    out = RES / "cropland_points.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    import ee

    ee_init.init()
    with RunLog("02_sample_cropland", {"n_points": catalogue.N_POINTS, "force": force}, paper=PAPER) as log:
        geoms = region_geometries()
        wc = ee.ImageCollection(catalogue.WORLDCOVER).first().select("Map")
        # 30 m grid: fraction of 10 m cropland cells in each 30 m cell, keep cells that are mostly cropland.
        crop = wc.eq(catalogue.CROPLAND_CLASS).reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject("EPSG:32632", None, 30)
        rows = []
        for code, geom in geoms.items():
            region = ee.Geometry(json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"])
            area_km2 = region.area(100).getInfo() / 1e6
            frac = crop.gte(0.5).selfMask()
            pts = frac.sample(region=region, scale=30, numPixels=catalogue.N_POINTS * OVERSAMPLE, seed=catalogue.SEED, geometries=True, tileScale=4)
            feats = pts.getInfo()["features"][: catalogue.N_POINTS]  # numPixels counts pixels drawn before the cropland mask removes some
            for i, f in enumerate(feats):
                lon, lat = f["geometry"]["coordinates"]
                rows.append({"region": code, "point_id": i, "lon": lon, "lat": lat})
            log.add(f"{code}_area_km2", round(area_km2, 1))
            log.count(f"{code}_points", len(feats))
            print(code, f"{area_km2:.0f} km2", len(feats), "points", flush=True)
        write_csv(pd.DataFrame(rows), out, "02_sample_cropland", [catalogue.GEOB_URL, catalogue.WORLDCOVER])
        log.add("cropland_rule", "WorldCover v200 class 40 mean over 3x3 10 m cells >= 0.5 at a 30 m grid (EPSG:32632)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
