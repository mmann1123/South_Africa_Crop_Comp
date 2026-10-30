#!/usr/bin/env python
"""Seeded sample points for both domains.

Source: per state (Iowa, Missouri, Kansas, Minnesota, North Dakota, South Dakota), up to 60 random CDL 2022 pixels
of each of the paper's five crops where the CDL confidence band is above 95. Target: random ESA WorldCover v200
cropland pixels (class 40) inside the five Gansu prefectures of the Hexi Corridor and the paper's stated extent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.hexi import catalogue as C  # noqa: E402

RES = paths.results_dir(C.PAPER)


def source_points(state: str, seed: int) -> pd.DataFrame:
    import ee

    geom = ee.FeatureCollection(C.GAUL1).filter(ee.Filter.eq("ADM0_NAME", "United States of America")).filter(ee.Filter.eq("ADM1_NAME", state)).geometry()
    cdl = ee.ImageCollection(C.CDL).filter(ee.Filter.eq("system:index", "2022")).first()
    img = cdl.select("cropland").updateMask(cdl.select("confidence").gt(C.CDL_CONF_MIN))
    codes = list(C.CDL_CLASSES)
    fc = img.stratifiedSample(numPoints=0, classBand="cropland", region=geom, scale=C.SAMPLE_SCALE_M, seed=seed, classValues=codes, classPoints=[C.POINTS_PER_STATE_CLASS] * len(codes), geometries=True, tileScale=4)
    rows = []
    for f in fc.getInfo()["features"]:
        lon, lat = f["geometry"]["coordinates"]
        rows.append({"group": state, "domain": "source", "crop": C.CDL_CLASSES[int(f["properties"]["cropland"])], "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


def target_region():
    import ee

    gansu = ee.FeatureCollection(C.GAUL2).filter(ee.Filter.eq("ADM0_NAME", "China")).filter(ee.Filter.eq("ADM1_NAME", C.GANSU))
    cities = gansu.filter(ee.Filter.inList("ADM2_NAME", C.HEXI_PREFECTURES))
    return cities.geometry().intersection(ee.Geometry.Rectangle(list(C.PAPER_EXTENT)), 1000)


def target_points(seed: int) -> pd.DataFrame:
    import ee

    geom = target_region()
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map")
    img = wc.updateMask(wc.eq(C.WC_CROPLAND))
    fc = img.stratifiedSample(numPoints=0, classBand="Map", region=geom, scale=C.SAMPLE_SCALE_M, seed=seed, classValues=[C.WC_CROPLAND], classPoints=[C.N_TARGET_POINTS], geometries=True, tileScale=4)
    rows = []
    for f in fc.getInfo()["features"]:
        lon, lat = f["geometry"]["coordinates"]
        rows.append({"group": C.TARGET, "domain": "target", "crop": "cropland", "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    out = RES / "points.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("01_points", {"force": force}, paper=C.PAPER) as log:
        log.add("ee_project", ee_init.init())
        parts = []
        for i, s in enumerate(C.STATES):
            df = source_points(s, C.SEED + i)
            print(s, len(df), flush=True)
            parts.append(df)
        parts.append(target_points(C.SEED))
        pts = pd.concat(parts).reset_index(drop=True)
        # sorted by group so that a chunk of points stays geographically compact
        pts.insert(0, "pid", range(len(pts)))
        write_csv(pts, out, "01_points", [C.CDL, C.GAUL1, C.GAUL2, C.WORLDCOVER])
        for (g, c), n in pts.groupby(["group", "crop"]).size().items():
            log.count(f"{g}|{c}", int(n))
        print(pts.groupby("group").size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
