#!/usr/bin/env python
"""Reproducible sample of forest pixels over Flanders: seeded random points inside the FAO GAUL
'Vlaams Gewest' polygon, kept where ESA WorldCover v200 (2021) is tree cover (class 10)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.verhulst import catalogue  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
BATCH = 5000
N_RANDOM = 40000  # about 10 percent of Flanders is tree cover


def flanders():
    import ee

    fc = ee.FeatureCollection(catalogue.GAUL).filter(ee.Filter.eq("ADM1_NAME", catalogue.GAUL_NAME))
    return fc.geometry()


def main(force: bool = False) -> None:
    out = RES / "forest_points.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    import ee

    with RunLog("02_forest_points", {"force": force, "n_points": catalogue.N_POINTS}, paper=PAPER) as log:
        log.add("ee_project", ee_init.init())
        region = flanders()
        wc = ee.ImageCollection(catalogue.WORLDCOVER).first().select("Map")
        log.add("flanders_area_km2", region.area().getInfo() / 1e6)
        rows = []
        for i in range(N_RANDOM // BATCH):
            pts = ee.FeatureCollection.randomPoints(region, BATCH, catalogue.SEED + i)
            s = wc.sampleRegions(pts, scale=catalogue.SCALE_M, geometries=True, tileScale=4)
            for f in s.getInfo()["features"]:
                lon, lat = f["geometry"]["coordinates"]
                rows.append({"lon": lon, "lat": lat, "worldcover": f["properties"]["Map"]})
            print(f"batch {i}: {len(rows)} sampled")
        df = pd.DataFrame(rows)
        log.add("worldcover_class_share", df["worldcover"].value_counts(normalize=True).round(4).to_dict())
        forest = df[df["worldcover"] == catalogue.TREE_CLASS].reset_index(drop=True)
        forest = forest.sample(n=min(catalogue.N_POINTS, len(forest)), random_state=catalogue.SEED).reset_index(drop=True)
        forest.insert(0, "pid", range(len(forest)))
        write_csv(forest, out, "02_forest_points", [catalogue.GAUL, catalogue.WORLDCOVER])
        log.count("random_points", len(df))
        log.count("forest_points", len(forest))
        print(len(forest), "forest points")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
