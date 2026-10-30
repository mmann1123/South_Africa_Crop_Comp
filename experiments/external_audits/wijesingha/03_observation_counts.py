#!/usr/bin/env python
"""Per cropland point, year and bimonth: valid Landsat observations (scene CLOUD_COVER < 60,
eemont maskClouds bits, every SR band unmasked so SLC-off stripes and fill count as no data) per
sensor and in total, and whether the bimonthly median is empty. Resumable by region and year."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.wijesingha import catalogue, wijesingha_lib as lib  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
OUT = RES / "obs_counts_points.csv.gz"


def valid_flag(img, sr_bands: list[str], cirrus: bool):
    """1 where the pixel survives maskClouds (QA_PIXEL bits 3 cloud, 4 shadow, plus 2 cirrus on
    Landsat 8) and every SR band used downstream is unmasked, else 0."""
    import ee

    qa = img.select("QA_PIXEL")
    ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    if cirrus:
        ok = ok.And(qa.bitwiseAnd(1 << 2).eq(0))
    present = img.select(sr_bands).mask().reduce(ee.Reducer.min())
    return ok.And(present).unmask(0).rename("v").toUint8()


def year_image(year: int):
    import ee

    bands = []
    for m in catalogue.BIMONTH_STARTS:
        start, end = lib.bimonth_window(year, m)
        total = None
        for s in lib.SENSOR_ORDER:
            cfg = catalogue.SENSORS[s]
            col = (
                ee.ImageCollection(cfg["id"])
                .filterDate(start, end)
                .filterMetadata("CLOUD_COVER", "less_than", catalogue.MAX_CLOUD_COVER)
            )
            zero = ee.ImageCollection([ee.Image.constant(0).rename("v").toUint8()])
            n = col.map(lambda i, cfg=cfg: valid_flag(i, cfg["sr"], cfg["cirrus"])).merge(zero).sum().rename(f"n_{s}_m{m:02d}")
            bands.append(n)
            total = n if total is None else total.add(n)
        bands.append(total.rename(f"n_all_m{m:02d}"))
    return ee.Image.cat(bands).toInt16()


def main(force: bool = False) -> None:
    import ee

    ee_init.init()
    pts = pd.read_csv(RES / "cropland_points.csv")
    done = pd.read_csv(OUT) if OUT.exists() and not force else pd.DataFrame()
    todo = [(r, y) for r in catalogue.REGION_NAME for y in catalogue.YEARS if done.empty or not ((done["region"] == r) & (done["year"] == y)).any()]
    if not todo:
        print(f"{OUT} complete; use --force")
        return
    cols = lib.count_columns(catalogue.BIMONTH_STARTS)
    with RunLog("03_observation_counts", {"force": force, "todo": len(todo)}, paper=PAPER) as log:
        parts = [done] if not done.empty else []
        for region, year in todo:
            t0 = time.time()
            p = pts[pts["region"] == region]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": int(r.point_id)}) for r in p.itertuples()])
            for attempt in range(4):
                try:
                    feats = year_image(year).sampleRegions(collection=fc, scale=30, tileScale=4, geometries=False).getInfo()["features"]
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    print("retry", region, year, e, flush=True)
                    time.sleep(10 * 2**attempt)
            df = pd.DataFrame([f["properties"] for f in feats])
            df.insert(0, "year", year)
            df.insert(0, "region", region)
            parts.append(df[["region", "year", "point_id", *cols]])
            pd.concat(parts).to_csv(OUT, index=False)  # written after every region-year so a rerun resumes
            print(region, year, len(df), "of", len(p), "points sampled", f"{time.time() - t0:.0f}s", flush=True)
        log.add("sensors", {s: c["id"] for s, c in catalogue.SENSORS.items()})
        log.add("rule", "valid = QA_PIXEL clear (bits 3,4; plus 2 on L8) and all six SR bands unmasked; scene CLOUD_COVER < 60")
        log.count("rows", int(sum(len(x) for x in parts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
