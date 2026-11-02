#!/usr/bin/env python
"""Per sample point and composite window (January-June, July-December, 2000-2005): valid Landsat 5 and 7
observations under the paper's rules (scene CLOUD_COVER < 50, QA_PIXEL clear, every SR band unmasked so
SLC-off stripes count as no data). A window with zero valid observations has a masked median composite.
Resumable by chunk of points."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.masolele import catalogue as cat, masolele_lib as lib  # noqa: E402

RES = paths.results_dir(cat.PAPER)
OUT = RES / "window_counts.csv.gz"
WINDOWS = lib.window_ids(cat.YEARS, cat.HALVES)


def valid_flag(img):
    import ee

    qa = img.select("QA_PIXEL")
    ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud, cloud shadow
    present = img.select(cat.SR).mask().reduce(ee.Reducer.min())
    return ok.And(present).unmask(0).rename("v").toUint8()


def count_image():
    import ee

    bands = []
    zero = ee.ImageCollection([ee.Image.constant(0).rename("v").toUint8()])
    for w in WINDOWS:
        start, end = lib.half_window(int(w[:4]), int(w[-1]))
        total = None
        for s in lib.SENSOR_ORDER:
            col = ee.ImageCollection(cat.SENSORS[s]).filterDate(start, end).filter(ee.Filter.lt("CLOUD_COVER", cat.MAX_CLOUD_COVER))
            n = col.map(valid_flag).merge(zero).sum().rename(f"n_{s}_{w}")
            bands.append(n)
            total = n if total is None else total.add(n)
        bands.append(total.rename(f"n_all_{w}"))
    return ee.Image.cat(bands).toInt16()


def main(force: bool = False) -> None:
    import ee

    ee_init.init()
    pts = pd.read_csv(RES / "points.csv")
    done = pd.read_csv(OUT) if OUT.exists() and not force else pd.DataFrame()
    have = set(done["point_id"]) if not done.empty else set()
    todo = pts[~pts["point_id"].isin(have)]
    if todo.empty:
        print(f"{OUT} complete; use --force")
        return
    cols = lib.count_columns(WINDOWS)
    img = count_image()
    with RunLog("02_window_counts", {"force": force, "todo": len(todo)}, paper=cat.PAPER) as log:
        parts = [done] if not done.empty else []
        t0 = time.time()
        for i in range(0, len(todo), cat.CHUNK):
            p = todo.iloc[i : i + cat.CHUNK]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": int(r.point_id)}) for r in p.itertuples()])
            for attempt in range(4):
                try:
                    feats = img.sampleRegions(collection=fc, scale=30, tileScale=4, geometries=False).getInfo()["features"]
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    print("retry", i, e, flush=True)
                    time.sleep(10 * 2**attempt)
            df = pd.DataFrame([f["properties"] for f in feats])
            df = p[["continent", "point_id"]].merge(df, on="point_id", how="left")
            parts.append(df[["continent", "point_id", *cols]])
            pd.concat(parts, ignore_index=True).to_csv(OUT, index=False)
            print(f"{i + len(p)}/{len(todo)} points, {df[cols].isna().any(axis=1).sum()} unsampled, {time.time() - t0:.0f}s", flush=True)
        log.add("sensors", cat.SENSORS)
        log.add("rule", "valid = QA_PIXEL bits 3 (cloud) and 4 (shadow) clear and all six SR bands unmasked; scene CLOUD_COVER < 50")
        log.count("rows", int(sum(len(x) for x in parts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
