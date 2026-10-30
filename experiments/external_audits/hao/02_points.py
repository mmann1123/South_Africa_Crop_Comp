#!/usr/bin/env python
"""Sample points for the four domains, seeded.

CONUS 2017: the paper's design on CDL 2017. A 100 km grid in EPSG:5070; in each cell up to 30
random pixels per crop (corn, soybean, cotton, spring wheat, canola, winter wheat incl. the
winter-wheat double-crop classes) with CDL confidence >= 95 and all 7 x 7 neighbours of the same
crop (fields over 4 ha). The pool is then cut to the class counts of Table 1 (8,448). The paper's
list of 294 HLS tiles is not given, so the cells cover all of CONUS and the tile set is whatever
those points fall in.
NE 2018: the same rule on CDL 2018 (corn, soybean) inside the paper's box.
AB 2018: AAFC Annual Crop Inventory 2018 canola and spring wheat inside the box (no confidence
layer exists), 7 x 7 homogeneous.
HS 2018: GFSAD30 cropland (the paper's own mask), 3 x 3 homogeneous, inside the box."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.hao import catalogue as cat, haolib  # noqa: E402

PAPER = cat.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
WORKERS = 20
# CONUS Albers extent of the lower 48 (EPSG:5070), rounded out to whole 100 km cells
X0, X1, Y0, Y1 = -2_400_000, 2_300_000, 200_000, 3_200_000


def homogeneous(crop, radius: int):
    return crop.focalMin(radius, "square", "pixels").eq(crop.focalMax(radius, "square", "pixels"))


def cdl_image(year: int, crops: list[str]):
    import ee

    codes = [c for k in crops for c in cat.CDL_CROPS[k]]
    ids = [cat.CROP_IDS[k] for k in crops for _ in cat.CDL_CROPS[k]]
    cdl = ee.Image(cat.CDL.format(year=year))
    crop = cdl.select("cropland").remap(codes, ids, 0)
    ok = crop.gt(0).And(cdl.select("confidence").gte(cat.CONF_MIN)).And(homogeneous(crop, cat.FIELD_RADIUS_PX))
    return crop.updateMask(ok).rename("crop").toByte()


def sample_cell(img, x: int, y: int, ids: list[int]) -> list[dict]:
    import ee

    reg = ee.Geometry.Rectangle([x, y, x + cat.GRID_M, y + cat.GRID_M], cat.GRID_CRS, False)
    fc = img.stratifiedSample(numPoints=0, classBand="crop", region=reg, scale=30, seed=cat.SEED, classValues=ids, classPoints=[cat.PER_CELL] * len(ids), geometries=True, tileScale=8)
    for attempt in range(4):
        try:
            feats = fc.getInfo()["features"]
            return [{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1], "crop_id": int(f["properties"]["crop"]), "cell_x": x, "cell_y": y} for f in feats]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"cell {x},{y} failed: {str(e)[:120]}", flush=True)
                return []
            import time

            time.sleep(10 * 2**attempt)


def conus_points() -> pd.DataFrame:
    crops = list(cat.CDL_CROPS)
    img = cdl_image(cat.TRAIN_YEAR, crops)
    ids = [cat.CROP_IDS[c] for c in crops]
    cache = ROOT / "cdl_cells"
    cache.mkdir(exist_ok=True)
    cells = [(x, y) for x in range(X0, X1, cat.GRID_M) for y in range(Y0, Y1, cat.GRID_M)]

    def work(c):
        f = cache / f"cell_{c[0]}_{c[1]}.json"
        if f.exists():
            return json.loads(f.read_text())
        rows = sample_cell(img, c[0], c[1], ids)
        f.write_text(json.dumps(rows))
        return rows

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = [r for part in ex.map(work, cells) for r in part]
    df = pd.DataFrame(rows)
    inv = {v: k for k, v in cat.CROP_IDS.items()}
    df["crop"] = df["crop_id"].map(inv)
    return df


def region_points(domain: str) -> pd.DataFrame:
    import ee

    box = ee.Geometry.Rectangle(cat.REGIONS[domain]["box"], "EPSG:4326", False)
    if domain == "NE":
        img = cdl_image(cat.TARGET_YEAR, cat.REGIONS[domain]["crops"])
        share = {"corn": 0.6, "soybean": 0.4}
        crops = cat.REGIONS[domain]["crops"]
        ids = [cat.CROP_IDS[c] for c in crops]
    elif domain == "AB":
        aci = ee.ImageCollection(cat.ACI).filterDate("2018-01-01", "2019-01-01").first().select("landcover")
        crops = cat.REGIONS[domain]["crops"]
        ids = [1, 2]
        crop = aci.remap([cat.ACI_CODES[c] for c in crops], ids, 0)
        img = crop.updateMask(crop.gt(0).And(homogeneous(crop, cat.FIELD_RADIUS_PX))).rename("crop").toByte()
        share = {"canola": 0.5, "spring_wheat": 0.5}
    else:  # HS: cropland only, no crop labels exist
        g = ee.ImageCollection(cat.GFSAD).filterBounds(box).mosaic().select("b1").eq(cat.GFSAD_CROP)
        img = g.updateMask(homogeneous(g, 1)).selfMask().rename("crop").toByte()
        crops, ids, share = ["cropland"], [1], {"cropland": 1.0}
    n = cat.N_POINTS[domain]
    fc = img.stratifiedSample(numPoints=0, classBand="crop", region=box, scale=30, seed=cat.SEED, classValues=ids, classPoints=[int(n * share[c]) for c in crops], geometries=True, tileScale=8)
    feats = fc.getInfo()["features"]
    inv = dict(zip(ids, crops))
    return pd.DataFrame([{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1], "crop": inv[int(f["properties"]["crop"])]} for f in feats])


def main(force: bool = False) -> None:
    out = RES / "points.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    from pyproj import Transformer

    with RunLog("02_points", {"force": force}, paper=PAPER) as log:
        log.add("ee_project", ee_init.init())
        conus = conus_points()
        log.count("conus_pool_after_cell_cap", len(conus))
        log.add("conus_pool_by_crop", conus["crop"].value_counts().to_dict())
        tr = Transformer.from_crs("EPSG:4326", cat.GRID_CRS, always_xy=True)
        x, y = tr.transform(conus["lon"].to_numpy(), conus["lat"].to_numpy())
        conus["cell"] = haolib.grid_cell(x, y, cat.GRID_M)
        conus = haolib.cap_per_cell(conus, "cell", "crop", cat.PER_CELL, cat.SEED)
        conus = haolib.subsample_to_counts(conus, "crop", cat.TABLE1, cat.SEED).reset_index(drop=True)
        conus["domain"] = "CONUS"
        frames = [conus[["domain", "crop", "lon", "lat", "cell"]]]
        for d in ("NE", "AB", "HS"):
            p = region_points(d)
            p["domain"], p["cell"] = d, -1
            frames.append(p[["domain", "crop", "lon", "lat", "cell"]])
            log.count(f"points_{d}", len(p))
        df = pd.concat(frames, ignore_index=True)
        df["chunk"] = -1
        for d in df["domain"].unique():
            m = df["domain"] == d
            df.loc[m, "chunk"] = haolib.assign_chunks(df.loc[m, "lon"], df.loc[m, "lat"])
        df.insert(0, "pid", np.arange(len(df)))
        write_csv(df, out, "02_points", [cat.CDL, cat.ACI, cat.GFSAD])
        log.count("conus_points", int((df["domain"] == "CONUS").sum()))
        print(df.groupby(["domain", "crop"]).size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
