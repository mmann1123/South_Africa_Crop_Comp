#!/usr/bin/env python
"""Cloud Score+ per parcel and shipped date for the FR1 (30TXT) parcels in the union
of the three folds' test sets of the DK1-trained source-only model (the DK1 to FR1
direction), plus block-level cropland fractions for FR1 and DK1.

For each parcel and shipped date: the fraction of the parcel's 10 m pixels with
cs_cdf < t for t in {0.5, 0.6, 0.7}, from the mosaic of the S2_HARMONIZED granules of
MGRS tile 30TXT on that UTC date (cs_cdf linked by system:index). Parcels with no
same-day coverage get NaN. Chunks of CHUNK parcels are saved under the data root and
the script resumes from them. Denmark ships no parcel polygons, so DK1 only gets the
block-level table (block polygons x WorldCover cropland)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common.ee_init import init  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch import eecloud  # noqa: E402
from experiments.external_audits.timematch_rerun import rerunlib  # noqa: E402

PAPER = "timematch_rerun"
ROOT = Path("/mnt/bigdrive/external_audits/timematch_rerun")
RES = Path(__file__).resolve().parent / "results"
THRS = (50, 60, 70)
CHUNK = 500
SRC, TGT = "denmark/32VNH/2017", "france/30TXT/2017"


def iso(d: int) -> str:
    s = str(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def fr1_test_union() -> np.ndarray:
    folds = rerunlib.make_folds(60332, 115755, SRC, TGT, seed=1)
    u = set()
    for f in folds:
        u |= f[TGT]["test"]
    return np.array(sorted(u))


def parcel_chunk(ee, geoms, idxs, dates, tile="30TXT"):
    fc = ee.FeatureCollection([ee.Feature(g, {"idx": int(i)}) for g, i in zip(geoms, idxs)])
    bands = []
    for d in dates:
        cs = eecloud.date_images(tile, iso(d)).select("cs_cdf").mosaic()
        for t in THRS:
            bands.append(cs.lt(t / 100).rename(f"d{d}_t{t}"))
    img = ee.Image.cat(bands)
    out = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4).getInfo()
    T = len(dates)
    arr = np.full((len(idxs), len(THRS), T), np.nan)
    pos = {int(i): k for k, i in enumerate(idxs)}
    for f in out["features"]:
        p = f["properties"]
        k = pos[int(p["idx"])]
        for ti, t in enumerate(THRS):
            for di, d in enumerate(dates):
                v = p.get(f"d{d}_t{t}")
                if v is not None:
                    arr[k, ti, di] = v
    return arr


def block_table(ee, tile_label, country, tile, dates, cropland):
    import geopandas as gpd

    blk = gpd.read_file(ROOT / "meta_fetch" / country / tile / "2017" / "meta" / "blocks" / f"blocks_{country}_{tile}_2017.shp")
    feats = []
    for r in blk.itertuples():
        minx, miny, maxx, maxy = r.geometry.bounds
        feats.append(ee.Feature(ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj=str(blk.crs), geodesic=False), {"block": int(r.id)}))
    fc = ee.FeatureCollection(feats)
    bands = []
    for d in dates:
        cs = eecloud.date_images(tile, iso(d)).select("cs_cdf").mosaic()
        for t in THRS:
            bands.append(cs.lt(t / 100).updateMask(cropland).rename(f"d{d}_t{t}"))
    out = ee.Image.cat(bands).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=60, tileScale=4).getInfo()
    rows = []
    for f in out["features"]:
        p = f["properties"]
        for d in dates:
            rows.append({"tile": tile_label, "block": int(p["block"]), "date": iso(d)} | {f"frac_cloudy_t{t}": p.get(f"d{d}_t{t}") for t in THRS})
    return pd.DataFrame(rows)


def main(max_chunks: int | None = None, force: bool = False) -> None:
    import geopandas as gpd

    with RunLog("02_ee_parcel_cloud", {"max_chunks": max_chunks, "force": force}, paper=PAPER) as log:
        import ee

        log.add("ee_project", init())
        log.add("collections", {"s2": eecloud.S2, "csplus": eecloud.CSPLUS})
        log.add("thresholds_cs_cdf_lt", [t / 100 for t in THRS])
        log.add("scale_m", 10)
        dates = json.loads((ROOT / "dates_FR1.json").read_text())
        union = fr1_test_union()
        log.count("fr1_parcels_in_test_union", len(union))
        g = gpd.read_parquet(ROOT / "fr1_parcels_wgs84.parquet")
        assert (g["idx"].values == np.arange(len(g))).all()
        cdir = ROOT / "ee_chunks"
        cdir.mkdir(exist_ok=True)
        n_chunks = (len(union) + CHUNK - 1) // CHUNK
        done = 0
        for c in range(n_chunks):
            f = cdir / f"fr1_chunk_{c:04d}.npz"
            if f.exists() and not force:
                done += 1
                continue
            if max_chunks is not None and done >= max_chunks:
                break
            idxs = union[c * CHUNK : (c + 1) * CHUNK]
            geoms = [ee.Geometry(json.loads(gpd.GeoSeries([g.geometry.iloc[i]]).to_json())["features"][0]["geometry"], None, False) for i in idxs]
            t0 = time.time()
            arr = None
            for attempt in range(4):
                try:
                    arr = parcel_chunk(ee, geoms, idxs, dates)
                    break
                except Exception as e:  # noqa: BLE001
                    print("retry", c, type(e).__name__, str(e)[:200], flush=True)
                    time.sleep(10 * 2**attempt)
            if arr is None:
                raise RuntimeError(f"chunk {c} failed")
            np.savez_compressed(f, idx=idxs, frac=arr.astype(np.float32))
            done += 1
            print(f"chunk {c + 1}/{n_chunks} {time.time() - t0:.1f}s", flush=True)
        log.count("chunks_done", done)
        log.count("chunks_total", n_chunks)
        # block-level tables
        out = RES / "block_cloud_by_date.csv"
        if not out.exists() or force:
            cropland = eecloud.cropland_mask()
            frames = [
                block_table(ee, "FR1", "france", "30TXT", dates, cropland),
                block_table(ee, "DK1", "denmark", "32VNH", json.loads((ROOT / "dates_DK1.json").read_text()), cropland),
            ]
            write_csv(pd.concat(frames, ignore_index=True), out, "02_ee_parcel_cloud", ["EE Cloud Score+ x WorldCover 2020 x block rectangles"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.max_chunks, a.force)
