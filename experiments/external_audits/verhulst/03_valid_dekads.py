#!/usr/bin/env python
"""Valid dekads per forest pixel and year (2018-2022) from Sentinel-2 L2A SCL in Earth Engine.

Per image: usable = SCL in {4, 5}; the unusable pixels (inside the footprint) are blurred with the
11 x 11 Gaussian kernel (sigma 1.5) and a pixel is dropped where the blurred share reaches a
threshold (variants none / d3 / d1, see catalogue.VARIANTS). Per dekad a pixel is valid if at least
one image has it usable, which is when a dekadal median exists. Sampled at 10 m in UTM 31N at the
seeded forest points; resumable by (year, chunk)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.verhulst import catalogue, verhulstlib as vl  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
CHUNK = 500


def year_image(year: int, region):
    """36 dekads x (1 observation count + one valid count per variant) bands, named d<k>_<variant>."""
    import ee

    kernel = ee.Kernel.gaussian(radius=catalogue.KERNEL_RADIUS, sigma=catalogue.KERNEL_SIGMA, units="pixels", normalize=True)
    coll = ee.ImageCollection(catalogue.EE_PRODUCT).filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    def per_image(img):
        scl = img.select("SCL").unmask(-1)
        inside = scl.gte(0)
        usable = scl.eq(4).Or(scl.eq(5))
        unusable = inside.And(usable.Not()).float()
        blur = unusable.convolve(kernel)
        bands = [inside.rename("obs")]
        for name, thr in catalogue.VARIANTS.items():
            ok = usable if thr is None else usable.And(blur.lt(thr))
            bands.append(ok.rename(name))
        return ee.Image.cat(bands).toInt8().copyProperties(img, ["system:time_start"])

    mapped = coll.map(per_image)
    layers = []
    for k in range(catalogue.N_DEKADS):
        a, b = vl.dekad_bounds(year, k)
        s = mapped.filterDate(a.isoformat(), b.isoformat()).sum()
        layers.append(s.rename([f"d{k:02d}_{n}" for n in ["obs", *catalogue.VARIANTS]]))
    return ee.Image.cat(layers), coll.size()


def main(force: bool = False, years: list[int] | None = None) -> None:
    import ee

    pts = pd.read_csv(RES / "forest_points.csv")
    years = years or catalogue.YEARS
    cache = ROOT / "chunks"
    cache.mkdir(exist_ok=True)
    done_any = False
    with RunLog("03_valid_dekads", {"force": force, "years": years}, paper=PAPER) as log:
        log.add("ee_project", ee_init.init())
        region = ee.FeatureCollection(catalogue.GAUL).filter(ee.Filter.eq("ADM1_NAME", catalogue.GAUL_NAME)).geometry()
        log.add("variants", catalogue.VARIANTS)
        for year in years:
            img, n_img = year_image(year, region)
            log.add(f"n_images_{year}", n_img.getInfo())
            for c0 in range(0, len(pts), CHUNK):
                f = cache / f"valid_{year}_{c0:05d}.csv.gz"
                if f.exists() and not force:
                    continue
                done_any = True
                sub = pts.iloc[c0 : c0 + CHUNK]
                fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"pid": int(r.pid)}) for r in sub.itertuples()])
                feats = img.sampleRegions(fc, scale=catalogue.SCALE_M, projection=catalogue.CRS, tileScale=4).getInfo()["features"]
                df = pd.DataFrame([{"pid": x["properties"]["pid"], **{k: v for k, v in x["properties"].items() if k != "pid"}} for x in feats])
                df.to_csv(f, index=False)
                print(year, c0, len(df), flush=True)
        if not done_any:
            log.discard()
            print("nothing to do; use --force")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*")
    main(**vars(ap.parse_args()))
