#!/usr/bin/env python
"""Does QA60 still flag cloud after the January 2022 processing-baseline change? Per Sentinel-2 image of 2021,
2022 and 2023 over each domain: share of the sample points under the image footprint where QA60 flags opaque
cloud or cirrus, the Cloud Score+ cloud share at the same points, and the ESA granule CLOUDY_PIXEL_PERCENTAGE.
Also whether QA60 has a value at all (Earth Engine masks it out in this period). Runs for the L1C and the L2A collection. Points: the first 60 of each group in points.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.hexi import catalogue as C, eelib  # noqa: E402

RES = paths.results_dir(C.PAPER)
ROOT = paths.paper_root(C.PAPER)
N_PTS = 60
DEFAULT_GROUPS = [C.TARGET, "Iowa", "North Dakota"]
PRODUCTS = {"L1C": C.S2_L1C, "L2A": C.S2_SR}


def month_images(product: str, geom, year: int, month: int) -> pd.DataFrame:
    import ee

    start = ee.Date.fromYMD(year, month, 1)
    coll = eelib.linked(year, geom, start, start.advance(1, "month"), product)

    def per_image(img):
        inside = img.select("B2").mask().unmask(0).gt(0)
        cloud = img.select("QA60").unmask(0).toInt32().bitwiseAnd(sum(1 << b for b in C.QA60_CLOUD_BITS)).neq(0)
        cs_cloud = img.select("cs_cdf").unmask(0).lt(C.CS_THRESHOLDS[C.CS_PRIMARY])
        has = img.select("QA60").mask().unmask(0).gt(0)  # QA60 is masked out (no value) in Earth Engine for part of 2022 to 2024
        stack = ee.Image.cat([cloud.rename("qa60"), cs_cloud.rename("cs"), has.rename("qa60has")]).updateMask(inside)
        red = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
        d = stack.reduceRegion(red, geom, 60, maxPixels=1e8, tileScale=4)
        return ee.Feature(None, {"id": img.get("system:index"), "qa60_frac": d.get("qa60_mean"), "qa60_any": d.get("qa60_max"), "cs_frac": d.get("cs_mean"), "qa60_has": d.get("qa60has_mean"), "cloudy_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"), "date": img.date().format("YYYY-MM-dd")})

    feats = coll.map(per_image).filter(ee.Filter.notNull(["qa60_frac"])).getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in feats])


def main(force: bool = False, groups: list[str] | None = None) -> None:
    import ee

    out = RES / "qa60_images.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    pts = pd.read_csv(RES / "points.csv")
    cache = ROOT / "qa60_images"
    cache.mkdir(exist_ok=True)
    with RunLog("03_qa60_images", {"force": force, "n_points": N_PTS}, paper=C.PAPER) as log:
        log.add("ee_project", ee_init.init())
        for group, g in pts.groupby("group", sort=False):
            if groups and group not in groups:
                continue
            sub = g.head(N_PTS)
            geom = ee.Geometry.MultiPoint([[r.lon, r.lat] for r in sub.itertuples()])
            for pname, product in PRODUCTS.items():
                for year in C.YEARS:
                    for month in range(1, 13):
                        f = cache / f"{group.replace(' ', '_')}_{pname}_{year}_{month:02d}.csv"
                        if f.exists() and not force:
                            continue
                        df = month_images(product, geom, year, month)
                        df.to_csv(f, index=False)
                        print(group, pname, year, month, len(df), flush=True)
        frames = []
        for f in sorted(cache.glob("*.csv")):
            group, pname, year, month = f.stem.rsplit("_", 3)
            d = pd.read_csv(f)
            if len(d):
                frames.append(d.assign(group=group.replace("_", " "), product=pname, year=int(year), month=int(month)))
        allimg = pd.concat(frames)
        allimg["domain"] = (allimg["group"] == C.TARGET).map({True: "target", False: "source"})
        allimg.to_csv(ROOT / "qa60_images_all.csv.gz", index=False)  # per-image rows stay in the data root
        agg = (
            allimg.assign(flag_any=allimg["qa60_any"] > 0)
            .groupby(["product", "domain", "year", "month"])
            .agg(n_images=("id", "size"), share_images_qa60_flag=("flag_any", "mean"), mean_qa60_frac=("qa60_frac", "mean"), qa60_present_share=("qa60_has", "mean"), mean_cs_cloud_frac=("cs_frac", "mean"), mean_cloudy_pct=("cloudy_pct", "mean"))
            .reset_index()
        )
        write_csv(agg, out, "03_qa60_images", [C.S2_L1C, C.S2_SR, C.CSPLUS])
        log.count("images", len(allimg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--groups", nargs="*", help="default: Hexi, Iowa and North Dakota (all 7 groups x 2 products x 3 years x 12 months is slow)")
    a = ap.parse_args()
    main(a.force, a.groups or DEFAULT_GROUPS)
