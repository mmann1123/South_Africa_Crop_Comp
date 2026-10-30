#!/usr/bin/env python
"""Per-point weekly counts of Sentinel-2 L1C images (COPERNICUS/S2_HARMONIZED) that are inside the footprint, pass
QA60, are Cloud Score+ clear, and pass both, for 52 seven-day bins of 2021, 2022 and 2023, at the seeded source
and target points. Resumable by (year, week block, chunk of points)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.hexi import catalogue as C, eelib  # noqa: E402

RES = paths.results_dir(C.PAPER)
ROOT = paths.paper_root(C.PAPER)


def chunks(pts: pd.DataFrame) -> list[pd.DataFrame]:
    """Chunks of at most C.CHUNK points that never span two groups and are cut along longitude (keeps each call geographically compact)."""
    out = []
    for _, g in pts.groupby("group", sort=False):
        g = g.sort_values(["lon", "lat"])
        out += [g.iloc[i : i + C.CHUNK] for i in range(0, len(g), C.CHUNK)]
    return out


def main(force: bool = False, years: list[int] | None = None) -> None:
    import ee

    pts = pd.read_csv(RES / "points.csv")
    years = years or C.YEARS
    cache = ROOT / "chunks"
    cache.mkdir(exist_ok=True)
    done_any = False
    with RunLog("02_weekly_counts", {"force": force, "years": years}, paper=C.PAPER) as log:
        log.add("ee_project", ee_init.init())
        log.add("collections", {"s2": C.S2_L1C, "cloud_score_plus": C.CSPLUS})
        log.add("cs_thresholds", C.CS_THRESHOLDS)
        for year in years:
            for ci, sub in enumerate(chunks(pts)):
                for w0 in range(0, C.N_WEEKS, C.WEEK_BLOCK):
                    weeks = list(range(w0, min(w0 + C.WEEK_BLOCK, C.N_WEEKS)))
                    f = cache / f"counts_{year}_c{ci:02d}_w{w0:02d}.csv.gz"
                    if f.exists() and not force:
                        continue
                    done_any = True
                    xy = [[r.lon, r.lat] for r in sub.itertuples()]
                    geom = ee.Geometry.MultiPoint(xy)
                    img = eelib.weekly_counts(year, geom, weeks)
                    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(p), {"pid": int(r.pid)}) for p, r in zip(xy, sub.itertuples())])
                    feats = img.sampleRegions(fc, ["pid"], scale=C.SCALE_M, tileScale=8).getInfo()["features"]
                    pd.DataFrame([x["properties"] for x in feats]).to_csv(f, index=False)
                    print(year, ci, w0, len(feats), flush=True)
        if not done_any:
            log.discard()
            print("nothing to do; use --force")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*")
    main(**vars(ap.parse_args()))
