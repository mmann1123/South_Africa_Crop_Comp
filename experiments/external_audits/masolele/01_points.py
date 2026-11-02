#!/usr/bin/env python
"""Sample points: up to PIXELS_PER_UNIT Hansen loss pixels (2001-2005) inside the 10 x 10 km unit at
every tropical 1 degree confluence point, per continent, as a stand-in for the De Sy et al. 2019
FRA-2010 units (not public). Units with no 2001-2005 loss give no points. Resumable by continent."""
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
OUT = RES / "points.csv"


def sample_units(units: list[tuple[int, int]]):
    import ee

    lo, hi = cat.LOSS_YEARS
    ly = ee.Image(cat.HANSEN).select("lossyear")
    loss = ly.updateMask(ly.gte(lo).And(ly.lte(hi))).rename("lossyear")

    def one(f):
        s = loss.sample(region=f.geometry(), scale=30, numPixels=20000, seed=cat.SEED, geometries=True, tileScale=4)
        s = s.randomColumn("r", cat.SEED).sort("r").limit(cat.PIXELS_PER_UNIT)
        return s.map(lambda p: p.set({"unit_lon": f.get("lon"), "unit_lat": f.get("lat")}))

    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(lib.unit_box(lon, lat, cat.BOX_HALF_DEG)), {"lon": lon, "lat": lat}) for lon, lat in units])
    return fc.map(one).flatten()


def main(force: bool = False) -> None:
    ee_init.init()
    done = pd.read_csv(OUT) if OUT.exists() and not force else pd.DataFrame()
    with RunLog("01_points", {"force": force}, paper=cat.PAPER) as log:
        parts = [done] if not done.empty else []
        for cont, lon_range in cat.CONTINENTS.items():
            if not done.empty and (done["continent"] == cont).any():
                continue
            units = lib.confluence_points(lon_range, cat.LAT_MAX)
            rows, t0 = [], time.time()
            for i in range(0, len(units), 150):
                chunk = units[i : i + 150]
                for attempt in range(4):
                    try:
                        feats = sample_units(chunk).getInfo()["features"]
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 3:
                            raise
                        print("retry", cont, i, e, flush=True)
                        time.sleep(10 * 2**attempt)
                for f in feats:
                    lon, lat = f["geometry"]["coordinates"]
                    pr = f["properties"]
                    rows.append({"continent": cont, "unit_lon": pr["unit_lon"], "unit_lat": pr["unit_lat"], "lon": lon, "lat": lat, "lossyear": 2000 + int(pr["lossyear"])})
                print(cont, f"units {i + len(chunk)}/{len(units)}", f"points {len(rows)}", f"{time.time() - t0:.0f}s", flush=True)
            parts.append(pd.DataFrame(rows))
            out = pd.concat(parts, ignore_index=True)
            out["point_id"] = range(len(out))
            out.to_csv(OUT, index=False)
            log.count(f"units_{cont}", len(units))
            log.count(f"points_{cont}", len(rows))
        log.add("hansen", cat.HANSEN)
        log.add("rule", f"lossyear {cat.LOSS_YEARS}, |lat| <= {cat.LAT_MAX}, unit half-width {cat.BOX_HALF_DEG} deg, {cat.PIXELS_PER_UNIT} pixels per unit, seed {cat.SEED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
