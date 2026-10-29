#!/usr/bin/env python
"""List every HLS v2.0 (L30 + S30) acquisition of tile 16TBL for 2017-2019 from
Earth Engine, with its granule cloud cover, and check that the Fmask cirrus bit
(bit 0) is never set on a spread of images (it is reserved in v2.0)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.ee_init import init  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.kerner import kernerlib  # noqa: E402

PAPER = "kerner"
RES = paths.results_dir(PAPER)
OUT = RES / "kerner_hls_acquisitions.csv"
BIT_OUT = RES / "kerner_fmask_bit_check.csv"
COLLECTIONS = {"S": "NASA/HLS/HLSS30/v002", "L": "NASA/HLS/HLSL30/v002"}
YEARS = (2017, 2018, 2019)
BIT_CHECK_PER_YEAR = 8


def tile_collection(sensor: str, year: int):
    import ee

    # The bounds filter makes the query fast; the index prefix makes it exact.
    side = kernerlib.TILE_RES * 3660
    rect = ee.Geometry.Rectangle([kernerlib.TILE_ULX, kernerlib.TILE_ULY - side, kernerlib.TILE_ULX + side, kernerlib.TILE_ULY], kernerlib.TILE_CRS, False)
    return ee.ImageCollection(COLLECTIONS[sensor]).filterBounds(rect).filterDate(f"{year}-01-01", f"{year + 1}-01-01").filter(ee.Filter.stringStartsWith("system:index", f"T{kernerlib.TILE}_")).sort("system:time_start")


def fetch_year(sensor: str, year: int) -> pd.DataFrame:
    import ee

    col = tile_collection(sensor, year)
    info = col.reduceColumns(
        ee.Reducer.toList(4),
        ["system:index", "CLOUD_COVERAGE", "SPATIAL_COVERAGE", "system:time_start"],
    ).get("list").getInfo()
    rows = []
    for idx, cc, sc, ts in info:
        name = f"{sensor}_{idx}"
        _, _, y, doy = kernerlib.parse_acquisition(name)
        rows.append({"year": year, "sensor": sensor, "image": name, "system_index": idx, "doy": doy, "cloud_coverage_pct": cc, "spatial_coverage_pct": sc, "time_start_ms": ts})
    return pd.DataFrame(rows)


def bit_check(acq: pd.DataFrame) -> pd.DataFrame:
    import ee

    rows = []
    for year, g in acq.groupby("year"):
        g = g[g["spatial_coverage_pct"] > 0].reset_index(drop=True)
        picks = g.iloc[[int(i) for i in pd.Series(range(BIT_CHECK_PER_YEAR)).mul(len(g) - 1).div(BIT_CHECK_PER_YEAR - 1).round()]].drop_duplicates("image")
        for _, r in picks.iterrows():
            ic = ee.ImageCollection(COLLECTIONS[r["sensor"]]).filter(ee.Filter.eq("system:index", r["system_index"]))
            im = ee.Image(ic.first()).select("Fmask")
            hist = im.reduceRegion(ee.Reducer.frequencyHistogram(), im.geometry(), 300, maxPixels=1e9).get("Fmask").getInfo() or {}
            tot = sum(hist.values())
            bit0 = sum(v for k, v in hist.items() if int(k) & 1)
            rows.append({"year": year, "image": r["image"], "doy": r["doy"], "sampled_cells_300m": round(tot, 1), "bit0_set_cells": round(bit0, 1), "bit0_share": bit0 / tot if tot else float("nan"), "fmask_values": ";".join(sorted(hist, key=int))})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    if OUT.exists() and BIT_OUT.exists() and not force:
        print("acquisition list and bit check already present; nothing to do (use --force to redo)")
        return
    with RunLog("02_hls_acquisitions", {"force": force}, paper=PAPER) as log:
        log.add("ee_project", init())
        log.add("collections", COLLECTIONS)
        acq = pd.concat([fetch_year(s, y) for y in YEARS for s in COLLECTIONS], ignore_index=True)
        acq = acq.sort_values(["year", "time_start_ms", "sensor"]).reset_index(drop=True)
        write_csv(acq, OUT, "02_hls_acquisitions", list(COLLECTIONS.values()))
        for (y, s), g in acq.groupby(["year", "sensor"]):
            log.count(f"n_{y}_{s}30", int(len(g)))
        bits = bit_check(acq)
        write_csv(bits, BIT_OUT, "02_hls_acquisitions", list(COLLECTIONS.values()))
        log.add("bit0_set_cells_total", float(bits["bit0_set_cells"].sum()))
        print(acq.groupby(["year", "sensor"]).size().unstack())
        print(bits[["year", "image", "bit0_set_cells", "fmask_values"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
