#!/usr/bin/env python
"""Per shipped TimeMatch date: fraction of cropland pixels flagged cloudy by
Cloud Score+ and s2cloudless, computed in Earth Engine. Resumable row by row."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.ee_init import init  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.timematch import eecloud  # noqa: E402

PAPER = "timematch"
OUT = paths.results_dir(PAPER) / "cloud_by_date.csv"
DATES = paths.results_dir(PAPER) / "timematch_dates.csv"
SUPPORT_KIND = "tile_bbox_x_worldcover40"


def main(smoke: bool = False, tiles: list[str] | None = None, force: bool = False) -> None:
    import ee

    with RunLog("02_ee_cloud_by_date", {"smoke": smoke, "tiles": tiles, "force": force}, paper=PAPER) as log:
        log.add("ee_project", init())
        log.add("collections", {"s2": eecloud.S2, "csplus": eecloud.CSPLUS, "s2cp": eecloud.S2CP, "cropland": eecloud.WORLDCOVER})
        log.add("thresholds", {"cs_cdf_clear_ge": eecloud.CS_CLEAR, "p40": eecloud.P40, "p16": eecloud.P16, "scale_m": eecloud.SCALE_M})
        dates = pd.read_csv(DATES)
        if tiles:
            dates = dates[dates["tile"].isin(tiles)]
        if smoke:
            dates = dates[dates["tile"] == "32VNH"].head(2)
        done = set()
        if OUT.exists() and not force:
            prev = pd.read_csv(OUT)
            done = set(zip(prev["tile"], prev["date"]))
        elif OUT.exists() and force:
            OUT.unlink()
        if not force and all((t, d) in done for t, d in zip(dates["tile"], dates["date"])):
            print("all dates already in results; nothing to do (use --force to redo)")
            log.discard()
            return
        cropland = eecloud.cropland_mask()
        supports = {}
        n_new, n_empty, n_failed = 0, 0, 0
        for (country, tile), grp in dates.groupby(["country", "tile"], sort=True):
            if tile not in supports:
                supports[tile] = eecloud.support_geometry(paths.paper_root(PAPER) / "footprints" / f"{country}_{tile}")
            support, crs = supports[tile]
            for _, r in grp.iterrows():
                if (tile, r["date"]) in done:
                    continue
                info, err = None, None
                for attempt in range(3):
                    try:
                        info = eecloud.cloud_fractions(eecloud.date_images(tile, r["date"]), support, cropland)
                        err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"
                        time.sleep(5 * 2**attempt)
                row = eecloud.row_from_info(country, tile, r["date"], float(r["tile_cloudy_pct"]), info, SUPPORT_KIND)
                if err:
                    row["n_images"] = -1
                    n_failed += 1
                    log.add(f"error_{tile}_{r['date']}", err)
                elif info is None:
                    n_empty += 1
                pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                n_new += 1
                print(f"{tile} {r['date']} n_img={row['n_images']} cs+={row['frac_cloudy_csplus']:.3f} p40={row['frac_cloudy_s2cp40']:.3f} nodata={row['frac_nodata']:.3f}", flush=True)
        log.count("rows_written", n_new)
        log.count("dates_without_images", n_empty)
        log.count("dates_failed", n_failed)
        if OUT.exists():
            log.count("rows_total", int(len(pd.read_csv(OUT))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
