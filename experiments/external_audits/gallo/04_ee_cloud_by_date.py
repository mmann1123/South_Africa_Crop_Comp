#!/usr/bin/env python
"""Per deposit date and area-year: Cloud Score+ cloudy and no-data fractions over the sampled patch
footprints (WorldCover cropland and all pixels), computed in Earth Engine. Resumable row by row."""
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
from experiments.external_audits.gallo import eecloud  # noqa: E402

PAPER = "gallo"
RES = paths.results_dir(PAPER)
OUT = RES / "cloud_by_date.csv"


def main(areas: list[str] | None = None, force: bool = False) -> None:
    with RunLog("04_ee_cloud_by_date", {"areas": areas, "force": force}, paper=PAPER) as log:
        log.add("ee_project", init())
        log.add("collections", {"s2": eecloud.S2, "csplus": eecloud.CSPLUS, "cropland": eecloud.WORLDCOVER})
        log.add("thresholds", {"cs_cdf_clear_ge": eecloud.CS_CLEAR, "scale_m": eecloud.SCALE_M, "match_offset_days_max": 1})
        dates = pd.read_csv(RES / "dates_by_area_year.csv", dtype={"date": str})
        fp = pd.read_csv(RES / "patch_footprints.csv")
        fp = fp[fp["status"] == "ok"]
        if areas:
            dates = dates[dates["area"].isin(areas)]
        done = set()
        if OUT.exists() and force:
            OUT.unlink()
        if OUT.exists():
            prev = pd.read_csv(OUT, dtype={"date": str})
            done = set(zip(prev["area"], prev["year"], prev["date"]))
        todo = dates[[(a, y, d) not in done for a, y, d in zip(dates["area"], dates["year"], dates["date"])]]
        if todo.empty:
            print("all dates already in results; nothing to do (use --force to redo)")
            log.discard()
            return
        cropland = eecloud.cropland_mask()
        n_new = n_failed = 0
        for area, g in todo.groupby("area", sort=True):
            sub = fp[fp["area"] == area]
            support = eecloud.support_geometry(list(zip(sub["left"], sub["bottom"], sub["right"], sub["top"], sub["crs"])))
            for year, gy in g.groupby("year", sort=True):
                ee_dates = set(eecloud.sensing_dates(support, int(year)))
                log.add(f"{area}_{year}_ee_dates", len(ee_dates))
                rows = []
                for d in gy["date"]:
                    ee_date, off = eecloud.match_date(d, ee_dates)
                    info, err = None, None
                    if ee_date is not None:
                        for attempt in range(3):
                            try:
                                info = eecloud.cloud_fractions(eecloud.date_images(support, ee_date), support, cropland)
                                err = None
                                break
                            except Exception as e:  # noqa: BLE001
                                err = e
                                time.sleep(5 * 2**attempt)
                    if err is not None:
                        n_failed += 1
                        print(f"FAILED {area} {year} {d}: {err}", flush=True)
                        continue
                    rows.append(eecloud.row_from_info(area, int(year), d, ee_date, off, info))
                    n_new += 1
                if rows:
                    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(area, year, len(rows), "dates", flush=True)
        log.count("rows_new", n_new)
        log.count("failed", n_failed)
        if OUT.exists():
            from experiments.external_audits.common.tables import write_csv
            write_csv(pd.read_csv(OUT, dtype={"date": str}), OUT, "04_ee_cloud_by_date", [eecloud.S2, eecloud.CSPLUS, eecloud.WORLDCOVER])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
