#!/usr/bin/env python
"""Per-image state read straight from the deposited rasters for a few patches per area-year: share of
no-data (all bands zero) pixels and of bright (blue above 2000, cloud or snow) pixels. This is what the
network receives, since no cloud mask was applied. Resumable; the download endpoint throttles, so each
file is retried with back-off and polite spacing."""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.gallo import gallolib  # noqa: E402

PAPER = gallolib.PAPER
RES = paths.results_dir(PAPER)
RAW = RES / "pixel_state_raw.csv"
N_PATCHES = 3  # first N sampled footprints per area, same patch ids in every year


def read_tif(url: str, tries: int = 60) -> np.ndarray:
    for attempt in range(tries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):
            with rasterio.open(io.BytesIO(r.content)) as ds:
                return ds.read()
        time.sleep(min(30, 0.5 + attempt))
    raise RuntimeError(f"could not fetch {url}")


def main(force: bool = False) -> None:
    inv = {v: k for k, v in gallolib.AREAS.items()}
    with RunLog("05_pixel_check", {"force": force, "n_patches": N_PATCHES}, paper=PAPER) as log:
        dates = pd.read_csv(RES / "dates_by_area_year.csv", dtype={"date": str})
        fp = pd.read_csv(RES / "patch_footprints.csv")
        fp = fp[fp["status"] == "ok"]
        if RAW.exists() and force:
            RAW.unlink()
        done = set()
        if RAW.exists():
            prev = pd.read_csv(RAW, dtype={"date": str})
            done = set(zip(prev["area"], prev["year"], prev["patch"], prev["date"]))
        n_new = 0
        for area in gallolib.AREAS.values():
            patches = fp[fp["area"] == area]["patch"].head(N_PATCHES).tolist()
            for year in gallolib.YEARS:
                ds = dates[(dates["area"] == area) & (dates["year"] == year)]["date"].tolist()
                rows = []
                for p in patches:
                    for d in ds:
                        if (area, year, p, d) in done:
                            continue
                        arr = read_tif(gallolib.file_url(inv[area], year, p, d))
                        rows.append({"area": area, "year": year, "patch": p, "date": d, "mean_blue": float(arr[0].mean()), **gallolib.image_state(arr)})
                        time.sleep(0.2)
                if rows:
                    pd.DataFrame(rows).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
                    n_new += len(rows)
                print(area, year, len(rows), flush=True)
        write_csv(pd.read_csv(RAW, dtype={"date": str}), RAW, "05_pixel_check", [gallolib.BASE])
        log.count("images_new", n_new)
        log.add("cloud_blue_threshold", gallolib.CLOUD_BLUE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
