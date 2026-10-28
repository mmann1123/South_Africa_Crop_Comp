#!/usr/bin/env python
"""Clustered stratified pixel samples of the You et al. 2021 recipe's validity
and zero-fill counters for the six DACCN regions, 2018 and 2019, variants A and B.

Design: a whole-state request exceeds Earth Engine's interactive memory limit
and a batch export of 300 state-wide points ran for over 40 minutes, so each
region-year-variant is sampled as N_CELLS random 5 km cells (fixed seed) with a
stratified sample of points inside each cell, computed interactively per cell
with retries. Cells are the same for both years and both variants of a region.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.ee_init import init  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.daccn import regions, you2021_recipe  # noqa: E402

PAPER = "daccn"
RES = paths.results_dir(PAPER)
OUT = RES / "daccn_counters.csv"
YEARS = (2018, 2019)
VARIANTS = ("A", "B")
N_CELLS = 16  # random 5 km cells per region
CELL_HALF_M = 2500
POINTS_PER_CELL = 300  # split equally across classes present in the frame
SEED = 42
SCALE = 10
WORKERS = 4
STEP_COLS = [f"{p}_{k:02d}" for p in ("valid", "zero") for k in range(22)]


def region_cells(region_name: str, country: str) -> list:
    """N_CELLS square cells (ee.Geometry) centred on seeded random points in the region."""
    import ee

    geom = regions.lookup(region_name, country).geometry()
    pts = ee.FeatureCollection.randomPoints(region=geom, points=N_CELLS, seed=SEED)
    cells = pts.map(lambda f: f.setGeometry(f.geometry().buffer(CELL_HALF_M).bounds()))
    return [ee.Geometry(c["geometry"]) for c in cells.getInfo()["features"]]


def sample_cell(cell, cell_id: int, region_name: str, country: str, year: int, variant: str) -> pd.DataFrame:
    import ee

    cls, frame, legend = regions.frame_image(country, year)
    per_class = max(1, POINTS_PER_CELL // len(legend))
    img = you2021_recipe.counters(cell, year, variant).addBands(cls)
    fc = img.stratifiedSample(numPoints=per_class, classBand="class", region=cell, scale=SCALE, seed=SEED + cell_id, geometries=False, tileScale=4, dropNulls=True)
    feats = None
    for attempt in range(4):
        try:
            feats = fc.getInfo()["features"]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(10 * 2**attempt)
    recs = []
    for f in feats:
        p = f["properties"]
        recs.append({"region": region_name, "country": country, "year": year, "variant": variant, "frame": frame, "cell": cell_id, "class": legend.get(int(p["class"]), str(p["class"])), "n_valid": p["n_valid"], "n_zero": p["n_zero"], **{c: p[c] for c in STEP_COLS}})
    return pd.DataFrame(recs)


def run_job(region_name: str, country: str, year: int, variant: str, cells: list, n_cells: int) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        parts = list(ex.map(lambda ic: sample_cell(ic[1], ic[0], region_name, country, year, variant), list(enumerate(cells))[:n_cells]))
    return pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()


def main(smoke: bool = False, force: bool = False, regions_only: list[str] | None = None) -> None:
    init()
    if OUT.exists() and force:
        OUT.unlink()
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev["region"], prev["year"], prev["variant"]))
    jobs = [(r, c, y, v) for (r, c) in regions.REGIONS for y in YEARS for v in VARIANTS]
    if regions_only:
        jobs = [j for j in jobs if j[0] in regions_only]
    if smoke:
        jobs = [("Iowa", regions.USA, 2018, "A"), ("Iowa", regions.USA, 2018, "B")]
    with RunLog("02_run_counters", {"smoke": smoke, "force": force, "regions_only": regions_only}, paper=PAPER) as log:
        log.add("collections", {"s2": you2021_recipe.S2, "gaul": regions.GAUL1, "cdl": regions.CDL, "worldcover": regions.WORLDCOVER})
        log.add("recipe", {"cloud_score_threshold": you2021_recipe.CLOUD_SCORE_THRESHOLD, "targets": you2021_recipe.TARGETS, "search": you2021_recipe.SEARCH, "variants": you2021_recipe.VARIANT_WINDOWS})
        log.add("cdl_classes", {"named": regions.CDL_CLASSES, "other_pooled": regions.CDL_OTHER, "confidence_min": regions.CONFIDENCE_MIN})
        log.add("design", {"n_cells": N_CELLS, "cell_half_m": CELL_HALF_M, "points_per_cell": POINTS_PER_CELL, "seed": SEED, "scale": SCALE})
        cells_by_region: dict[str, list] = {}
        n_jobs = 0
        for region_name, country, year, variant in jobs:
            if (region_name, year, variant) in done and not smoke:
                continue
            if region_name not in cells_by_region:
                cells_by_region[region_name] = region_cells(region_name, country)
                log.add(f"cells_{region_name}", [c.bounds().getInfo()["coordinates"][0][0] for c in cells_by_region[region_name]])
            t = time.time()
            df = run_job(region_name, country, year, variant, cells_by_region[region_name], 2 if smoke else N_CELLS)
            if smoke:
                print(f"{region_name} {year} {variant}: {len(df)} points in {time.time() - t:.0f} s")
                print(df.groupby("class")[["n_valid", "n_zero"]].agg(["mean", "count"]).round(2))
                print({c: round(v, 3) for c, v in df[[c for c in STEP_COLS if c.startswith('zero')]].mean().items() if v > 0})
                continue
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            n_jobs += 1
            log.add(f"{region_name}_{year}_{variant}", {"points": int(len(df)), "cells_with_points": int(df["cell"].nunique()) if len(df) else 0, "seconds": round(time.time() - t)})
            print(f"{region_name} {year} {variant}: {len(df)} points from {df['cell'].nunique() if len(df) else 0} cells, mean n_valid {df['n_valid'].mean():.2f}, share any zero {(df['n_zero'] > 0).mean():.3f} ({time.time() - t:.0f} s)", flush=True)
        log.count("jobs_run", n_jobs)
        if OUT.exists():
            log.count("rows_total", int(len(pd.read_csv(OUT))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--regions", nargs="*", default=None, dest="regions_only")
    main(**vars(ap.parse_args()))
