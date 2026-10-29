#!/usr/bin/env python
"""Per-step snow flags on the valid composites of the You et al. 2021 recipe, on the same
random 5 km cells and stratified points as 02_run_counters.py (same cell generator, seed, per-class
point count and per-cell sampling seed), for regions x {2018, 2019} x variants {A, B}.

Output: results/daccn_snow_counters.csv (resumable by region-year-variant; --force redoes) and
results/daccn_snow_l2a_coverage.csv (scene counts of L1C, L2A and MOD10A1 per region-year).
Row alignment with daccn_counters.csv is verified per job (valid_kk must match) and logged.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.ee_init import init  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.daccn import regions, snowrule, you2021_snow  # noqa: E402

rc = importlib.import_module("experiments.external_audits.daccn.02_run_counters")
PAPER = "daccn"
RES = paths.results_dir(PAPER)
OUT = RES / "daccn_snow_counters.csv"
COV = RES / "daccn_snow_l2a_coverage.csv"
YEARS, VARIANTS = rc.YEARS, rc.VARIANTS
KEYS = snowrule.VALID + snowrule.SNOW + snowrule.SCL + snowrule.MODIS
FAILED: list[dict] = []


def sample_cell(cell, cell_id, region_name, country, year, variant) -> pd.DataFrame:
    cls, frame, legend = regions.frame_image(country, year)
    per_class = max(1, rc.POINTS_PER_CELL // len(legend))
    img = you2021_snow.snow_image(cell, year, variant).addBands(cls)
    fc = img.stratifiedSample(numPoints=per_class, classBand="class", region=cell, scale=rc.SCALE, seed=rc.SEED + cell_id, geometries=False, tileScale=rc.TILE_SCALE, dropNulls=True)
    feats = None
    for attempt in range(4):
        try:
            feats = fc.getInfo()["features"]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                FAILED.append({"region": region_name, "year": year, "variant": variant, "cell": cell_id, "error": f"{type(e).__name__}: {e}"[:200]})
                return pd.DataFrame()
            time.sleep(10 * 2**attempt)
    recs = [{"region": region_name, "year": year, "variant": variant, "cell": cell_id, "class": legend.get(int(f["properties"]["class"]), str(f["properties"]["class"])), **{c: f["properties"][c] for c in KEYS}} for f in feats]
    return pd.DataFrame(recs)


def alignment(df: pd.DataFrame, region: str, year: int, variant: str) -> dict:
    """Compare this job's valid_kk with daccn_counters.csv rows of the same job, cell by cell."""
    ref_path = RES / "daccn_counters.csv"
    if not ref_path.exists() or df.empty:
        return {"checked": False}
    ref = pd.read_csv(ref_path, usecols=["region", "year", "variant", "cell", "class"] + snowrule.VALID)
    ref = ref[(ref["region"] == region) & (ref["year"] == year) & (ref["variant"] == variant)]
    cells = sorted(set(df["cell"]) & set(ref["cell"]))
    same = tot = 0
    for c in cells:
        a, b = df[df["cell"] == c].reset_index(drop=True), ref[ref["cell"] == c].reset_index(drop=True)
        tot += 1
        if len(a) == len(b) and (a["class"] == b["class"]).all() and (a[snowrule.VALID].values == b[snowrule.VALID].values).all():
            same += 1
    return {"checked": True, "cells_in_both": tot, "cells_identical_valid_rows": same, "cells_snow_only": len(set(df["cell"]) - set(ref["cell"])), "cells_ref_only": len(set(ref["cell"]) - set(df["cell"]))}


def main(smoke: bool = False, force: bool = False, regions_only: list[str] | None = None) -> None:
    init()
    if force:
        for p in (OUT, COV):
            if p.exists():
                p.unlink()
    done = set()
    if OUT.exists() and not smoke:
        prev = pd.read_csv(OUT, usecols=["region", "year", "variant"])
        done = set(zip(prev["region"], prev["year"], prev["variant"]))
    jobs = [(r, c, y, v) for (r, c) in regions.REGIONS for y in YEARS for v in VARIANTS]
    if regions_only:
        jobs = [j for j in jobs if j[0] in regions_only]
    if smoke:
        jobs = [("Minnesota", regions.USA, 2018, "B")]
    cov_done = pd.read_csv(COV) if COV.exists() and not smoke else pd.DataFrame(columns=["region", "year"])
    todo_cov = [(r, c, y) for (r, c) in regions.REGIONS for y in YEARS if not ((cov_done["region"] == r) & (cov_done["year"] == y)).any() and (not regions_only or r in regions_only)]
    if not smoke and not todo_cov and all((r, y, v) in done for r, c, y, v in jobs):
        print("all jobs already in results; nothing to do (use --force to redo)")
        return
    with RunLog("04_snow_counters", {"smoke": smoke, "force": force, "regions_only": regions_only}, paper=PAPER) as log:
        log.add("collections", {"l1c": you2021_snow.R.S2, "l2a_scl": you2021_snow.S2_SR, "modis": you2021_snow.MODIS})
        log.add("snow_rule", {"ndsi_min": snowrule.NDSI_MIN, "nir_min": snowrule.NIR_MIN, "green_min": snowrule.GREEN_MIN, "modis_min": snowrule.MODIS_MIN, "scl_class": 11, "ndsi": "(B3 - B11)/(B3 + B11) of the 10-day median composite"})
        log.add("design", {"n_cells": rc.N_CELLS, "cell_half_m": rc.CELL_HALF_M, "points_per_cell": rc.POINTS_PER_CELL, "seed": rc.SEED, "scale": rc.SCALE, "same_cells_as": "02_run_counters.py"})
        cells_by_region: dict = {}
        get_cells = lambda rn, ct: cells_by_region.setdefault(rn, rc.region_cells(rn, ct))  # noqa: E731
        # scene-count coverage over the sampled cells (the state polygon is too heavy for filterBounds counts)
        cov_rows = []
        if not smoke:
            import ee

            for r, c, y in todo_cov:
                geom = ee.Geometry.MultiPolygon([ee.Geometry(x).coordinates() for x in get_cells(r, c)])
                cov_rows.append({"region": r, "country": c, "year": y, **{f"n_{k}": v for k, v in you2021_snow.l2a_coverage(geom, y).items()}})
                print("coverage", cov_rows[-1], flush=True)
            if cov_rows:
                write_csv(pd.concat([cov_done, pd.DataFrame(cov_rows)], ignore_index=True), COV, "04_snow_counters", [you2021_snow.R.S2, you2021_snow.S2_SR, you2021_snow.MODIS])
        n_jobs = 0
        for region_name, country, year, variant in jobs:
            if (region_name, year, variant) in done:
                continue
            cells = get_cells(region_name, country)
            t = time.time()
            ncell = 2 if smoke else rc.N_CELLS
            with ThreadPoolExecutor(max_workers=rc.WORKERS) as ex:
                parts = list(ex.map(lambda ic: sample_cell(ic[1], ic[0], region_name, country, year, variant), list(enumerate(cells))[:ncell]))
            df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
            if df.empty:
                log.add(f"{region_name}_{year}_{variant}", {"points": 0, "note": "every cell failed; left pending for a resume"})
                continue
            al = alignment(df, region_name, year, variant)
            n_snow = (df[snowrule.SNOW].values * df[snowrule.VALID].values).sum(axis=1)
            print(f"{region_name} {year} {variant}: {len(df)} pts, any snow {(n_snow > 0).mean():.3f}, mean snow steps {n_snow.mean():.2f}, align {al} ({time.time() - t:.0f} s)", flush=True)
            if smoke:
                continue
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            n_jobs += 1
            log.add(f"{region_name}_{year}_{variant}", {"points": int(len(df)), "cells": int(df["cell"].nunique()), "seconds": round(time.time() - t), "alignment_with_counters": al})
        log.count("jobs_run", n_jobs)
        log.add("failed_cells", FAILED)
        log.count("failed_cells", len(FAILED))
        if OUT.exists():
            log.count("rows_total", int(len(pd.read_csv(OUT, usecols=["region"]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--regions", nargs="*", default=None, dest="regions_only")
    main(**vars(ap.parse_args()))
