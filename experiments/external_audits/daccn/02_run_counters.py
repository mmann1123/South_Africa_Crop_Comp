#!/usr/bin/env python
"""Stratified pixel samples of the You et al. 2021 recipe's validity and
zero-fill counters for the six DACCN regions, 2018 and 2019, variants A and B.

Interactive requests over a whole state exceed Earth Engine's per-request
memory limit, so each job is a batch table export to an Earth Engine asset
(larger budget), read back with getInfo once complete. Assets are deleted
after they are read unless --keep-assets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
TASKS = RES / "ee_tasks.json"
ASSET_FOLDER = "projects/mmann1123-malawi/assets/external_audits_daccn"  # Cloud project id of EE project 327687702789
YEARS = (2018, 2019)
VARIANTS = ("A", "B")
N_POINTS = 4800  # total per job; getInfo on an asset is capped at 5000 features
SEED = 42
SCALE = 10
STEP_COLS = [f"{p}_{k:02d}" for p in ("valid", "zero") for k in range(22)]


def job_name(region_name: str, year: int, variant: str) -> str:
    return f"{region_name.lower()}_{year}_{variant}"


def build_collection(region_name: str, country: str, year: int, variant: str, n_points: int):
    feat = regions.lookup(region_name, country)
    geom = feat.geometry()
    cls, frame, legend = regions.frame_image(country, year)
    per_class = max(1, n_points // len(legend))
    points = cls.stratifiedSample(numPoints=per_class, classBand="class", region=geom, scale=SCALE, seed=SEED, geometries=True, tileScale=4, dropNulls=True)
    img = you2021_recipe.counters(geom, year, variant).addBands(cls)
    fc = img.sampleRegions(collection=points, scale=SCALE, tileScale=16, geometries=False)
    return fc, frame, legend, per_class


def submit(region_name: str, country: str, year: int, variant: str, n_points: int) -> dict:
    import ee

    fc, frame, legend, per_class = build_collection(region_name, country, year, variant, n_points)
    name = job_name(region_name, year, variant)
    asset = f"{ASSET_FOLDER}/{name}"
    try:
        ee.data.deleteAsset(asset)
    except Exception:  # noqa: BLE001
        pass
    task = ee.batch.Export.table.toAsset(collection=fc, description=name, assetId=asset)
    task.start()
    return {"region": region_name, "country": country, "year": year, "variant": variant, "frame": frame, "legend": {str(k): v for k, v in legend.items()}, "per_class": per_class, "task_id": task.id, "asset": asset, "state": "SUBMITTED"}


def collect(job: dict, keep_assets: bool) -> pd.DataFrame:
    import ee

    feats = ee.FeatureCollection(job["asset"]).getInfo()["features"]
    legend = {int(k): v for k, v in job["legend"].items()}
    recs = []
    for f in feats:
        p = f["properties"]
        recs.append({"region": job["region"], "country": job["country"], "year": job["year"], "variant": job["variant"], "frame": job["frame"], "class": legend.get(int(p["class"]), str(p["class"])), "n_valid": p["n_valid"], "n_zero": p["n_zero"], **{c: p[c] for c in STEP_COLS}})
    if not keep_assets:
        try:
            ee.data.deleteAsset(job["asset"])
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(recs)


def main(smoke: bool = False, force: bool = False, keep_assets: bool = False, poll_s: int = 60) -> None:
    import ee

    init()
    if OUT.exists() and force:
        OUT.unlink()
    if TASKS.exists() and force:
        TASKS.unlink()
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev["region"], prev["year"], prev["variant"]))
    jobs = [(r, c, y, v) for (r, c) in regions.REGIONS for y in YEARS for v in VARIANTS]
    if smoke:
        jobs = [("Iowa", regions.USA, 2018, "A")]
    with RunLog("02_run_counters", {"smoke": smoke, "force": force, "keep_assets": keep_assets}, paper=PAPER) as log:
        log.add("collections", {"s2": you2021_recipe.S2, "gaul": regions.GAUL1, "cdl": regions.CDL, "worldcover": regions.WORLDCOVER})
        log.add("recipe", {"cloud_score_threshold": you2021_recipe.CLOUD_SCORE_THRESHOLD, "targets": you2021_recipe.TARGETS, "search": you2021_recipe.SEARCH, "variants": you2021_recipe.VARIANT_WINDOWS})
        log.add("cdl_classes", {"named": regions.CDL_CLASSES, "other_pooled": regions.CDL_OTHER, "confidence_min": regions.CONFIDENCE_MIN})
        log.add("asset_folder", ASSET_FOLDER)
        pending = json.loads(TASKS.read_text()) if TASKS.exists() and not smoke else []
        pending = [j for j in pending if (j["region"], j["year"], j["variant"]) not in done]
        have = {(j["region"], j["year"], j["variant"]) for j in pending}
        for region_name, country, year, variant in jobs:
            if (region_name, year, variant) in done or (region_name, year, variant) in have:
                continue
            j = submit(region_name, country, year, variant, 300 if smoke else N_POINTS)
            pending.append(j)
            print(f"submitted {j['task_id']} {job_name(region_name, year, variant)}", flush=True)
        if not smoke:
            TASKS.write_text(json.dumps(pending, indent=1))
        n_collected = 0
        while pending:
            time.sleep(poll_s)
            states = {t["id"]: t for t in ee.data.getTaskList()}
            still = []
            for j in pending:
                st = states.get(j["task_id"], {}).get("state", "UNKNOWN")
                if st == "COMPLETED":
                    df = collect(j, keep_assets)
                    if smoke:
                        print(df.groupby("class")[["n_valid", "n_zero"]].agg(["mean", "count"]))
                        print(df[[c for c in STEP_COLS if c.startswith("zero")]].mean().round(3).to_dict())
                        return
                    df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                    n_collected += 1
                    log.add(job_name(j["region"], j["year"], j["variant"]), {"points": int(len(df)), "per_class_requested": j["per_class"], "frame": j["frame"], "legend": j["legend"], "task_id": j["task_id"]})
                    print(f"{j['region']} {j['year']} {j['variant']}: {len(df)} points, mean n_valid {df['n_valid'].mean():.2f}, share any zero {(df['n_zero'] > 0).mean():.3f}", flush=True)
                elif st in ("FAILED", "CANCELLED"):
                    log.add(f"failed_{job_name(j['region'], j['year'], j['variant'])}", states[j["task_id"]].get("error_message", st))
                    print(f"FAILED {j['region']} {j['year']} {j['variant']}: {states[j['task_id']].get('error_message', st)}", flush=True)
                else:
                    still.append(j)
            pending = still
            if not smoke:
                TASKS.write_text(json.dumps(pending, indent=1))
        log.count("jobs_collected", n_collected)
        if OUT.exists():
            log.count("rows_total", int(len(pd.read_csv(OUT))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-assets", action="store_true", dest="keep_assets")
    ap.add_argument("--poll-s", type=int, default=60, dest="poll_s")
    main(**vars(ap.parse_args()))
