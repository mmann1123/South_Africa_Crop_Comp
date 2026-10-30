#!/usr/bin/env python
"""Georeferenced footprints of a seeded random sample of patches per area, read from one label
raster (y.tif) each. The Kaggle download endpoint throttles hard and answers 404 when it does, so
every file is retried with back-off; a file that never arrives is recorded as failed."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.gallo import gallolib  # noqa: E402

PAPER = gallolib.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
N_PER_AREA = 30
REF_YEAR = 2019  # footprint read from this year's label raster; one other year is cross-checked


def label_patches() -> dict:
    """Patch ids with a y.tif per (area, year) from the saved listing."""
    out = defaultdict(set)
    for line in open(ROOT / "listing.tsv"):
        p = gallolib.parse_name(line.split("\t")[0])
        if p and p["kind"] == "label":
            out[(p["area"], p["year"])].add(p["patch"])
    return out


def get_file(area_dir: str, year: int, patch: int, name: str, dest: Path, tries: int = 80) -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = gallolib.file_url(area_dir, year, patch, name)
    for attempt in range(tries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):
            dest.write_bytes(r.content)
            return True
        time.sleep(min(30, 1 + attempt))
    return False


def main(force: bool = False) -> None:
    out = RES / "patch_footprints.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    inv = {v: k for k, v in gallolib.AREAS.items()}
    rng = np.random.default_rng(42)
    with RunLog("03_footprints", {"force": force, "n_per_area": N_PER_AREA}, paper=PAPER) as log:
        avail = label_patches()
        rows, shas = [], []
        for label in gallolib.AREAS.values():
            ids = sorted(avail[(label, REF_YEAR)])
            pick = sorted(rng.choice(ids, size=min(N_PER_AREA, len(ids)), replace=False).tolist())
            for p in pick:
                dest = ROOT / "footprints" / f"{label}_{REF_YEAR}_{p}_y.tif"
                if not get_file(inv[label], REF_YEAR, p, "y", dest):
                    rows.append({"area": label, "patch": p, "status": "failed"})
                    continue
                with rasterio.open(dest) as ds:
                    b = ds.bounds
                    lon, lat = rasterio.warp.transform(ds.crs, "EPSG:4326", [(b.left + b.right) / 2], [(b.bottom + b.top) / 2])
                    rows.append({"area": label, "patch": p, "status": "ok", "crs": str(ds.crs), "left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top, "lon": lon[0], "lat": lat[0], "height": ds.height, "width": ds.width, "bands": ds.count})
                shas.append({"key": dest.name, "url": gallolib.file_url(inv[label], REF_YEAR, p, "y"), "bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            print(label, sum(r["area"] == label and r["status"] == "ok" for r in rows), flush=True)
        man = paths.manifest_path(PAPER)
        old = [e for e in fetch.manifest_entries(man) if e["key"] not in {s["key"] for s in shas}]
        man.write_text(json.dumps(old + shas, indent=2))
        write_csv(pd.DataFrame(rows), out, "03_footprints", [gallolib.BASE])
        log.count("ok", sum(r["status"] == "ok" for r in rows))
        log.count("failed", sum(r["status"] == "failed" for r in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
