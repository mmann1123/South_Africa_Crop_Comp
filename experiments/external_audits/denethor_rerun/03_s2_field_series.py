#!/usr/bin/env python
"""Field-mean Sentinel-2 series for the DENETHOR random-forest arm, built the
way the authors' s2dataset.setup() builds them (lukaskondmann/DENETHOR 55573a8):
fid raster with features.rasterize(zip(geometry, fid), all_touched=True) on the
bbox grid (later polygons overwrite earlier ones), every label polygon kept (no
area filter in S2Dataset), the field mean over its pixels of the 12 bands of
BANDS.npy and of CLP.npy for each of the 144 acquisitions.

Tile-years: 18E-242N 2018 (train), 17E-243N 2019 (test) and 17E-243N 2018 (the
test tile in the training year, labelled with the 2019 test labels, as the
paper's ablation does). BANDS.npy (19.9 GB each) is downloaded with resumable
range requests, reduced, and deleted. Output s2_<tile>_<year>.npz: X (F, 144,
12) float32 raw DN means, cld (F, 144) float32 mean CLP (0 to 255), fid,
crop_id, timestamps."""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio import features

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, zonal  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.denethor import s2clear  # noqa: E402

PAPER = "denethor_rerun"
BASE = "https://data.source.coop/esa/fusion-competition/sentinel-2/"
LABEL_OF = {"18E-242N": "18E-242N", "17E-243N": "17E-243N"}
TILE_YEARS = [("18E-242N", 2018), ("17E-243N", 2019), ("17E-243N", 2018), ("18E-242N", 2019)]
CLP_ONLY = {("18E-242N", 2019)}  # train tile in the test year: cloud series only, for density matching
T_CHUNK = 8


def fetch(url: str, dest: Path) -> Path:
    """Resumable download with Range requests."""
    size = int(requests.head(url, timeout=120, allow_redirects=True).headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == size:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(50):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        if have > size:
            part.unlink()
            have = 0
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=300) as r:
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"  # a 200 reply carries the whole file
                with open(part, mode) as fh:
                    for ch in r.iter_content(1 << 22):
                        fh.write(ch)
        except Exception as e:  # noqa: BLE001
            print(f"{dest.name}: retry {attempt} after {type(e).__name__}", flush=True)
            time.sleep(10)
    if part.stat().st_size != size:
        raise IOError(f"{url}: incomplete")
    part.replace(dest)
    return dest


def process(tile: str, year: int, log: RunLog, keep_bands: bool) -> None:
    out = paths.paper_root(PAPER) / f"s2_{tile}_{year}.npz"
    if out.exists():
        print(f"{out} exists")
        return
    ty = f"s2-utm-33N-{tile}-{year}"
    d = paths.paper_root(PAPER) / "s2" / ty
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clp_only = (tile, year) in CLP_ONLY
    want = ["bbox.pkl", "timestamp.pkl", "data/CLP.npy"] + ([] if clp_only else ["data/BANDS.npy"])
    files = {k: fetch(f"{BASE}{ty}/{k}", d / Path(k).name) for k in want}
    minx, miny, maxx, maxy, crs = s2clear.load_bbox(files["bbox.pkl"], tile, "33N")
    with open(files["timestamp.pkl"], "rb") as fh:
        ts = pd.to_datetime(pd.Series(pickle.load(fh)), utc=True).dt.tz_localize(None)
    clp = np.load(files["data/CLP.npy"], mmap_mode="r")
    bands = None if clp_only else np.load(files["data/BANDS.npy"], mmap_mode="r")
    T, width, height = clp.shape[:3]
    C = 0 if clp_only else bands.shape[3]
    labels = gpd.read_file(paths.paper_root(PAPER) / "labels" / f"{LABEL_OF[tile]}.geojson").to_crs(crs)
    transform = zonal.grid_from_bbox(minx, miny, maxx, maxy, size=width)
    fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True, transform=transform, out_shape=(width, height))
    fids = labels.fid.unique()
    crop = labels.crop_id.values  # the authors zip fid.unique() with crop_id.values
    pos = {f: i for i, f in enumerate(fids)}
    flat = fid_mask.ravel()
    lut = np.full(int(flat.max()) + 1, -1, np.int64)
    for f, i in pos.items():
        if f <= flat.max():
            lut[int(f)] = i
    idx = np.where(flat > 0, lut[flat], -1)
    n = len(fids)
    cnt = np.bincount(idx[idx >= 0], minlength=n).astype(np.float64)
    keep = cnt > 0
    X = np.zeros((n, T, C), np.float32)
    cld = np.zeros((n, T), np.float32)
    sel = idx >= 0
    ii = idx[sel]
    for t0_ in range(0, T, T_CHUNK):
        t1 = min(T, t0_ + T_CHUNK)
        b = None if clp_only else np.asarray(bands[t0_:t1]).reshape(t1 - t0_, -1, C)
        c = np.asarray(clp[t0_:t1]).reshape(t1 - t0_, -1)
        for t in range(t1 - t0_):
            for k in range(C):
                X[:, t0_ + t, k] = np.bincount(ii, weights=b[t, sel, k], minlength=n) / np.maximum(cnt, 1)
            cld[:, t0_ + t] = np.bincount(ii, weights=c[t, sel], minlength=n) / np.maximum(cnt, 1)
    np.savez(out, X=X[keep], cld=cld[keep], fid=fids[keep], crop_id=crop[keep], timestamps=ts.astype(str).values)
    write_csv(pd.DataFrame({"fid": fids[keep], "crop_id": crop[keep], "n_pixels": cnt[keep].astype(int)}),
              paths.results_dir(PAPER) / f"s2_fields_{tile}_{year}.csv", "03_s2_field_series", [f"{BASE}{ty}/data/BANDS.npy"])
    if not keep_bands and not clp_only:
        files["data/BANDS.npy"].unlink()
    log.add(f"{tile}_{year}", {"fields": int(keep.sum()), "fields_without_pixels": int((~keep).sum()), "acquisitions": int(T), "minutes": round((time.time() - t0) / 60, 1)})


def main(keep_bands: bool) -> None:
    with RunLog("03_s2_field_series", {"keep_bands": keep_bands}, paper=PAPER) as log:
        for tile, year in TILE_YEARS:
            process(tile, year, log, keep_bands)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-bands", action="store_true")
    main(**vars(ap.parse_args()))
