#!/usr/bin/env python
"""Extract fixed per-field pixel sets from the Planet Fusion daily cubes of the
two DENETHOR benchmark tile-years (18E-242N 2018 train, 17E-243N 2019 test).

Per field kept by the authors' dataset.py (polygon area > 1000 m2 in the label
CRS), the polygon is rasterised with all_touched=True on the 3 m PF grid, as in
dataset.py, and 128 pixel positions are drawn once (without replacement when the
field has at least 128 pixels, with replacement otherwise; numpy seed 0 over the
fields in label order). For every day of the year the PF-SR GeoTIFF (4 bands,
int16, about 330 MB) and the PF-QA GeoTIFF are downloaded, the 128 x 4 values
and PF-QA layer 1 (percent synthetic) at the same pixels are kept, and both
files are deleted. Days run in parallel; each finished day is cached as an .npz,
so the script resumes. A final pass stacks the days into
pf_<tile>_<year>.npz: sr (365, F, 128, 4) int16, qa (365, F, 128) int16, plus
the field table. Storing 128 pixels rather than all (about 80 GB) is a
deviation; training still draws 20 per field per sample, as the authors do."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio import features
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402

PAPER = "denethor_rerun"
BASE = "https://data.source.coop/esa/fusion-competition/"
LABELS = {"18E-242N": "br-18E-242N-crop-labels-train-2018.geojson", "17E-243N": "br-17E-243N-crop-labels-test-2019.geojson"}
BENCHMARK = [("18E-242N", 2018), ("17E-243N", 2019)]
N_PIX = 128
MIN_AREA = 1000


def sr_url(tile: str, date: str) -> str:
    return f"{BASE}planet/UTM-24000/33N/{tile}/PF-SR/{date}.tif"


def qa_url(tile: str, date: str) -> str:
    return f"{BASE}planet/UTM-24000/33N/{tile}/PF-QA/{date}.tif"


def root() -> Path:
    return paths.paper_root(PAPER)


def label_frame(tile: str) -> gpd.GeoDataFrame:
    p = root() / "labels" / f"{tile}.geojson"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(BASE + LABELS[tile], timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    g = gpd.read_file(p)
    return g.loc[g.geometry.area > MIN_AREA].reset_index(drop=True)


def sample_positions(g: gpd.GeoDataFrame, crs, transform, shape, n_pix: int = N_PIX, seed: int = 0):
    """(F, n_pix) row and col indices on the full PF grid, and pixels per field."""
    gg = g.to_crs(crs)
    rng = np.random.default_rng(seed)
    rows = np.zeros((len(gg), n_pix), np.int32)
    cols = np.zeros((len(gg), n_pix), np.int32)
    npx = np.zeros(len(gg), np.int64)
    for i, geom in enumerate(gg.geometry):
        win = from_bounds(*geom.bounds, transform=transform).round_offsets(op="floor").round_lengths(op="ceil")
        r0, c0 = int(win.row_off), int(win.col_off)
        h, w = max(int(win.height), 1), max(int(win.width), 1)
        wt = rasterio.windows.transform(win, transform)
        m = features.rasterize([geom], out_shape=(h, w), transform=wt, all_touched=True, fill=0, default_value=1, dtype="uint8")
        rr, cc = np.nonzero(m)
        rr, cc = rr + r0, cc + c0
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[ok], cc[ok]
        npx[i] = len(rr)
        if len(rr) == 0:
            raise ValueError(f"field {g.fid.iloc[i]} covers no pixel")
        pick = rng.choice(len(rr), size=n_pix, replace=len(rr) < n_pix)
        rows[i], cols[i] = rr[pick], cc[pick]
    return rows, cols, npx


def _download(url: str, dest: Path) -> str:
    sha = hashlib.sha256()
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                exp = int(r.headers.get("Content-Length", 0)) or None
                n = 0
                with open(dest, "wb") as fh:
                    for ch in r.iter_content(1 << 22):
                        fh.write(ch)
                        sha.update(ch)
                        n += len(ch)
            if exp is not None and n != exp:
                raise IOError(f"{url}: {n} of {exp}")
            return sha.hexdigest()
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
            sha = hashlib.sha256()
    raise RuntimeError("unreachable")


def do_day(tile: str, date: str, rows: np.ndarray, cols: np.ndarray, cache: Path, tmp: Path) -> dict:
    out = cache / f"{date}.npz"
    if out.exists():
        z = np.load(out)
        return {"date": date, "sha_sr": str(z["sha_sr"]), "sha_qa": str(z["sha_qa"]), "cached": True}
    fs, fq = tmp / f"{tile}_{date}_sr.tif", tmp / f"{tile}_{date}_qa.tif"
    try:
        sha_sr = _download(sr_url(tile, date), fs)
        sha_qa = _download(qa_url(tile, date), fq)
        with rasterio.open(fs) as ds:
            arr = ds.read()  # (4, 8000, 8000) int16
        sr = np.ascontiguousarray(arr[:, rows, cols].transpose(1, 2, 0))  # (F, n_pix, 4)
        del arr
        with rasterio.open(fq) as ds:
            qa = ds.read(1)[rows, cols]
        np.savez(out.with_suffix(".tmp.npz"), sr=sr, qa=qa.astype(np.int16), sha_sr=sha_sr, sha_qa=sha_qa)
        out.with_suffix(".tmp.npz").replace(out)
    finally:
        fs.unlink(missing_ok=True)
        fq.unlink(missing_ok=True)
    return {"date": date, "sha_sr": sha_sr, "sha_qa": sha_qa, "cached": False}


def process(tile: str, year: int, workers: int, smoke: bool, log: RunLog) -> None:
    final = root() / f"pf_{tile}_{year}.npz"
    if final.exists():
        print(f"{final} exists")
        return
    g = label_frame(tile)
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")]
    if smoke:
        dates = dates[120:122]
    with rasterio.open("/vsicurl/" + sr_url(tile, dates[0])) as ds:
        crs, transform, shape = ds.crs, ds.transform, ds.shape
    pos = root() / f"positions_{tile}.npz"
    if pos.exists():
        z = np.load(pos)
        rows, cols, npx = z["rows"], z["cols"], z["npx"]
    else:
        rows, cols, npx = sample_positions(g, crs, transform, shape)
        np.savez(pos, rows=rows, cols=cols, npx=npx)
    cache = root() / "pf_days" / f"{tile}_{year}"
    cache.mkdir(parents=True, exist_ok=True)
    tmp = root() / "tmp"
    tmp.mkdir(exist_ok=True)
    recs, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(do_day, tile, d, rows, cols, cache, tmp) for d in dates]
        for i, f in enumerate(as_completed(futs), 1):
            recs.append(f.result())
            if i % 10 == 0:
                print(f"{tile} {year}: {i}/{len(dates)} days, {(time.time() - t0) / 60:.0f} min", flush=True)
    recs = pd.DataFrame(recs).sort_values("date")
    write_csv(recs.drop(columns="cached"), paths.results_dir(PAPER) / f"pf_day_manifest_{tile}_{year}.csv", "01_extract_pf", [BASE + "planet/UTM-24000/33N/<tile>/PF-{SR,QA}/<date>.tif"])
    if smoke:
        return
    sr = np.stack([np.load(cache / f"{d}.npz")["sr"] for d in dates])
    qa = np.stack([np.load(cache / f"{d}.npz")["qa"] for d in dates])
    fields = pd.DataFrame({"fid": g["fid"].values, "crop_id": g["crop_id"].values, "crop_name": g["crop_name"].values, "n_pixels": npx})
    np.savez(final, sr=sr, qa=qa, fid=fields["fid"].values, crop_id=fields["crop_id"].values, dates=np.array(dates))
    write_csv(fields, paths.results_dir(PAPER) / f"fields_{tile}_{year}.csv", "01_extract_pf", [BASE + LABELS[tile]])
    log.add(f"{tile}_{year}", {"fields": int(len(g)), "days": len(dates), "minutes": round((time.time() - t0) / 60, 1), "fields_under_128_pixels": int((npx < N_PIX).sum())})


def main(workers: int, smoke: bool, only: str | None) -> None:
    with RunLog("01_extract_pf", {"workers": workers, "smoke": smoke, "only": only}, paper=PAPER) as log:
        for tile, year in BENCHMARK:
            if only and only != tile:
                continue
            process(tile, year, workers, smoke, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None)
    main(**vars(ap.parse_args()))
