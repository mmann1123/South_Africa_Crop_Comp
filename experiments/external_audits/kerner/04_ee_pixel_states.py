#!/usr/bin/env python
"""Per sampled pixel and year: the HLS v2.0 state (0 no data, 1 clear, 2 flagged by
Fmask bits 0-3) at every L30 and S30 acquisition of tile 16TBL, read in Earth
Engine at the pixel centre on the tile's own 30 m grid. Chunked, resumable by
pixel id; one row per pixel with a digit string ordered like
results/kerner_hls_acquisitions.csv."""
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
from experiments.external_audits.kerner import kernerlib  # noqa: E402

PAPER = "kerner"
RES = paths.results_dir(PAPER)
OUT = RES / "kerner_pixel_states.csv"
SAMPLE = RES / "kerner_sample.csv"
ACQ = RES / "kerner_hls_acquisitions.csv"
COLLECTIONS = {"S": "NASA/HLS/HLSS30/v002", "L": "NASA/HLS/HLSL30/v002"}
CHUNK = 200
WORKERS = 3
TILE_SCALE = 4


def state_image(acq: pd.DataFrame):
    """One byte band per acquisition, named by the digit position."""
    import ee

    bands = []
    for i, r in enumerate(acq.itertuples()):
        f = ee.Image(f"{COLLECTIONS[r.sensor]}/{r.system_index}").select("Fmask")
        bands.append(f.bitwiseAnd(kernerlib.FMASK_BITS).neq(0).add(1).unmask(0).toByte().rename(f"a{i:03d}"))
    return ee.Image.cat(bands)


def read_chunk(img, chunk: pd.DataFrame, n_acq: int) -> pd.DataFrame:
    import ee

    feats = [ee.Feature(ee.Geometry.Point([float(r.x), float(r.y)], kernerlib.TILE_CRS), {"pid": r.pid}) for r in chunk.itertuples()]
    fc = ee.FeatureCollection(feats)
    proj = kernerlib.TILE_CRS
    transform = [kernerlib.TILE_RES, 0, kernerlib.TILE_ULX, 0, -kernerlib.TILE_RES, kernerlib.TILE_ULY]
    red = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), crs=proj, crsTransform=transform, tileScale=TILE_SCALE)
    for attempt in range(4):
        try:
            out = red.getInfo()["features"]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"chunk failed after retries: {type(e).__name__}: {str(e)[:150]}", flush=True)
                return pd.DataFrame()
            time.sleep(10 * 2**attempt)
    rows = []
    for f in out:
        p = f["properties"]
        digits = "".join(str(int(p.get(f"a{i:03d}") or 0)) for i in range(n_acq))
        rows.append({"pid": p["pid"], "states": digits})
    return pd.DataFrame(rows)


def main(smoke: bool = False, force: bool = False, years: list[int] | None = None) -> None:
    sample = pd.read_csv(SAMPLE)
    acq_all = pd.read_csv(ACQ)
    if force and OUT.exists() and not smoke:
        OUT.unlink()
    done = set(pd.read_csv(OUT)["pid"]) if OUT.exists() and not smoke else set()
    todo_years = [y for y in (years or sorted(sample["year"].unique())) if (~sample[sample["year"] == y]["pid"].isin(done)).any()]
    if not todo_years:
        print("all pixels already in results; nothing to do (use --force to redo)")
        return
    with RunLog("04_ee_pixel_states", {"smoke": smoke, "force": force, "years": years}, paper=PAPER) as log:
        log.add("ee_project", init())
        log.add("collections", COLLECTIONS)
        log.add("design", {"chunk": CHUNK, "workers": WORKERS, "tile_scale": TILE_SCALE, "fmask_bits_mask": kernerlib.FMASK_BITS, "grid": {"crs": kernerlib.TILE_CRS, "ulx": kernerlib.TILE_ULX, "uly": kernerlib.TILE_ULY, "res": kernerlib.TILE_RES}})
        n_new = 0
        for year in todo_years:
            acq = acq_all[acq_all["year"] == year].reset_index(drop=True)
            img = state_image(acq)
            s = sample[(sample["year"] == year) & ~sample["pid"].isin(done)]
            if smoke:
                s = s.groupby("class").head(7)
            chunks = [s.iloc[i : i + CHUNK] for i in range(0, len(s), CHUNK)]
            t = time.time()
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for df in ex.map(lambda c: read_chunk(img, c, len(acq)), chunks):
                    if not len(df):
                        continue
                    if smoke:
                        print(df.head(3).to_string())
                        print(pd.Series([sum(ch != "0" for ch in v) for v in df["states"]]).describe())
                    else:
                        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                    n_new += len(df)
            log.add(f"year_{year}", {"n_acq": int(len(acq)), "pixels_requested": int(len(s)), "seconds": round(time.time() - t)})
            print(f"{year}: {len(s)} pixels, {len(acq)} acquisitions, {time.time() - t:.0f} s", flush=True)
        log.count("pixels_written", n_new)
        if smoke:
            log.discard()
        elif OUT.exists():
            log.count("rows_total", int(len(pd.read_csv(OUT))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    main(**vars(ap.parse_args()))
