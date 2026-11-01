#!/usr/bin/env python
"""Per parcel and date summary features (median and mean of the 10 bands over the
parcel's pixels, reflectance / 65535) for the FR1 parcels in the test union and for
all DK1 parcels. These feed the Cloud Score+ emulator used for DK1, whose parcel
polygons are not shipped. Run in env deep_field: python -I 04_extract_features.py"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROOT = Path("/mnt/bigdrive/external_audits/timematch_rerun")
DATA = ROOT / "timematch_data" / "timematch_data"
OUT = ROOT / "features"


def feat(args):
    import zarr

    tile_name, i = args
    a = np.asarray(zarr.load(str(DATA / tile_name / "data" / f"{i}.zarr"))).astype(np.float32) / 65535.0  # (T, C, S)
    return np.concatenate([np.median(a, axis=2), a.mean(axis=2)], axis=1).astype(np.float32)


def run(label, tile_name, idxs):
    f = OUT / f"features_{label}.npz"
    if f.exists():
        print("exists", f)
        return
    t0 = time.time()
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(feat, [(tile_name, int(i)) for i in idxs], chunksize=64))
    np.savez_compressed(f, idx=np.asarray(idxs), feats=np.stack(res))
    print(label, len(idxs), f"{time.time() - t0:.0f}s", flush=True)


def main():
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.external_audits.timematch_rerun import rerunlib

    OUT.mkdir(exist_ok=True)
    src, tgt = "denmark/32VNH/2017", "france/30TXT/2017"
    folds = rerunlib.make_folds(60332, 115755, src, tgt, seed=1)
    union = sorted(set().union(*[f[tgt]["test"] for f in folds]))
    run("FR1_union", tgt, union)
    run("DK1_all", src, list(range(60332)))


if __name__ == "__main__":
    main()
