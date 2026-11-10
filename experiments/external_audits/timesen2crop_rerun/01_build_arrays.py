#!/usr/bin/env python
"""Read TimeSen2Crop.zip (Zenodo 4715631, 1,212,513 entries) into one compressed
array file per tile. Reads the zip directly, never extracts it.

Output <root>/arrays/<tile>.npz: X (N, T, 9) int16 reflectance x 1e4 in file
order B1..B9, flag (N, T) int8 (0 clear, 1 cloud, 2 shadow, 3 snow), y (N) int8
class folder 0..15, sid (N) int32 sample file number, dates (T) yyyymmdd.
Samples whose row count differs from the tile's dates.csv are counted and
dropped (results/array_manifest.csv)."""
from __future__ import annotations

import io
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402

PAPER = "timesen2crop_rerun"
ZIP = paths.paper_root(PAPER) / "TimeSen2Crop.zip"


def one_tile(args: tuple[str, list[str]]) -> dict:
    tile, names = args
    z = zipfile.ZipFile(ZIP)
    dates = np.loadtxt(io.BytesIO(z.read(f"TimeSen2Crop/{tile}/dates.csv")), skiprows=1, dtype=np.int32, ndmin=1)
    T = len(dates)
    X, F, Y, S, bad = [], [], [], [], 0
    for n in names:
        parts = n.split("/")
        a = np.loadtxt(io.BytesIO(z.read(n)), delimiter=",", skiprows=1, dtype=np.int32, ndmin=2)
        if a.shape != (T, 10):
            bad += 1
            continue
        X.append(a[:, :9].astype(np.int16))
        F.append(a[:, 9].astype(np.int8))
        Y.append(int(parts[2]))
        S.append(int(parts[3][:-4]))
    out = paths.paper_root(PAPER) / "arrays" / f"{tile}.npz"
    np.savez(out, X=np.stack(X), flag=np.stack(F), y=np.array(Y, np.int8), sid=np.array(S, np.int32), dates=dates)
    return {"tile": tile, "samples": len(Y), "wrong_shape": bad, "acquisitions": T, "first": int(dates[0]), "last": int(dates[-1])}


def main() -> None:
    with RunLog("01_build_arrays", {}, paper=PAPER) as log:
        (paths.paper_root(PAPER) / "arrays").mkdir(exist_ok=True)
        by = defaultdict(list)
        for n in zipfile.ZipFile(ZIP).namelist():
            p = n.split("/")
            if len(p) == 4 and p[3].endswith(".csv") and p[2].isdigit():
                by[p[1]].append(n)
        with ProcessPoolExecutor(16) as ex:
            rows = list(ex.map(one_tile, sorted(by.items())))
        df = pd.DataFrame(rows)
        write_csv(df, paths.results_dir(PAPER) / "array_manifest.csv", "01_build_arrays", ["TimeSen2Crop.zip md5 b5b7aad3fef192e78252e11c9a0e5cb8"])
        print(df.to_string())
        log.count("samples", int(df["samples"].sum()))


if __name__ == "__main__":
    main()
