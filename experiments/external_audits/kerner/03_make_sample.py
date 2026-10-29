#!/usr/bin/env python
"""Stratified random pixel sample (seed 42) on CDL corn / soy / other per year,
with each pixel's tile grid position, UTM coordinates and the three released
growth-stage DOYs (greenup, peak, senescence)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.kerner import kernerlib  # noqa: E402

PAPER = "kerner"
RES = paths.results_dir(PAPER)
OUT = RES / "kerner_sample.csv"
MED_OUT = RES / "kerner_phenology_medians.csv"
N_PER_CLASS = 1000
SEED = 42
YEARS = (2017, 2018, 2019)


def main(force: bool = False, n_per_class: int = N_PER_CLASS) -> None:
    if OUT.exists() and not force:
        print("sample already present; nothing to do (use --force to redo)")
        return
    root = paths.paper_root(PAPER) / "data"
    parts, meds = [], []
    with RunLog("03_make_sample", {"force": force, "n_per_class": n_per_class}, paper=PAPER) as log:
        for year in YEARS:
            labels = np.load(root / f"cdl_labels_16TBL_{year}.npy", allow_pickle=False)
            pheno = np.load(root / "phenology" / f"16TBL_{year}_phenology.npy", allow_pickle=False)
            valid = (pheno >= 1).all(axis=2)  # DOY 0 marks no phenology
            log.add(f"valid_share_{year}", float(valid.mean()))
            for c, name in kernerlib.CLASS_NAMES.items():
                m = (labels == c) & valid
                meds.append({"year": year, "class": name, "n_pixels_valid": int(m.sum()), **{f"median_{s}": float(np.median(pheno[..., k][m])) for k, s in enumerate(kernerlib.GROWTH_STAGES)}})
            r, c, cls = kernerlib.stratified_sample(labels, valid, n_per_class, seed=SEED)
            x, y = kernerlib.pixel_to_xy(r, c)
            df = pd.DataFrame({"year": year, "pid": [f"{year}_{i}" for i in range(len(r))], "class": [kernerlib.CLASS_NAMES[v] for v in cls], "row": r, "col": c, "x": x, "y": y})
            for k, s in enumerate(kernerlib.GROWTH_STAGES):
                df[s] = pheno[r, c, k].astype(int)
            parts.append(df)
            log.count(f"sample_{year}", int(len(df)))
        sample = pd.concat(parts, ignore_index=True)
        write_csv(sample, OUT, "03_make_sample", [str(root)])
        write_csv(pd.DataFrame(meds), MED_OUT, "03_make_sample", [str(root)])
        print(sample.groupby(["year", "class"]).size().unstack())
        print(pd.DataFrame(meds).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    main(**vars(ap.parse_args()))
