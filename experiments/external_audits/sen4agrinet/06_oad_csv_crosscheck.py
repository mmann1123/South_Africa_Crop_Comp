#!/usr/bin/env python
"""Cross-check the fill rates against the published OAD test CSVs of Scenarios 2
and 3. Each CSV row is a parcel with the mean of every band in each of the 12
monthly bins (the fill rule has already been applied when the CSV was built). A
month is flagged as filled when all 13 band means lie on the time-weighted line
through other months within 1e-6 and the pattern is consistent (see
fillrule.infer_filled). Scenario 2 test is France 2019, Scenario 3 test is
Catalonia 2020, so the year is known. Rows are parcels, not patches, so the
shares are parcel-weighted (and pixel-weighted using the `counts` column)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import fillrule  # noqa: E402

PAPER = "sen4agrinet"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
CASES = {2: ("France", 2019), 3: ("Catalonia", 2020)}  # scenario -> test domain
WINDOWS = {"PAD_Apr-Sep": fillrule.PAD_MONTHS, "OAD_May-Nov": fillrule.OAD_MONTHS}


def load_means(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    cols = [f"{b}_{m:02d}_mean" for b in BANDS for m in range(12)]
    df = pd.read_csv(path, usecols=["parcel_id", "label", "counts"] + cols)
    arr = df[cols].to_numpy(float).reshape(len(df), len(BANDS), 12).transpose(0, 2, 1)  # N x 12 x B
    return df[["parcel_id", "label", "counts"]], arr


def main(force: bool = False) -> None:
    res = paths.results_dir(PAPER)
    out = res / "oad_csv_crosscheck.csv"
    with RunLog("06_oad_csv_crosscheck", {"force": force}, paper=PAPER) as log:
        if out.exists() and not force:
            print(f"{out.name} exists; use --force to redo")
            log.discard()
            return
        rows, mrows = [], []
        for scen, (country, year) in CASES.items():
            f = paths.paper_root(PAPER) / "dropbox" / "oad" / f"exp{scen}_patches5000_strat_coco_test.csv.gz"
            t0 = time.time()
            meta, arr = load_means(f)
            filled, extrap = fillrule.infer_filled(arr, year)
            log.count(f"scenario{scen}_rows", len(meta))
            log.add(f"scenario{scen}_seconds", round(time.time() - t0, 1))
            for subset, mask in (("all rows", np.ones(len(meta), bool)), ("label > 0", (meta["label"] > 0).to_numpy())):
                for weighting, w in (("parcels", np.ones(len(meta))), ("pixels", meta["counts"].to_numpy(float))):
                    ww = w[mask]
                    for name, months in WINDOWS.items():
                        sel = [m - 1 for m in months]
                        f_share = (filled[mask][:, sel].mean(axis=1) * ww).sum() / ww.sum()
                        e_share = (extrap[mask][:, sel].mean(axis=1) * ww).sum() / ww.sum()
                        any_share = ((filled[mask][:, sel].any(axis=1)) * ww).sum() / ww.sum()
                        rows.append({"scenario": scen, "country": country, "year": year, "subset": subset, "weighting": weighting, "window": name, "n_rows": int(mask.sum()), "share_filled": f_share, "share_extrapolated": e_share, "share_rows_any_fill": any_share})
                    if subset == "all rows":
                        for m in range(12):
                            mrows.append({"scenario": scen, "country": country, "year": year, "weighting": weighting, "month": m + 1, "share_filled": float((filled[:, m] * w).sum() / w.sum())})
        write_csv(pd.DataFrame(rows), out, "06_oad_csv_crosscheck", [f"exp{s}_patches5000_strat_coco_test.csv.gz" for s in CASES])
        write_csv(pd.DataFrame(mrows), res / "oad_csv_month_filled.csv", "06_oad_csv_crosscheck", [f"exp{s}_patches5000_strat_coco_test.csv.gz" for s in CASES])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
