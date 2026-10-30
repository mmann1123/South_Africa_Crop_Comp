#!/usr/bin/env python
"""Per-pixel valid-observation counts and 6-day-grid interpolation measures per site, under three
masks (the paper's full recipe, SCL only, and no mask = pixel covered), with WorldCover strata
and bootstrap intervals over points (seed 42)."""
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
from experiments.external_audits.maleki import catalogue as C, interplib as il  # noqa: E402

RES = paths.results_dir(C.PAPER)
ROOT = paths.paper_root(C.PAPER)
MASKS = {"paper_masks": 1, "scl_only": 2, "no_mask": 0}  # column offset in the (c, v, s) triplets
N_BOOT = 1000


def load(site: str) -> dict:
    z = np.load(ROOT / f"{site}_validity.npz")
    arr = z["arr"]
    ok = arr[:, -1] >= 0
    return {"arr": arr[ok][:, :-1].astype(bool), "wc": arr[ok][:, -1].astype(int), "doy": z["doy"], "xy": z["xy"][ok], "n_dropped": int((~ok).sum())}


def boot_ci(x: np.ndarray, fn, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    s = np.array([fn(x[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def stratum_rows(site: str, mask: str, stratum: str, m: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    n = len(m["n_valid"])
    row = {"site": site, "mask": mask, "stratum": stratum, "n_points": n}
    for key in ("n_valid", "share_not_observed", "longest_unobserved_steps", "longest_gap_days"):
        s = il.summarise(m[key].astype(float))
        row.update({f"{key}_{k}": round(v, 3) for k, v in s.items()})
    lo, hi = boot_ci(m["n_valid"].astype(float), np.median, rng)
    row.update({"n_valid_median_lo": lo, "n_valid_median_hi": hi})
    lo, hi = boot_ci(m["share_not_observed"], np.mean, rng)
    row.update({"share_not_observed_mean_lo": round(lo, 4), "share_not_observed_mean_hi": round(hi, 4)})
    row["interp_steps_mean"] = round(float(m["interp_steps"].mean()), 2)
    row["extrap_steps_mean"] = round(float(m["extrap_steps"].mean()), 2)
    row["obs_steps_mean"] = round(float(m["obs_steps"].mean()), 2)
    row["longest_unobserved_days_median"] = float(np.median(m["longest_unobserved_steps"])) * C.GRID_STEP
    row["pct_pixels_gt_20_unobserved_steps"] = round(100 * float((60 - m["obs_steps"] > 20).mean()), 1)
    row["pct_pixels_longest_gap_gt_60d"] = round(100 * float((m["longest_gap_days"] > 60).mean()), 1)
    return row


def main(force: bool = False) -> None:
    out = RES / "interp_summary.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    grid = il.grid_days(C.GRID_STEP, C.GRID_N)
    rng = np.random.default_rng(42)
    with RunLog("03_interp_metrics", {"force": force, "n_boot": N_BOOT}, paper=C.PAPER) as log:
        srows, mrows, prows = [], [], []
        for site in C.SITES:
            d = load(site)
            log.count(f"{site}_points_used", len(d["wc"]))
            log.count(f"{site}_points_without_data", d["n_dropped"])
            for mask, off in MASKS.items():
                v = d["arr"][:, off::3]
                m = il.pixel_metrics(v, d["doy"], grid, C.HALF_WINDOW)
                strata = {"all": np.ones(len(v), bool)} | {name: d["wc"] == code for code, name in C.WC_CLASSES.items()}
                for name, sel in strata.items():
                    if sel.sum() >= 30:
                        srows.append(stratum_rows(site, mask, name, {k: x[sel] for k, x in m.items()}, rng))
                if mask == "paper_masks":
                    mc = il.monthly_counts(v, d["doy"], C.YEAR)
                    cc = il.monthly_counts(d["arr"][:, 0::3], d["doy"], C.YEAR)
                    for mo in range(12):
                        mrows.append({"site": site, "month": mo + 1, "valid_mean": round(float(mc[:, mo].mean()), 3), "valid_median": float(np.median(mc[:, mo])), "pct_pixels_zero_valid": round(100 * float((mc[:, mo] == 0).mean()), 1), "covered_dates_mean": round(float(cc[:, mo].mean()), 3), "valid_share_of_covered": round(float(mc[:, mo].sum() / max(cc[:, mo].sum(), 1)), 3)})
                    prows.append(pd.DataFrame({"site": site, "lon": d["xy"][:, 0], "lat": d["xy"][:, 1], "worldcover": d["wc"], **{k: x for k, x in m.items()}}))
            print(site, "done", flush=True)
        write_csv(pd.DataFrame(srows), out, "03_interp_metrics", [str(ROOT / f"{s}_validity.npz") for s in C.SITES])
        write_csv(pd.DataFrame(mrows), RES / "monthly_valid.csv", "03_interp_metrics", [str(ROOT / f"{s}_validity.npz") for s in C.SITES])
        write_csv(pd.concat(prows), RES / "pixel_metrics.csv", "03_interp_metrics", [str(ROOT / f"{s}_validity.npz") for s in C.SITES])
        log.count("summary_rows", len(srows))
        log.add("grid_days", [int(grid[0]), int(grid[-1])])
        log.add("half_window", C.HALF_WINDOW)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
