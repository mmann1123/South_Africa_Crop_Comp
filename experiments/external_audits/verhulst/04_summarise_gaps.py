#!/usr/bin/env python
"""Per-year, per-variant gap structure of the forest-pixel dekadal series: valid dekads, shares of
interpolated and edge-filled dekads, longest gaps, monthly and dekad-level valid shares."""
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
from experiments.external_audits.verhulst import catalogue, verhulstlib as vl  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
N_BOOT = 200


def load_year(year: int) -> pd.DataFrame:
    files = sorted((ROOT / "chunks").glob(f"valid_{year}_*.csv.gz"))
    if not files:
        raise RuntimeError(f"no chunks for {year}; run 03_valid_dekads.py")
    return pd.concat([pd.read_csv(f) for f in files]).sort_values("pid").reset_index(drop=True)


def matrices(df: pd.DataFrame, variant: str) -> np.ndarray:
    return np.column_stack([df[f"d{k:02d}_{variant}"].to_numpy() > 0 for k in range(catalogue.N_DEKADS)])


def boot_se(values: np.ndarray, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    return float(values[idx].mean(1).std())


def main(force: bool = False) -> None:
    out = RES / "valid_dekads_summary.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    rng = np.random.default_rng(catalogue.SEED)
    summ, dek = [], []
    with RunLog("04_summarise_gaps", {"force": force}, paper=PAPER) as log:
        for year in catalogue.YEARS:
            df = load_year(year)
            log.count(f"pixels_{year}", len(df))
            obs = np.column_stack([df[f"d{k:02d}_obs"].to_numpy() for k in range(catalogue.N_DEKADS)])
            for variant in catalogue.VARIANTS:
                v = matrices(df, variant)
                row = {"year": year, "variant": variant, **vl.year_summary(v)}
                row["valid_dekads_se"] = boot_se(v.sum(1).astype(float), rng)
                row["acquisitions_per_pixel"] = float(obs.sum(1).mean())
                row["dekads_without_acquisition_mean"] = float((obs == 0).sum(1).mean())
                summ.append(row)
                mshare = vl.monthly_valid_share(v)
                for k in range(catalogue.N_DEKADS):
                    dek.append({"year": year, "variant": variant, "dekad": k + 1, "month": k // 3 + 1, "valid_share": float(v[:, k].mean()), "acquired_share": float((obs[:, k] > 0).mean()), "month_valid_share": float(mshare[k // 3])})
        s = pd.DataFrame(summ)
        write_csv(s, out, "04_summarise_gaps", [str(ROOT / "chunks")])
        write_csv(pd.DataFrame(dek), RES / "dekad_valid_share.csv", "04_summarise_gaps", [str(ROOT / "chunks")])
        log.count("summary_rows", len(s))
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(s[s["variant"] == catalogue.PRIMARY].round(3).T.to_string())
            print(s.pivot(index="year", columns="variant", values="valid_dekads_mean").round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
