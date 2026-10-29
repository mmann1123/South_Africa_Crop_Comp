#!/usr/bin/env python
"""Apply the loader's monthly-median / linear-interpolation / extrapolation fill
rule (re-implemented in fillrule.py) to the sampled patch time stacks and
summarise, per scenario and split, the share of filled, interpolated and
extrapolated patch-months in the PAD window (Apr-Sep) and the OAD window as
coded (May-Nov), plus per-month missing shares and observations per patch-year."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import fillstats  # noqa: E402

PAPER = "sen4agrinet"


def main(force: bool = False) -> None:
    res = paths.results_dir(PAPER)
    with RunLog("05_fill_rates", {"force": force}, paper=PAPER) as log:
        if (res / "fill_by_split.csv").exists() and not force:
            print("fill tables exist; use --force to redo")
            log.discard()
            return
        sample = pd.read_csv(res / "patch_sample.csv")
        stacks = pd.read_csv(res / "patch_time_stacks.csv.gz")
        comp = pd.read_csv(res / "split_composition.csv")
        comp = comp[comp["scenario"].isin([2, 3]) & comp["split"].isin(["train", "test"])]
        pf = fillstats.patch_fill_table(sample, stacks)
        log.count("sampled_patch_rows", len(pf))
        log.count("sampled_unique_patches", int(pf["stem"].nunique()))
        log.count("stacks_with_no_acquisition_in_year", int((pf["n_obs_in_year"] == 0).sum()))
        inputs = ["patch_sample.csv", "patch_time_stacks.csv.gz", "split_composition.csv"]
        write_csv(pf, res / "patch_fill.csv", "05_fill_rates", inputs)
        write_csv(fillstats.split_summary(pf, comp), res / "fill_by_split.csv", "05_fill_rates", inputs)
        write_csv(fillstats.month_missing(pf, comp, ["scenario", "split", "country", "year"]), res / "month_missing_by_split.csv", "05_fill_rates", inputs)
        write_csv(fillstats.month_missing(pf, comp, ["country", "year", "tile"]), res / "month_missing_by_tile_year.csv", "05_fill_rates", inputs)
        write_csv(fillstats.obs_by_country_year(pf), res / "obs_by_country_year.csv", "05_fill_rates", inputs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
