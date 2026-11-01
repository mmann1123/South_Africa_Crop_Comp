#!/usr/bin/env python
"""Summaries from results/conditions_*.csv: per-fold table, condition summary (mean and SD
over folds), cloud effects against the random-removal control, and the effect as a share
of the residual gap to the target-trained bound (archive upperbound_* results). Also the
TimeMatch residual from the archive for context. Run in env ee or deep_field."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch_rerun import rsummary  # noqa: E402

PAPER = "timematch_rerun"
RES = HERE / "results"
OUT = Path("/mnt/bigdrive/external_audits/timematch_rerun/models/outputs")
ARCH = {"DK1_to_FR1": ("upperbound_32VNH_to_30TXT", "timematch_32VNH_to_30TXT", "france_30TXT_2017"),
        "FR1_to_DK1": ("upperbound_30TXT_to_32VNH", "timematch_30TXT_to_32VNH", "denmark_32VNH_2017")}


def archive_scores():
    ub, tm = {}, {}
    for d, (u, t, tgt) in ARCH.items():
        for f, v in enumerate(json.loads((OUT / u / f"overall_{tgt}.json").read_text())["macro_f1"]):
            ub[(d, f)] = v
        for f, v in enumerate(json.loads((OUT / t / f"overall_{tgt}.json").read_text())["macro_f1"]):
            tm[(d, f)] = v
    return ub, tm


def main() -> None:
    with RunLog("07_summarise", {}, paper=PAPER) as log:
        frames = [pd.read_csv(p) for p in sorted(RES.glob("conditions_*.csv"))]
        df = pd.concat(frames, ignore_index=True)
        ub, tm = archive_scores()
        log.add("upper_bound_macro_f1_by_fold", {f"{k[0]}_{k[1]}": v for k, v in ub.items()})
        log.add("timematch_macro_f1_by_fold", {f"{k[0]}_{k[1]}": v for k, v in tm.items()})
        ft = rsummary.fold_table(df)
        write_csv(ft, RES / "conditions_by_fold.csv", "07_summarise", [str(p.name) for p in sorted(RES.glob("conditions_*.csv"))])
        cs = rsummary.condition_summary(ft)
        write_csv(cs, RES / "conditions_summary.csv", "07_summarise", ["conditions_by_fold.csv"])
        ef = rsummary.effects(ft, ub, tm)
        write_csv(ef, RES / "cloud_effect_by_fold.csv", "07_summarise", ["conditions_by_fold.csv", "archive upperbound_*/timematch_* overall json"])
        es = rsummary.effects_summary(ef)
        write_csv(es, RES / "cloud_effect_summary.csv", "07_summarise", ["cloud_effect_by_fold.csv"])
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print(cs[["direction", "cloud_source", "threshold", "condition", "macro_f1_mean", "macro_f1_fold_sd", "accuracy_mean", "mean_dates_kept"]].round(3).to_string())
        print(es[["direction", "cloud_source", "threshold", "effect_b_mean", "effect_d_mean", "share_of_shifted_gap_mean", "share_of_timematch_gap_mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
