#!/usr/bin/env python
"""Step 1, the hard gate: reproduce the paper's source-only evaluation with the
released models, using the authors' own loaders and evaluation function
(create_evaluation_loaders + evaluation.evaluation, test split, all dates, all
pixels), for DK1 to FR1 and FR1 to DK1, three folds. Compares with the metrics
stored in the Zenodo archive and with Table 1 of the paper.
Run in env deep_field:  python -I 03_reproduce.py"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch_rerun import evalcore, rerunlib  # noqa: E402

PAPER = "timematch_rerun"
RES = HERE / "results"
PAPER_TABLE1 = {"DK1_to_FR1": (28.3, 1.9), "FR1_to_DK1": (24.9, 2.0)}  # macro F1 %, std over folds


def main() -> None:
    with RunLog("03_reproduce", {}, paper=PAPER) as log:
        import torch

        evalcore.setup_repo()
        from dataset import PixelSetData, create_evaluation_loaders
        from evaluation import evaluation

        log.add("torch", torch.__version__)
        log.add("authors_commit", "a652df95282de9a3fc12d2fd204f438ff4ccb122")
        rows = []
        for dname, (src, tgt, mdir) in evalcore.DIRECTIONS.items():
            classes = evalcore.source_classes(src)
            log.add(f"classes_{dname}", classes)
            s_name, t_name = evalcore.SRC_NAMES[src], evalcore.SRC_NAMES[tgt]
            n_s = len(PixelSetData(str(evalcore.DATA_ROOT), s_name, classes))
            n_t = len(PixelSetData(str(evalcore.DATA_ROOT), t_name, classes))
            folds = rerunlib.make_folds(n_s, n_t, s_name, t_name, seed=1)
            cfg = SimpleNamespace(model="pseltae", seq_length=30, data_root=str(evalcore.DATA_ROOT), classes=classes,
                                  num_workers=8, batch_size=128, num_pixels=64)
            for fold in range(3):
                _, test_loader = create_evaluation_loaders(t_name, folds[fold], cfg, sample_pixels_val=True)
                model = evalcore.load_model(evalcore.MODELS / mdir, fold, len(classes))
                m = evaluation(model, test_loader, torch.device("cuda"), classes, mode="test")
                stored = json.loads((evalcore.MODELS / mdir / f"fold_{fold}" / f"test_metrics_{t_name.replace('/', '_')}.json").read_text())
                rows.append({"direction": dname, "fold": fold, "n_test": len(test_loader.dataset),
                             **{f"rerun_{k}": m[k] for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")},
                             **{f"stored_{k}": stored[k] for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")}})
                (RES / f"class_report_{dname}_fold{fold}.txt").write_text(m["classification_report"])
                print(rows[-1], flush=True)
        df = pd.DataFrame(rows)
        df["d_macro_f1"] = df["rerun_macro_f1"] - df["stored_macro_f1"]
        df["d_accuracy"] = df["rerun_accuracy"] - df["stored_accuracy"]
        write_csv(df, RES / "reproduction_per_fold.csv", "03_reproduce", ["released models fold_*/model.pt", "test_metrics_*.json"])
        summ = []
        for dname, g in df.groupby("direction"):
            pm, ps = PAPER_TABLE1[dname]
            summ.append({"direction": dname,
                         "rerun_macro_f1_mean": 100 * g["rerun_macro_f1"].mean(), "rerun_macro_f1_std": 100 * g["rerun_macro_f1"].std(ddof=0),
                         "stored_macro_f1_mean": 100 * g["stored_macro_f1"].mean(), "paper_table1_macro_f1": pm, "paper_table1_std": ps,
                         "rerun_oa_mean": 100 * g["rerun_accuracy"].mean(), "stored_oa_mean": 100 * g["stored_accuracy"].mean(),
                         "max_abs_fold_diff_macro_f1_pts": 100 * g["d_macro_f1"].abs().max()})
        write_csv(pd.DataFrame(summ), RES / "reproduction_summary.csv", "03_reproduce", ["reproduction_per_fold.csv"])
        print(pd.DataFrame(summ).T)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
