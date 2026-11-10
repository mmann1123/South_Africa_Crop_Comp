#!/usr/bin/env python
"""Summarise the DENETHOR re-run.

PF arm (02_train_eval.py): per condition, mean and SD over seeds of validation
and test accuracy, macro F1 and kappa; the paired per-seed contrasts A1 - A2
(does knowing which days are synthesised help?) and A1 - A0, with t intervals
over seeds; and, descriptively, A0 test accuracy (fraction of seeds correct per
field) by tercile of the field's share of mostly-synthetic pixel-days in 2019,
in May and over the year.
RF arm (04_rf.py): contrasts over draws (Welch t) of month-matched against
random thinning, at training time on the 2019 test tile and at test time on
the 2018-imagery test tile, and of training thinning against the forest-seed
spread (rf_contrasts.csv). The 2018-imagery accuracies are not differenced
against 2019 because only 2019 labels exist for that tile.
Figure: figures/denethor_rerun.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402

PAPER = "denethor_rerun"
RES = paths.results_dir(PAPER)
PAPER_TEST, PAPER_VAL = 0.6725, 0.8802


def t_ci(d: np.ndarray) -> tuple[float, float, float]:
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return m, m - h, m + h


def main() -> None:
    with RunLog("05_summarise", {}, paper=PAPER) as log:
        runs = pd.read_csv(RES / "train_eval_summary.csv")
        cols = ["val_accuracy", "test_accuracy", "test10_accuracy", "test_f1_macro", "test_kappa"]
        cond = runs.groupby("cond")[cols].agg(["mean", "std", "count"])
        cond.columns = ["_".join(c) for c in cond.columns]
        cond = cond.reset_index()
        cond["paper_test_accuracy"], cond["paper_val_accuracy"] = PAPER_TEST, PAPER_VAL
        write_csv(cond, RES / "pf_condition_summary.csv", "05_summarise", ["results/train_eval_summary.csv"])
        w = runs.pivot(index="seed", columns="cond", values=["test_accuracy", "test_f1_macro", "test10_accuracy"])
        rows = []
        for metric in ("test_accuracy", "test_f1_macro", "test10_accuracy"):
            for a, b in (("A1", "A2"), ("A1", "A0"), ("A2", "A0")):
                d = (w[metric][a] - w[metric][b]).dropna()
                m, lo, hi = t_ci(d)
                rows.append({"metric": metric, "contrast": f"{a} - {b}", "seeds": len(d), "mean": m, "ci_low": lo, "ci_high": hi,
                             "wilcoxon_p": stats.wilcoxon(d).pvalue if len(d) >= 6 else np.nan})
        contrasts = pd.DataFrame(rows)
        write_csv(contrasts, RES / "pf_contrasts.csv", "05_summarise", ["results/train_eval_summary.csv"])
        z = np.load(paths.paper_root(PAPER) / "pf_17E-243N_2019.npz")
        qa, fid = z["qa"], z["fid"]
        syn = (qa > 50)
        field = pd.DataFrame({"fid": fid, "syn_may": syn[120:151].mean(axis=(0, 2)), "syn_year": syn.mean(axis=(0, 2)),
                              "syn_apr_oct": syn[90:304].mean(axis=(0, 2))})
        preds = []
        for p in sorted((RES / "runs").glob("A0_seed*/test_predictions.csv")):
            d = pd.read_csv(p)
            preds.append(d.assign(correct=(d["true"] == d["pred_single"]).astype(float))[["fid", "correct"]])
        acc = pd.concat(preds).groupby("fid")["correct"].mean().rename("a0_share_correct").reset_index()
        field = field.merge(acc, on="fid")
        strat = []
        for var in ("syn_may", "syn_apr_oct", "syn_year"):
            field["tercile"] = pd.qcut(field[var], 3, labels=["low", "mid", "high"], duplicates="drop")
            for t, g in field.groupby("tercile", observed=True):
                strat.append({"variable": var, "tercile": t, "fields": len(g), "range_low": g[var].min(), "range_high": g[var].max(),
                              "a0_accuracy": g["a0_share_correct"].mean()})
            r = stats.spearmanr(field[var], field["a0_share_correct"])
            strat.append({"variable": var, "tercile": "spearman_rho", "fields": len(field), "a0_accuracy": r.statistic, "range_low": r.pvalue})
        strat = pd.DataFrame(strat)
        write_csv(strat, RES / "pf_synthetic_strata.csv", "05_summarise", ["pf_17E-243N_2019.npz", "results/runs/A0_seed*/test_predictions.csv"])
        rfc = RES / "rf_conditions.csv"
        if rfc.exists():
            rf = pd.read_csv(rfc)
            rf = rf[rf["rule"] == "literal_cld_lt_40"]
            rf_t = []
            for test, a, b in (("test_tile_2019", "B1train", "B1trainctrl"), ("test_tile_2019", "B1train", "B0seeds"),
                               ("test_tile_2019", "B1trainctrl", "B0seeds"), ("test_tile_2018", "B1", "B1ctrl")):
                for metric in ("accuracy", "f1_macro"):
                    xa = rf[(rf["condition"] == a) & (rf["test"] == test)][metric].values
                    xb = rf[(rf["condition"] == b) & (rf["test"] == test)][metric].values
                    t = stats.ttest_ind(xa, xb, equal_var=False)
                    rf_t.append({"test": test, "contrast": f"{a} - {b}", "metric": metric, "mean_a": xa.mean(), "mean_b": xb.mean(),
                                 "diff": xa.mean() - xb.mean(), "welch_p": t.pvalue, "draws": len(xa)})
            rf_t = pd.DataFrame(rf_t)
            write_csv(rf_t, RES / "rf_contrasts.csv", "05_summarise", ["results/rf_conditions.csv"])
            print(rf_t.round(4).to_string())
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for i, c in enumerate(["A0", "A1", "A2"]):
            v = runs.loc[runs["cond"] == c, "test_accuracy"]
            ax[0].scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(v)), v, s=18, alpha=0.7)
            ax[0].plot([i - 0.2, i + 0.2], [v.mean()] * 2, color="black")
        ax[0].axhline(PAPER_TEST, ls="--", color="grey", lw=1)
        ax[0].set_xticks(range(3), ["A0 published", "A1 + PF-QA", "A2 + permuted QA"])
        ax[0].set_ylabel("Test accuracy, 17E-243N 2019")
        ax[0].set_title("(a) PseLTae on Planet Fusion, per seed")
        s = strat[(strat["variable"] == "syn_may") & (strat["tercile"] != "spearman_rho")]
        ax[1].bar(s["tercile"].astype(str), s["a0_accuracy"], color="#4a7ab5")
        ax[1].set_ylabel("A0 accuracy (share of seeds correct)")
        ax[1].set_xlabel("Tercile of field's May 2019 synthetic share")
        ax[1].set_title("(b) Accuracy by synthetic share")
        fig.tight_layout()
        out = paths.figures_dir(PAPER) / "denethor_rerun.png"
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        print(cond.round(4).to_string())
        print(contrasts.round(4).to_string())
        print(strat.round(4).to_string())
        log.count("runs", len(runs))


if __name__ == "__main__":
    main()
