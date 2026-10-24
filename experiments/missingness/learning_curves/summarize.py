"""Learning curves on the training-median-filled holdout, 5% to 100% of training fields.

Combines the four sweep files (rows de-duplicated on model x fraction x draw, last kept, since a
retried step re-appends):
  experiments/sparse_gate_backbones/results_sweep.csv               TempCNN, CNN-BiLSTM (dense, gated)
  experiments/holdout_fill_check/results/sweep_deep_trainmed.csv    L-TAE, L-TAE-S
  experiments/holdout_fill_check/results/sweep_tabnet_trainmed.csv  pixel TabNet
  experiments/holdout_fill_check/results/sweep_classical_trainmed.csv  field XGBoost, pixel LR
Draws: 10 at 5/10/15%, 5 at 25/50/75%, 1 at 100%. Per model and fraction: mean holdout macro-F1
and a 95% t-interval over draws. Models use different train/validation splits, so draws are not
paired across models; differences are unpaired (Welch). Writes results/learning_curves.csv and
figures/learning_curves.{pdf,png}.
Run: python experiments/missingness/learning_curves/summarize.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP = os.path.join(REPO, "experiments")
FILES = ["sparse_gate_backbones/results_sweep.csv", "holdout_fill_check/results/sweep_deep_trainmed.csv",
         "holdout_fill_check/results/sweep_tabnet_trainmed.csv", "holdout_fill_check/results/sweep_classical_trainmed.csv"]
NAMES = {"cnn_bilstm_dense": "CNN-BiLSTM", "cnn_bilstm_sparsemax": "CNN-BiLSTM, gated",
         "tempcnn_dense": "TempCNN", "tempcnn_sparsemax": "TempCNN, gated",
         "ltae_pixel": "L-TAE", "ltae_sparse_pixel": "L-TAE-S", "tabnet_pixel": "TabNet (pixel)",
         "xgboost_field": "XGBoost (field)", "base_lr_pixel": "Logistic regression (pixel)"}
FAMILY = {"CNN-BiLSTM": "dense", "TempCNN": "dense", "L-TAE": "dense", "CNN-BiLSTM, gated": "sparse",
          "TempCNN, gated": "sparse", "L-TAE-S": "sparse", "TabNet (pixel)": "tree",
          "XGBoost (field)": "tree", "Logistic regression (pixel)": "linear"}


def load():
    d = pd.concat([pd.read_csv(os.path.join(EXP, f))[["model", "fraction", "subsample_seed", "oos_f1_macro"]]
                   for f in FILES])
    d["fraction"] = d["fraction"].round(4)
    d = d.drop_duplicates(["model", "fraction", "subsample_seed"], keep="last")
    d = d[d["model"].isin(NAMES)]
    d["model"] = d["model"].map(NAMES)
    return d


def summarize(d):
    rows = []
    for (m, f), g in d.groupby(["model", "fraction"]):
        x = g["oos_f1_macro"].values
        n, mu, sd = len(x), x.mean(), x.std(ddof=1) if len(x) > 1 else np.nan
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append({"model": m, "family": FAMILY[m], "fraction": f, "n_draws": n, "mean_f1m": mu,
                     "sd": sd, "ci_lo": mu - h, "ci_hi": mu + h})
    return pd.DataFrame(rows)


def welch(d, a, b, f):
    x = d[(d.model == a) & (d.fraction == f)]["oos_f1_macro"].values
    y = d[(d.model == b) & (d.fraction == f)]["oos_f1_macro"].values
    t = stats.ttest_ind(x, y, equal_var=False)
    return x.mean() - y.mean(), t.pvalue


def figure(s):
    sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
    from figstyle import apply_style, FAMILY_COLORS
    apply_style()
    col = {"dense": FAMILY_COLORS["dense"], "sparse": FAMILY_COLORS["sparse"], "tree": FAMILY_COLORS["tree"],
           "linear": FAMILY_COLORS["linear"]}
    marker = dict(zip(NAMES.values(), ["o", "s", "^", "D", "v", "P", "X", "*", "h"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m, g in s.groupby("model"):
        g = g.sort_values("fraction")
        ls = "--" if FAMILY[m] == "sparse" else "-"
        ax.plot(g.fraction, g.mean_f1m, ls=ls, marker=marker[m], ms=4, lw=1.3, color=col[FAMILY[m]], label=m)
        ax.fill_between(g.fraction, g.ci_lo, g.ci_hi, color=col[FAMILY[m]], alpha=0.08, lw=0)
    fr = sorted(s.fraction.unique())
    ax.set_xscale("log")
    ax.set_xticks(fr, [f"{100 * f:g}%" for f in fr])
    ax.minorticks_off()
    ax.set_xlabel("Share of training fields (log scale; 5% = 132 fields)", fontsize=10)
    ax.set_ylabel("Holdout macro-F1 (training-median fill)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "..", "figures", f"learning_curves.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    s = summarize(d).round(4)
    s.to_csv(os.path.join(HERE, "..", "results", "learning_curves.csv"), index=False)
    figure(s)
    pd.set_option("display.width", 200)
    print(s.pivot(index="model", columns="fraction", values="mean_f1m").round(3).to_string())
    print(s.pivot(index="model", columns="fraction", values="n_draws").to_string())
    ref = "Logistic regression (pixel)"
    for f in (0.05, 0.1, 0.15):
        for m in sorted(set(NAMES.values()) - {ref}):
            dlt, p = welch(d, m, ref, f)
            print(f"{100 * f:4g}%  {m:24s} - LR {dlt:+.3f}  p={p:.3f}")
