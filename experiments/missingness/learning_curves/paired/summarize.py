"""Paired low-data comparison: 5, 10 and 15 percent of training fields, 10 draws, every model
trained on the identical fields of each draw (PAIRED_FID_FILE; make_splits.py), scored on the
training-median-filled holdout.

For each fraction and model: mean holdout macro-F1 and a 95 percent t-interval over the 10 draws.
For each fraction and model pair against the reference (pixel logistic regression by default):
the per-draw difference, its mean, a paired t-interval and a Wilcoxon signed-rank p-value, and
how many of the 10 draws favour the model. Also the gated-minus-dense contrast for TempCNN,
CNN-BiLSTM and L-TAE.
Writes results/paired_summary.csv, results/paired_contrasts.csv and prints both.
Run: python experiments/missingness/learning_curves/paired/summarize.py [--ref base_lr_pixel]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NAMES = {"cnn_bilstm_dense": "CNN-BiLSTM", "cnn_bilstm_sparsemax": "CNN-BiLSTM, gated",
         "tempcnn_dense": "TempCNN", "tempcnn_sparsemax": "TempCNN, gated",
         "ltae_pixel": "L-TAE", "ltae_sparse_pixel": "L-TAE-S", "tabnet_pixel": "TabNet (pixel)",
         "xgboost_field": "XGBoost (field)", "base_lr_pixel": "Logistic regression (pixel)"}
GATE_PAIRS = [("tempcnn_sparsemax", "tempcnn_dense"), ("cnn_bilstm_sparsemax", "cnn_bilstm_dense"),
              ("ltae_sparse_pixel", "ltae_pixel")]


def load():
    files = sorted(glob.glob(os.path.join(RES, "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith("paired_")]
    d = pd.concat([pd.read_csv(f)[["model", "fraction", "subsample_seed", "oos_f1_macro"]] for f in files])
    d["fraction"] = d["fraction"].round(4)
    return d.drop_duplicates(["model", "fraction", "subsample_seed"], keep="last")


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - h, x.mean() + h


def contrast(d, a, b, frac):
    w = d[d["fraction"] == frac].pivot(index="subsample_seed", columns="model", values="oos_f1_macro")
    if a not in w or b not in w:
        return None
    diff = (w[a] - w[b]).dropna()
    if len(diff) < 2:
        return None
    m, lo, hi = t_ci(diff)
    p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    return {"fraction": frac, "model": NAMES.get(a, a), "reference": NAMES.get(b, b), "draws": len(diff),
            "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p, "draws_favouring_model": int((diff > 0).sum())}


def main(ref):
    d = load()
    rows = []
    for (m, f), g in d.groupby(["model", "fraction"]):
        mu, lo, hi = t_ci(g["oos_f1_macro"])
        rows.append({"model": NAMES.get(m, m), "key": m, "fraction": f, "n_draws": len(g), "mean_f1m": mu, "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(rows).sort_values(["fraction", "mean_f1m"], ascending=[True, False])
    cons = []
    for f in sorted(d["fraction"].unique()):
        for m in d["model"].unique():
            if m != ref:
                c = contrast(d, m, ref, f)
                if c:
                    cons.append(c)
        for a, b in GATE_PAIRS:
            c = contrast(d, a, b, f)
            if c:
                cons.append(c)
    cons = pd.DataFrame(cons)
    os.makedirs(RES, exist_ok=True)
    summ.to_csv(os.path.join(RES, "paired_summary.csv"), index=False)
    cons.to_csv(os.path.join(RES, "paired_contrasts.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print()
    print(cons.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="base_lr_pixel")
    main(**vars(ap.parse_args()))
