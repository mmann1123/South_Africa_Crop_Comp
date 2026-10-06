"""
Paired field-bootstrap tests for between-model differences on the holdout tile.

In every replicate the same resampled holdout fields are scored for both models,
so field difficulty cancels out (a paired test). Reports Delta macro-F1, Delta
Cohen's kappa, and Delta accuracy with 95% percentile CIs and two-sided bootstrap
p-values, plus an exact McNemar test on per-field correct/incorrect.

Every model is read from its per-field prediction CSV; the script first checks
that each CSV reproduces the canonical confusion matrix in scoring_results/
(or, for models without one, is listed in NO_CM) and stops if any does not.

Output: scoring_results/paired_bootstrap_gaps.csv
Origin: experiments/paired_bootstrap/paired_bootstrap.py (reviewer R1-3).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "scoring_results")
sys.path.insert(0, SCRIPT_DIR)

B = 10000
SEED = 42

PRED = {
    "TabNet (pixel)": "predictions_tabnet.csv",
    "L-TAE-S (pixel)": "predictions_ltae_s.csv",
    "L-TAE-S Field (field)": "predictions_ltae_s_field.csv",
    "L-TAE (pixel)": "predictions_ltae.csv",
    "L-TAE Field (field)": "predictions_ltae_field.csv",
    "Transformer (pixel)": "predictions_transformer.csv",
    "CNN-BiLSTM (pixel)": "predictions_cnn_bilstm.csv",
    "CNN-BiLSTM Field (field)": "predictions_cnn_bilstm_field.csv",
    "TempCNN (pixel)": "predictions_tempcnn.csv",
    "TempCNN Field (field)": "predictions_tempcnn_field.csv",
    "TabNet Temporal Field (field)": "predictions_tabnet_temporal_field.csv",
    "XGBoost (field)": "predictions_xgboost.csv",
    "Base XGBoost (pixel)": "predictions_base_xgb.csv",
}
NO_CM = {"L-TAE-S (pixel)", "L-TAE-S Field (field)", "Transformer (pixel)"}

# (A, B): Delta = metric(A) - metric(B)
PAIRS = [
    ("L-TAE-S (pixel)", "L-TAE (pixel)"),
    ("L-TAE-S Field (field)", "L-TAE Field (field)"),
    ("Transformer (pixel)", "L-TAE (pixel)"),
    ("L-TAE-S (pixel)", "Transformer (pixel)"),
    ("TabNet (pixel)", "L-TAE-S (pixel)"),
    ("TabNet (pixel)", "L-TAE (pixel)"),
    ("TabNet (pixel)", "CNN-BiLSTM (pixel)"),
    ("L-TAE-S (pixel)", "CNN-BiLSTM (pixel)"),
    ("L-TAE Field (field)", "L-TAE (pixel)"),
    ("CNN-BiLSTM Field (field)", "CNN-BiLSTM (pixel)"),
    ("TempCNN Field (field)", "TempCNN (pixel)"),
    ("TabNet (pixel)", "TabNet Temporal Field (field)"),
    ("XGBoost (field)", "Base XGBoost (pixel)"),
]


def confusion(yt, yp, K, w=None):
    return np.bincount(yt * K + yp, weights=w, minlength=K * K).reshape(K, K)


def metrics(cm):
    """macro-F1, Cohen's kappa, accuracy from a KxK confusion matrix (rows = true)."""
    cm = cm.astype(float)
    n = cm.sum()
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    po = tp.sum() / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return np.array([f1.mean(), (po - pe) / (1 - pe), po])


def main():
    from compare_predictions import load_ground_truth
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    preds = {n: pd.read_csv(os.path.join(SCRIPT_DIR, f))[["fid", "crop_name"]]
                .drop_duplicates("fid").set_index("fid")["crop_name"] for n, f in PRED.items()}
    classes = sorted(set(gt.unique()).union(*[set(p.unique()) for p in preds.values()]))
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)

    # Provenance: each CSV must reproduce its canonical confusion matrix.
    for n, p in preds.items():
        if n in NO_CM:
            continue
        cmf = os.path.join(RESULTS_DIR, f"confusion_matrix_{n.replace(' ', '_').replace('-', '_')}.csv")
        canon = pd.read_csv(cmf, index_col=0)
        m = pd.concat([gt.rename("t"), p.rename("p")], axis=1, join="inner")
        ours = pd.DataFrame(confusion(m["t"].map(cidx).values, m["p"].map(cidx).values, K),
                            index=classes, columns=classes).loc[canon.index, canon.columns]
        if not (ours.values == canon.values).all():
            sys.exit(f"Provenance check failed: {n} does not reproduce {os.path.basename(cmf)}")
    print("Provenance check passed for all models with a canonical confusion matrix.")

    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in PAIRS:
        m = pd.concat([gt.rename("t"), preds[a].rename("a"), preds[b].rename("b")], axis=1, join="inner")
        yt, ya, yb = (m[c].map(cidx).values for c in ("t", "a", "b"))
        N = len(m)
        pa, pb = metrics(confusion(yt, ya, K)), metrics(confusion(yt, yb, K))
        d = np.empty((B, 3))
        for i in range(B):
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
            d[i] = metrics(confusion(yt, ya, K, w)) - metrics(confusion(yt, yb, K, w))
        ca, cb = ya == yt, yb == yt
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_mcn = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        for j, met in enumerate(["macro_f1", "kappa", "accuracy"]):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            rows.append({"Model A": a, "Model B": b, "metric": met,
                         "A": round(pa[j], 4), "B": round(pb[j], 4), "Delta": round(pa[j] - pb[j], 4),
                         "Delta_lo": round(lo, 4), "Delta_hi": round(hi, 4),
                         "p_two_sided": round(min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean())), 4),
                         "significant_95": bool(lo > 0 or hi < 0),
                         "A_only_correct": n10, "B_only_correct": n01, "mcnemar_p": round(p_mcn, 4)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "paired_bootstrap_gaps.csv"), index=False)
    print(out[out.metric == "macro_f1"].drop(columns="metric").to_string(index=False))


if __name__ == "__main__":
    main()
