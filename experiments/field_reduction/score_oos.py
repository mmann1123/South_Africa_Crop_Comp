"""Score an OOS prediction CSV against holdout ground truth.

Replicates the exact metric methodology of out_of_sample/compare_predictions.py
(weighted/macro F1, Cohen kappa, and one-vs-rest hard-label cross-entropy) so the
results are directly comparable to Tables 3 and 4 in the writeup.

Usage:
    python score_oos.py results/predictions/ltae_sparse_field_frac_1.00.csv
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, log_loss,
)
from sklearn.preprocessing import label_binarize

from experiment_config import TEST_LABELS_GEOJSON


def score(pred_csv):
    gdf = gpd.read_file(TEST_LABELS_GEOJSON)
    gt = gdf[["fid", "crop_name"]].copy()
    gt["fid"] = gt["fid"].astype(int)
    gt = gt.rename(columns={"crop_name": "true_label"})

    pred = pd.read_csv(pred_csv)
    merged = gt.merge(pred, on="fid", how="inner")
    y_true = merged["true_label"]
    y_pred = merged["crop_name"]

    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    # One-vs-rest hard-label cross entropy (matches compare_predictions.py)
    labels = sorted(y_true.unique())
    y_true_bin = label_binarize(y_true, classes=labels)
    y_pred_bin = label_binarize(y_pred, classes=labels)
    eps = 1e-7
    y_pred_prob = np.clip(y_pred_bin.astype(float), eps, 1 - eps)
    xent = log_loss(y_true_bin, y_pred_prob)

    print(f"File:   {pred_csv}")
    print(f"Fields: {len(merged)}")
    print(f"  ST F1 macro:    {f1_m:.4f}")
    print(f"  ST F1 weighted: {f1_w:.4f}")
    print(f"  ST Cohen kappa: {kappa:.4f}")
    print(f"  ST Accuracy:    {acc:.4f}")
    print(f"  ST Cross-Ent:   {xent:.4f}")
    return dict(fields=len(merged), f1_macro=f1_m, f1_weighted=f1_w,
                kappa=kappa, accuracy=acc, xent=xent)


if __name__ == "__main__":
    score(sys.argv[1])
