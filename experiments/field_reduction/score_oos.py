"""Score an OOS prediction CSV against holdout ground truth.

Replicates the exact metric methodology of out_of_sample/compare_predictions.py
(weighted/macro F1, Cohen kappa, and one-vs-rest hard-label cross-entropy) so the
results are directly comparable to Tables 3 and 4 in the writeup.

Usage:
    python score_oos.py results/predictions/ltae_sparse_field_frac_1.00.csv
    python score_oos.py results/predictions/ltae_sparse_pixel_frac_1.00.csv \
        --per-class-out ../../out_of_sample/scoring_results/per_class_L_TAE_S_(pixel).csv
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, cohen_kappa_score, log_loss,
)
from sklearn.preprocessing import label_binarize

from experiment_config import TEST_LABELS_GEOJSON


def score(pred_csv, per_class_out=None):
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

    per_crop_ce = {}
    for i, crop in enumerate(labels):
        per_crop_ce[crop] = log_loss(y_true_bin[:, i], y_pred_prob[:, i])

    print(f"File:   {pred_csv}")
    print(f"Fields: {len(merged)}")
    print(f"  ST F1 macro:    {f1_m:.4f}")
    print(f"  ST F1 weighted: {f1_w:.4f}")
    print(f"  ST Cohen kappa: {kappa:.4f}")
    print(f"  ST Accuracy:    {acc:.4f}")
    print(f"  ST Cross-Ent:   {xent:.4f}")

    if per_class_out:
        per_class = pd.DataFrame(
            classification_report(y_true, y_pred, output_dict=True)
        ).T
        per_class.index.name = "class"
        per_class["cross_entropy"] = per_class.index.map(
            lambda c: per_crop_ce.get(c, np.nan)
        )
        os.makedirs(os.path.dirname(per_class_out) or ".", exist_ok=True)
        per_class.to_csv(per_class_out)
        print(f"  Per-class CSV: {per_class_out}")

    return dict(fields=len(merged), f1_macro=f1_m, f1_weighted=f1_w,
                kappa=kappa, accuracy=acc, xent=xent)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pred_csv")
    p.add_argument("--per-class-out", default=None,
                   help="Path to write per-class CSV in the same format as "
                        "out_of_sample/scoring_results/per_class_*.csv")
    args = p.parse_args()
    score(args.pred_csv, per_class_out=args.per_class_out)
