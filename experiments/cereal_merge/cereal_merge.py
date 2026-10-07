"""
Reviewer R1-7: how much accuracy is recovered if the confusable cereals are reported
as one class? No retraining: classes are merged in each model's holdout confusion
matrix (rows = true, cols = predicted) and metrics recomputed.

Schemes
  5-class (as published)
  wheat+barley  : Barley + Wheat -> "Wheat/Barley"; small-grain grazing kept separate (4 classes)
  all cereals   : Barley + Wheat + Small grain grazing -> "Cereals" (3 classes)

Sources: out_of_sample/scoring_results/confusion_matrix_*.csv (canonical), or the
per-field prediction CSV for models without one (L-TAE-S, Transformer).
Read-only on out_of_sample/; writes experiments/cereal_merge/results/cereal_merge.csv.
Run: ~/miniconda3/envs/deep_field/bin/python experiments/cereal_merge/cereal_merge.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OOS = os.path.join(REPO, "out_of_sample")
SR = os.path.join(OOS, "scoring_results")
sys.path.insert(0, OOS)

# Table 1 label -> canonical confusion-matrix name (or prediction CSV)
MODELS = {
    "TabNet (pixel)": "TabNet (pixel)",
    "L-TAE-S (field)": "predictions_ltae_s_field.csv",
    "L-TAE-S (pixel)": "predictions_ltae_s.csv",
    "L-TAE (field)": "L-TAE Field (field)",
    "Transformer (pixel)": "predictions_transformer.csv",
    "TempCNN (pixel)": "TempCNN (pixel)",
    "XGBoost (field)": "XGBoost (field)",
    "LightGBM (pixel)": "Base LightGBM (pixel)",
    "Logistic Regression (pixel)": "Base LR (pixel)",
    "LightGBM (field)": "LightGBM (field)",
    "Voting (pixel)": "Voting (pixel)",
    "Random Forest (pixel)": "Base RF (pixel)",
    "XGBoost (pixel)": "Base XGBoost (pixel)",
    "L-TAE (pixel)": "L-TAE (pixel)",
    "TempCNN (field)": "TempCNN Field (field)",
    "Stacking (field)": "Stacking (field)",
    "CNN-BiLSTM (pixel)": "CNN-BiLSTM (pixel)",
    "CNN-BiLSTM (field)": "CNN-BiLSTM Field (field)",
    "3D CNN (patch)": "3D CNN (patch)",
    "SMOTE Stacked (field)": "SMOTE Stacked (field)",
    "Multi-Channel 2D CNN (patch)": "Multi-Ch CNN (patch)",
    "TabNet raw field-avg (field)": "TabNet Temporal Field (field)",
    "TabNet xr_fresh (field)": "TabNet Field (field)",
}
SCHEMES = {
    "5-class": {},
    "wheat+barley": {"Barley": "Wheat/Barley", "Wheat": "Wheat/Barley"},
    "all cereals": {"Barley": "Cereals", "Wheat": "Cereals", "Small grain grazing": "Cereals"},
}


def load_cm(src, gt):
    if src.endswith(".csv"):
        p = pd.read_csv(os.path.join(OOS, src)).drop_duplicates("fid").set_index("fid")["crop_name"]
        m = pd.concat([gt.rename("t"), p.rename("p")], axis=1, join="inner")
        return pd.crosstab(m["t"], m["p"])
    f = os.path.join(SR, f"confusion_matrix_{src.replace(' ', '_').replace('-', '_')}.csv")
    return pd.read_csv(f, index_col=0)


def merge(cm, mapping):
    cm = cm.rename(index=lambda c: mapping.get(c, c), columns=lambda c: mapping.get(c, c))
    cm = cm.groupby(level=0).sum().T.groupby(level=0).sum().T
    labels = sorted(set(cm.index) | set(cm.columns))
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def metrics(cm):
    M = cm.values.astype(float)
    tp, sup, pred = np.diag(M), M.sum(1), M.sum(0)
    f1 = np.where(sup + pred > 0, 2 * tp / (sup + pred), 0.0)
    n = M.sum()
    po, pe = tp.sum() / n, (sup * pred).sum() / n ** 2
    return f1.mean(), (po - pe) / (1 - pe), po, dict(zip(cm.index, f1))


def main():
    from compare_predictions import load_ground_truth
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    rows = []
    for name, src in MODELS.items():
        cm = load_cm(src, gt)
        for scheme, mapping in SCHEMES.items():
            f1m, k, acc, pc = metrics(merge(cm, mapping))
            rows.append({"model": name, "scheme": scheme, "macro_f1": f1m, "kappa": k, "accuracy": acc,
                         "f1_merged_class": pc.get("Cereals", pc.get("Wheat/Barley", np.nan)),
                         **{f"f1_{c}": v for c, v in pc.items()}})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out.round(4).to_csv(os.path.join(HERE, "results", "cereal_merge.csv"), index=False)
    piv = out.pivot(index="model", columns="scheme", values="macro_f1").loc[list(MODELS), list(SCHEMES)]
    piv["gain_all_cereals"] = piv["all cereals"] - piv["5-class"]
    print("=== Macro-F1 by class scheme ===")
    print(piv.round(3).to_string())
    print("\n=== F1 of the merged class ===")
    print(out[out.scheme != "5-class"].pivot(index="model", columns="scheme", values="f1_merged_class")
          .loc[list(MODELS)].round(3).to_string())
    print("\n=== Accuracy by class scheme ===")
    print(out.pivot(index="model", columns="scheme", values="accuracy").loc[list(MODELS), list(SCHEMES)].round(3).to_string())


if __name__ == "__main__":
    main()
