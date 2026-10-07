"""
Mini-experiment (reviewer R2-Q7, transfer distance): does holdout accuracy fall with
distance from the training region?

For each holdout field, compute the distance (km) from its centroid to the nearest
training field, bin fields by distance, and score each model per bin (macro-F1 with a
field bootstrap CI, plus accuracy). Because crop mix can differ by bin, it also reports
class composition per bin and a class-adjusted test: a logistic regression of per-field
correctness on distance with crop-class fixed effects.

Read-only on out_of_sample/; writes to experiments/distance_decay/results/.
Run: ~/miniconda3/envs/deep_field/bin/python experiments/distance_decay/distance_decay.py
"""
import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from shapely.ops import unary_union
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OOS = os.path.join(REPO, "out_of_sample")
OUT = os.path.join(HERE, "results")
BASE = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp"

MODELS = {
    "TabNet (pixel)": "predictions_tabnet.csv",
    "L-TAE-S (pixel)": "predictions_ltae_s.csv",
    "L-TAE (pixel)": "predictions_ltae.csv",
    "XGBoost (field)": "predictions_xgboost.csv",
    "CNN-BiLSTM (pixel)": "predictions_cnn_bilstm.csv",
}
BINS = [0, 5, 10, 15, 25]
B = 2000


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.concat([gpd.read_file(f) for f in glob.glob(
        f"{BASE}/ref_fusion_competition_south_africa_train_labels/*/labels.geojson")])
    te = gpd.read_file(glob.glob(f"{BASE}/ref_fusion_competition_south_africa_test_labels/*/labels.geojson")[0])
    crs = tr.estimate_utm_crs()
    tr, te = tr.to_crs(crs), te.to_crs(crs)
    union = unary_union(tr.geometry.values)
    te["dist_km"] = te.geometry.centroid.distance(union) / 1e3
    te["bin"] = pd.cut(te["dist_km"], BINS, include_lowest=True)
    gt = te.drop_duplicates("fid").set_index("fid")[["crop_name", "dist_km", "bin"]]

    comp = pd.crosstab(gt["bin"], gt["crop_name"], normalize="index").round(2)
    comp["n_fields"] = gt["bin"].value_counts().sort_index()
    comp.to_csv(os.path.join(OUT, "class_composition_by_bin.csv"))
    print("=== Class composition by distance bin ===")
    print(comp.to_string())

    rng = np.random.default_rng(42)
    rows, slopes = [], []
    for name, f in MODELS.items():
        p = pd.read_csv(os.path.join(OOS, f)).drop_duplicates("fid").set_index("fid")["crop_name"]
        m = gt.join(p.rename("pred"), how="inner")
        m["correct"] = (m["pred"] == m["crop_name"]).astype(int)
        labels = sorted(m["crop_name"].unique())
        for b, g in m.groupby("bin", observed=True):
            yt, yp = g["crop_name"].values, g["pred"].values
            f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
            bs = [f1_score(yt[i], yp[i], labels=labels, average="macro", zero_division=0)
                  for i in (rng.integers(0, len(g), len(g)) for _ in range(B))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"model": name, "bin_km": str(b), "n": len(g), "macro_f1": round(f1, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "accuracy": round(g["correct"].mean(), 3)})
        # class-adjusted distance effect on per-field correctness (per 10 km)
        fit = smf.logit("correct ~ I(dist_km/10) + C(crop_name)", data=m).fit(disp=0)
        coef, (clo, chi) = fit.params.iloc[1], fit.conf_int().iloc[1]
        slopes.append({"model": name, "logodds_per_10km": round(coef, 3), "ci_lo": round(clo, 3),
                       "ci_hi": round(chi, 3), "p": round(fit.pvalues.iloc[1], 4),
                       "odds_ratio_per_10km": round(np.exp(coef), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "macro_f1_by_distance.csv"), index=False)
    sl = pd.DataFrame(slopes)
    sl.to_csv(os.path.join(OUT, "distance_effect_class_adjusted.csv"), index=False)
    print("\n=== Macro-F1 by distance bin ===")
    print(res.pivot(index="model", columns="bin_km", values="macro_f1").to_string())
    print("\n=== Accuracy by distance bin ===")
    print(res.pivot(index="model", columns="bin_km", values="accuracy").to_string())
    print("\n=== Class-adjusted effect of distance on correctness (logit, per 10 km) ===")
    print(sl.to_string(index=False))


if __name__ == "__main__":
    main()
