"""Score the fill-value sweep.

Inputs: experiments/missingness/fill_sweep/predictions/ (fill_sweep.py, run_classical.sh) and,
for the classical models' published rule and train_median, experiments/holdout_fill_check/
predictions/{key}_orig.csv / {key}_trainmed.csv (rebuilt-feature baselines).

Outputs (experiments/missingness/fill_sweep/results/):
  f1_by_fill.csv      macro-F1 per model x fill, all fields and the September-observed /
                      September-missing subsets
  zero_vs_median.csv  zero fill minus train_median per model: macro-F1 difference with a paired
                      field bootstrap 95% CI (B = 10,000, seed 42), fields changed, McNemar p
  ranking.csv         model rank under each fill and Spearman correlation with train_median
Run in the deep_field env: python experiments/missingness/score_fill_sweep.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402
from config import MERGED_DL_TEST_PATH  # noqa: E402

SWEEP = os.path.join(HERE, "fill_sweep", "predictions")
FC = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
RES = os.path.join(HERE, "fill_sweep", "results")
FILLS = ["zero", "self_median", "train_mean", "train_median", "interp"]
B, SEED = 10_000, 42

# display name -> (key, family); family "raw" = raw-sequence models, "xr" = xr_fresh classical
MODELS = {
    "CNN-BiLSTM, pixel": ("cnn_bilstm", "raw"), "TempCNN, pixel": ("tempcnn", "raw"),
    "L-TAE, pixel": ("ltae", "raw"), "L-TAE-S, pixel": ("ltae_s", "raw"),
    "Transformer, pixel": ("transformer", "raw"), "TimesNet, pixel": ("timesnet", "raw"),
    "TabNet (raw), pixel": ("tabnet", "raw"), "MiniRocket, pixel": ("minirocket", "raw"),
    "TempCNN, field": ("tempcnn_field", "raw"), "L-TAE, field": ("ltae_field", "raw"),
    "L-TAE-S, field": ("ltae_s_field", "raw"), "CNN-BiLSTM, field": ("cnn_bilstm_field", "raw"),
    "TimesNet, field": ("timesnet_field", "raw"), "MiniRocket, field": ("minirocket_field", "raw"),
    "TabNet (raw, field-avg), field": ("tabnet_temporal_field", "raw"),
    "XGBoost, field": ("xgboost", "xr"), "LightGBM, field": ("lgbm", "xr"),
    "LightGBM, pixel": ("base_lgbm", "xr"), "Logistic Regression, pixel": ("base_lr", "xr"),
    "Random Forest, pixel": ("base_rf", "xr"), "XGBoost, pixel": ("base_xgb", "xr"),
    "Voting, pixel": ("voting", "xr"), "Stacking, field": ("stacking", "xr"),
    "SMOTE Stacked, field": ("smote_stacked", "xr"), "TabNet (xr_fresh), field": ("tabnet_field", "xr"),
}


def path(key, family, fill):
    if family == "raw":
        return os.path.join(SWEEP, f"{key}_{fill}.csv")
    if fill == "train_median":
        return os.path.join(FC, f"{key}_trainmed.csv")
    return os.path.join(SWEEP, f"{key}_fill_{fill}.csv")


def read(p):
    return pd.read_csv(p)[["fid", "crop_name"]].drop_duplicates("fid").set_index("fid")["crop_name"]


def main():
    os.makedirs(RES, exist_ok=True)
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    cl = sorted(gt.unique())
    ci = {c: i for i, c in enumerate(cl)}
    K = len(cl)
    t = pq.read_table(MERGED_DL_TEST_PATH, columns=["fid", "B2_September"]).to_pandas()
    gap = t["B2_September"].isna().groupby(t["fid"]).any()
    subsets = {"all": gt.index, "sept_observed": gap.index[~gap], "sept_missing": gap.index[gap]}

    def f1(pred, ids):
        idx = gt.index.intersection(pred.index).intersection(ids)
        return metrics(confusion(gt.reindex(idx).map(ci).values, pred.reindex(idx).map(ci).values, K))[0]

    rows, missing_files = [], []
    preds = {}
    for name, (key, fam) in MODELS.items():
        for fill in FILLS + (["published"] if fam == "xr" else []):
            p = os.path.join(FC, f"{key}_orig.csv") if fill == "published" else path(key, fam, fill)
            if not os.path.exists(p):
                missing_files.append(os.path.relpath(p, REPO))
                continue
            preds[(name, fill)] = read(p)
            rows.append({"model": name, "family": fam, "fill": fill,
                         **{f"f1m_{s}": f1(preds[(name, fill)], ids) for s, ids in subsets.items()}})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(RES, "f1_by_fill.csv"), index=False)

    rng = np.random.default_rng(SEED)
    zr = []
    for name in MODELS:
        if (name, "zero") not in preds or (name, "train_median") not in preds:
            continue
        a, b = preds[(name, "zero")], preds[(name, "train_median")]
        idx = gt.index.intersection(a.index).intersection(b.index)
        yt, ya, yb = (s.reindex(idx).map(ci).values for s in (gt, a, b))
        N = len(idx)
        d = np.array([metrics(confusion(yt, ya, K, w))[0] - metrics(confusion(yt, yb, K, w))[0]
                      for w in (np.bincount(rng.integers(0, N, N), minlength=N).astype(float) for _ in range(B))])
        n10, n01 = int(((ya == yt) & (yb != yt)).sum()), int(((ya != yt) & (yb == yt)).sum())
        zr.append({"model": name, "family": MODELS[name][1],
                   "f1m_zero": metrics(confusion(yt, ya, K))[0], "f1m_train_median": metrics(confusion(yt, yb, K))[0],
                   "delta": metrics(confusion(yt, ya, K))[0] - metrics(confusion(yt, yb, K))[0],
                   "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5),
                   "fields_changed": int((ya != yb).sum()),
                   "mcnemar_p": binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0})
    pd.DataFrame(zr).round(4).to_csv(os.path.join(RES, "zero_vs_median.csv"), index=False)

    wide = df.pivot(index="model", columns="fill", values="f1m_all")
    ranks = wide.rank(ascending=False, method="min")
    rk = ranks.copy()
    rk.loc["Spearman vs train_median"] = [spearmanr(wide[c], wide["train_median"], nan_policy="omit")[0]
                                          for c in wide.columns]
    rk.round(3).to_csv(os.path.join(RES, "ranking.csv"))

    pd.set_option("display.width", 220)
    order = wide["train_median"].sort_values(ascending=False).index
    print(wide.loc[order, [c for c in ["published"] + FILLS if c in wide.columns]].round(3).to_string())
    print()
    print(pd.DataFrame(zr).set_index("model")[["delta", "ci_lo", "ci_hi", "fields_changed"]].round(3).to_string())
    print()
    print(rk.loc[["Spearman vs train_median"]].round(3).to_string())
    if missing_files:
        print(f"\n{len(missing_files)} prediction files not yet available, e.g. {missing_files[:3]}")


if __name__ == "__main__":
    main()
