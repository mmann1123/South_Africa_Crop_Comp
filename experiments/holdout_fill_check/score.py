"""Score the holdout-fill re-runs and re-rank the benchmark.

For every re-run model: (1) does the 'orig' re-run reproduce the published prediction
file (fields that differ)? (2) macro-F1 / kappa / accuracy of published, orig and
trainmed; (3) paired field bootstrap (B=10,000, seed 42) and exact McNemar of trainmed
vs. orig (same checkpoints, only the holdout fill differs). For the xr_fresh models 'orig' uses
features rebuilt from the raw series (rebuild_xrfresh.py), which differ from the published
feature file on 85 fields, so 'orig' need not equal the published predictions; the
trainmed-vs-orig comparison is still like for like. The patch CNNs were not re-run and
enter the ranking with their published predictions. Run in the deep_field env:
  python experiments/holdout_fill_check/score.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

OOS = os.path.join(REPO, "out_of_sample")
PRED = os.path.join(HERE, "predictions")
B, SEED = 10_000, 42

# Table 1 name -> (published file, re-run key or None if not re-run)
MODELS = {
    "TabNet (raw), pixel": ("predictions_tabnet.csv", "tabnet"),
    "L-TAE-S, field": ("predictions_ltae_s_field.csv", "ltae_s_field"),
    "L-TAE-S, pixel": ("predictions_ltae_s.csv", "ltae_s"),
    "L-TAE, field": ("predictions_ltae_field.csv", "ltae_field"),
    "Transformer, pixel": ("predictions_transformer.csv", "transformer"),
    "TempCNN, pixel": ("predictions_tempcnn.csv", "tempcnn"),
    "TimesNet, pixel": ("predictions_timesnet.csv", "timesnet"),
    "XGBoost, field": ("predictions_xgboost.csv", "xgboost"),
    "LightGBM, pixel": ("predictions_base_lgbm.csv", "base_lgbm"),
    "Logistic Regression, pixel": ("predictions_base_lr.csv", "base_lr"),
    "LightGBM, field": ("predictions_lgbm.csv", "lgbm"),
    "Voting, pixel": ("predictions_voting.csv", "voting"),
    "Random Forest, pixel": ("predictions_base_rf.csv", "base_rf"),
    "XGBoost, pixel": ("predictions_base_xgb.csv", "base_xgb"),
    "L-TAE, pixel": ("predictions_ltae.csv", "ltae"),
    "TempCNN, field": ("predictions_tempcnn_field.csv", "tempcnn_field"),
    "Stacking, field": ("predictions_stacking.csv", "stacking"),
    "TimesNet, field": ("predictions_timesnet_field.csv", "timesnet_field"),
    "CNN-BiLSTM, pixel": ("predictions_cnn_bilstm.csv", "cnn_bilstm"),
    "CNN-BiLSTM, field": ("predictions_cnn_bilstm_field.csv", "cnn_bilstm_field"),
    "3D CNN, patch": ("predictions_3d_cnn.csv", None),
    "MiniRocket, pixel": ("predictions_minirocket.csv", "minirocket"),
    "SMOTE Stacked, field": ("predictions_smote_stacked.csv", "smote_stacked"),
    "MiniRocket, field": ("predictions_minirocket_field.csv", "minirocket_field"),
    "Multi-Channel 2D CNN, patch": ("predictions_multi_channel_cnn.csv", None),
    "TabNet (raw, field-avg), field": ("predictions_tabnet_temporal_field.csv", "tabnet_temporal_field"),
    "TabNet (xr_fresh), field": ("predictions_tabnet_field.csv", "tabnet_field"),
}


def read(path):
    return pd.read_csv(path)[["fid", "crop_name"]].drop_duplicates("fid").set_index("fid")["crop_name"]


def main():
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    classes = sorted(gt.unique())
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    rng = np.random.default_rng(SEED)

    def enc(s, idx):
        return s.reindex(idx).map(cidx).values

    rows = []
    for name, (pub_file, key) in MODELS.items():
        pub = read(os.path.join(OOS, pub_file))
        idx = gt.index.intersection(pub.index)
        yt = enc(gt, idx)
        r = {"model": name, "rerun": key is not None, "n_fields": len(idx)}
        r["pub_f1m"], r["pub_kappa"], r["pub_acc"] = metrics(confusion(yt, enc(pub, idx), K))
        r["final_f1m"] = r["pub_f1m"]
        if key and not all(os.path.exists(os.path.join(PRED, f"{key}_{v}.csv")) for v in ("orig", "trainmed")):
            print(f"[pending] {name}: re-run files missing, using published predictions")
            key = None
            r["rerun"] = False
        if key:
            orig = read(os.path.join(PRED, f"{key}_orig.csv"))
            med = read(os.path.join(PRED, f"{key}_trainmed.csv"))
            r["orig_fields_differ_from_pub"] = int((orig.reindex(pub.index) != pub).sum())
            yo, ym = enc(orig, idx), enc(med, idx)
            po, pm = metrics(confusion(yt, yo, K)), metrics(confusion(yt, ym, K))
            r["orig_f1m"], r["orig_kappa"], r["orig_acc"] = po
            r["med_f1m"], r["med_kappa"], r["med_acc"] = pm
            N = len(idx)
            d = np.empty((B, 3))
            for i in range(B):
                w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
                d[i] = metrics(confusion(yt, ym, K, w)) - metrics(confusion(yt, yo, K, w))
            r["delta_f1m"] = pm[0] - po[0]
            r["delta_f1m_lo"], r["delta_f1m_hi"] = np.percentile(d[:, 0], [2.5, 97.5])
            r["delta_acc"] = pm[2] - po[2]
            r["fields_changed"] = int((yo != ym).sum())
            n10, n01 = int(((ym == yt) & (yo != yt)).sum()), int(((ym != yt) & (yo == yt)).sum())
            r["mcnemar_p"] = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            r["final_f1m"] = pm[0]
        rows.append(r)

    df = pd.DataFrame(rows)
    df["rank_published"] = df["pub_f1m"].rank(ascending=False, method="min").astype(int)
    df["rank_trainmed"] = df["final_f1m"].rank(ascending=False, method="min").astype(int)
    out = os.path.join(HERE, "results", "fill_comparison.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.round(4).to_csv(out, index=False)
    cols = ["model", "rerun", "orig_fields_differ_from_pub", "pub_f1m", "orig_f1m", "med_f1m",
            "delta_f1m", "delta_f1m_lo", "delta_f1m_hi", "fields_changed", "rank_published", "rank_trainmed"]
    pd.set_option("display.width", 250)
    print(df.sort_values("final_f1m", ascending=False)[cols].round(3).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
