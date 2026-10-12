"""Score the R2-Q3a baselines and compare them with the published models.

Reads new predictions from results/predictions/ and published ones from out_of_sample/
(read-only). Writes only to experiments/recent_baselines/results/:

  provenance_check.csv   published comparators re-scored here must match Table 1 (F1m, 4 dp)
  summary.csv            Table-1-style row per model (in-region F1m, delta, OOR F1m + 95% CI,
                         kappa, weighted F1, accuracy, Xent, per-seed spread and time)
  paired_bootstrap.csv   paired field bootstrap (B=10,000, seed 42) + exact McNemar
  per_class/*.csv        per-class report for each new model

Metrics use experiments/field_reduction/score_oos.score (same methodology as
out_of_sample/compare_predictions.py, including the official hard-label Xent). The
paired bootstrap reuses confusion() and metrics() from
experiments/paired_bootstrap/paired_bootstrap.py and copies its resampling loop.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/score.py
"""
import contextlib
import io
import json
import os
import sys

import common  # noqa: F401  (sets sys.path for field_reduction helpers)
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(common.REPO, "experiments", "paired_bootstrap"))
from paired_bootstrap import confusion, metrics  # noqa: E402
from score_oos import score  # noqa: E402

import geopandas as gpd  # noqa: E402

OOS = os.path.join(common.REPO, "out_of_sample")
B, SEED = 10000, 42

# New models: display name -> (prediction stem, model dir name)
NEW = {
    "MiniRocket (pixel)": ("minirocket_pixel", "minirocket_pixel"),
    "MiniRocket (field)": ("minirocket_field", "minirocket_field"),
    "TimesNet (pixel)": ("timesnet_pixel", "timesnet_pixel"),
    "TimesNet (field)": ("timesnet_field", "timesnet_field"),
}
# Published comparators: display name -> (out_of_sample CSV, name in f1_macro_train_vs_oos.csv)
PUBLISHED = {
    "L-TAE-S (pixel)": ("predictions_ltae_s.csv", "L-TAE-S (pixel)"),
    "L-TAE-S (field)": ("predictions_ltae_s_field.csv", "L-TAE-S Field (field)"),
    "L-TAE (pixel)": ("predictions_ltae.csv", "L-TAE (pixel)"),
    "L-TAE (field)": ("predictions_ltae_field.csv", "L-TAE Field (field)"),
    "Transformer (pixel)": ("predictions_transformer.csv", "Transformer (pixel)"),
    "TempCNN (pixel)": ("predictions_tempcnn.csv", "TempCNN (pixel)"),
    "TempCNN (field)": ("predictions_tempcnn_field.csv", "TempCNN Field (field)"),
    "TabNet (pixel)": ("predictions_tabnet.csv", "TabNet (pixel)"),
    "Logistic Regression (pixel)": ("predictions_base_lr.csv", "Base LR (pixel)"),
    "XGBoost (field)": ("predictions_xgboost.csv", "XGBoost (field)"),
}
# (A, B, question); delta = metric(A) - metric(B)
PAIRS = [
    ("MiniRocket (pixel)", "L-TAE-S (pixel)", "dense random features vs. learned sparse selection"),
    ("MiniRocket (pixel)", "TabNet (pixel)", "dense random features vs. learned sparse selection"),
    ("MiniRocket (pixel)", "L-TAE (pixel)", "random convolutions vs. dense attention"),
    ("MiniRocket (pixel)", "Logistic Regression (pixel)", "random-convolution features vs. raw features, linear head"),
    ("MiniRocket (field)", "L-TAE-S (field)", "dense random features vs. learned sparse selection"),
    ("MiniRocket (field)", "L-TAE (field)", "random convolutions vs. dense attention"),
    ("MiniRocket (field)", "XGBoost (field)", "random convolutions vs. tree ensemble on xr_fresh"),
    ("TimesNet (pixel)", "L-TAE-S (pixel)", "modern capacity vs. sparse selection"),
    ("TimesNet (pixel)", "L-TAE (pixel)", "modern capacity vs. dense attention"),
    ("TimesNet (pixel)", "Transformer (pixel)", "modern capacity vs. attention capacity"),
    ("TimesNet (pixel)", "TempCNN (pixel)", "2D temporal convolution vs. 1D temporal convolution"),
    ("TimesNet (field)", "L-TAE-S (field)", "modern capacity vs. sparse selection"),
    ("TimesNet (field)", "L-TAE (field)", "modern capacity vs. dense attention"),
    ("TimesNet (field)", "TempCNN (field)", "2D temporal convolution vs. 1D temporal convolution"),
    ("MiniRocket (pixel)", "TimesNet (pixel)", "random features vs. learned capacity"),
    ("MiniRocket (field)", "TimesNet (field)", "random features vs. learned capacity"),
]


def quiet_score(csv, per_class_out=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return score(csv, per_class_out=per_class_out)


def ground_truth():
    gdf = gpd.read_file(common.TEST_LABELS_GEOJSON)
    gt = gdf[["fid", "crop_name"]].copy()
    gt["fid"] = gt["fid"].astype(int)
    return gt.drop_duplicates("fid").set_index("fid")["crop_name"]


def read_pred(csv):
    p = pd.read_csv(csv)[["fid", "crop_name"]].drop_duplicates("fid")
    return p.set_index("fid")["crop_name"]


def single_ci(yt, yp, K, rng):
    """Percentile 95% CI of macro-F1 from a field bootstrap."""
    N = len(yt)
    vals = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        vals[i] = metrics(confusion(yt, yp, K, w))[0]
    return np.percentile(vals, [2.5, 97.5])


def paired(yt, ya, yb, K, rng):
    """Paired bootstrap of (macro-F1, kappa, accuracy) deltas + exact McNemar."""
    N = len(yt)
    d = np.empty((B, 3))
    for i in range(B):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        d[i] = np.array(metrics(confusion(yt, ya, K, w))) - np.array(metrics(confusion(yt, yb, K, w)))
    ca, cb = ya == yt, yb == yt
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mcn = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return d, n10, n01, p_mcn


def main():
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    gt = ground_truth()
    pub_table = pd.read_csv(os.path.join(OOS, "scoring_results", "f1_macro_train_vs_oos.csv")).set_index("Model")

    csvs, in_region, seeds = {}, {}, {}
    for name, (csv, key) in PUBLISHED.items():
        csvs[name] = os.path.join(OOS, csv)
        in_region[name] = pub_table.loc[key, "Train F1 (macro)"]
    for name, (stem, mdir) in NEW.items():
        path = os.path.join(common.PRED_DIR, f"{stem}.csv")
        meta_path = os.path.join(common.MODELS_DIR, mdir, "metadata.json")
        if not (os.path.exists(path) and os.path.exists(meta_path)):
            print(f"skip {name}: not trained yet")
            continue
        meta = json.load(open(meta_path))
        csvs[name] = path
        in_region[name] = meta["metrics"]["f1_macro"]
        seeds[name] = (stem, meta)

    # 1. provenance: re-scored published comparators must reproduce Table 1
    prov = []
    for name, (csv, key) in PUBLISHED.items():
        s = quiet_score(csvs[name])
        pub = pub_table.loc[key, "OOS F1 (macro)"]
        prov.append({"model": name, "csv": csv, "f1m_here": round(s["f1_macro"], 4),
                     "f1m_published": round(pub, 4), "match": round(s["f1_macro"], 4) == round(pub, 4)})
    prov = pd.DataFrame(prov)
    prov.to_csv(os.path.join(common.RESULTS_DIR, "provenance_check.csv"), index=False)
    print(prov.to_string(index=False))
    if not prov["match"].all():
        sys.exit("Provenance check failed: a published comparator does not reproduce Table 1")

    preds = {n: read_pred(c) for n, c in csvs.items()}
    classes = sorted(set(gt.unique()).union(*[set(p.unique()) for p in preds.values()]))
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)

    def aligned(*names):
        m = pd.concat([gt.rename("t")] + [preds[n].rename(n) for n in names], axis=1, join="inner")
        return [m["t"].map(cidx).values] + [m[n].map(cidx).values for n in names]

    # 2. summary rows
    rows = []
    for name in list(NEW) + list(PUBLISHED):
        if name not in csvs:
            continue
        per_class = None
        if name in seeds:
            per_class = os.path.join(common.RESULTS_DIR, "per_class", f"{seeds[name][0]}.csv")
        s = quiet_score(csvs[name], per_class_out=per_class)
        yt, yp = aligned(name)
        lo, hi = single_ci(yt, yp, K, np.random.default_rng(SEED))
        row = {"model": name, "new": name in NEW, "in_region_f1m": round(in_region[name], 4),
               "oor_f1m": round(s["f1_macro"], 4), "delta": round(s["f1_macro"] - in_region[name], 4),
               "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "kappa": round(s["kappa"], 4),
               "f1_weighted": round(s["f1_weighted"], 4), "accuracy": round(s["accuracy"], 4),
               "xent": round(s["xent"], 4), "fields": s["fields"]}
        if name in seeds:
            stem, meta = seeds[name]
            seed_f1 = [quiet_score(os.path.join(common.PRED_DIR, f"{stem}_seed{sd}.csv"))["f1_macro"]
                       for sd in meta["seeds"]]
            row.update({"seed_f1m_mean": round(float(np.mean(seed_f1)), 4),
                        "seed_f1m_sd": round(float(np.std(seed_f1, ddof=1)), 4) if len(seed_f1) > 1 else None,
                        "per_seed_train_seconds": meta["per_seed_seconds_mean"]})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(common.RESULTS_DIR, "summary.csv"), index=False)
    print(summary.to_string(index=False))

    # 3. paired bootstrap (fresh rng per pair -> results independent of pair order)
    out = []
    for a, b, q in PAIRS:
        if a not in preds or b not in preds:
            continue
        yt, ya, yb = aligned(a, b)
        d, n10, n01, p_mcn = paired(yt, ya, yb, K, np.random.default_rng(SEED))
        pa, pb = np.array(metrics(confusion(yt, ya, K))), np.array(metrics(confusion(yt, yb, K)))
        for j, met in enumerate(["macro_f1", "kappa", "accuracy"]):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            p = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
            out.append({"A": a, "B": b, "question": q, "metric": met, "n_fields": len(yt),
                        "A_point": round(pa[j], 4), "B_point": round(pb[j], 4),
                        "delta": round(pa[j] - pb[j], 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "p_boot": round(p, 4), "sig_95": bool(lo > 0 or hi < 0),
                        "mcnemar_A_only_correct": n10, "mcnemar_B_only_correct": n01,
                        "mcnemar_p": float(f"{p_mcn:.3g}")})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(common.RESULTS_DIR, "paired_bootstrap.csv"), index=False)
    if len(res):
        print(res[res.metric == "macro_f1"].drop(columns=["question", "n_fields", "metric"]).to_string(index=False))


if __name__ == "__main__":
    main()
