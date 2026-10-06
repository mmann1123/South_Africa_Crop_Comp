"""
Paired field-bootstrap tests for key between-model comparisons (reviewer R1-3).

The manuscript's Methods describe a *paired* bootstrap for between-model gaps, but
out_of_sample/bootstrap_ci.py resamples each model independently (unpaired) and does
not include the L-TAE-S vs. L-TAE comparisons. This script runs a true paired test:
in every replicate the SAME resampled holdout fields are scored for both models, so
field difficulty cancels out.

Read-only with respect to the published pipeline: it reads out_of_sample/predictions_*.csv
and the canonical confusion matrices, and writes only to experiments/paired_bootstrap/results/.

Steps
  1. Provenance check: each prediction CSV must reproduce the published Table 2 macro-F1
     and kappa, and (where one exists) its canonical confusion matrix exactly.
  2. Paired bootstrap (B = 10,000, fixed seed) of Delta macro-F1, Delta kappa, Delta accuracy.
     (Table 2's "Xent" is computed from hard labels and equals -log(1e-7) x error rate,
     so Delta accuracy carries the same information.)
  3. Exact McNemar test on per-field correct/incorrect as a complementary check.

Run (any env with pandas, numpy, scipy, geopandas):
  ~/miniconda3/envs/deep_field/bin/python experiments/paired_bootstrap/paired_bootstrap.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OOS = os.path.join(REPO, "out_of_sample")
CM_DIR = os.path.join(OOS, "scoring_results")
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
sys.path.insert(0, OOS)
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))

B = 10000
SEED = 42

# name -> (prediction CSV, canonical confusion-matrix CSV or None, published F1m, published kappa)
MODELS = {
    "L-TAE-S (pixel)": ("predictions_ltae_s.csv", None, 0.60, 0.53),
    "L-TAE-S (field)": ("predictions_ltae_s_field.csv", None, 0.60, 0.52),
    "L-TAE (pixel)": ("predictions_ltae.csv", "confusion_matrix_L_TAE_(pixel).csv", 0.58, 0.49),
    "L-TAE (field)": ("predictions_ltae_field.csv", "confusion_matrix_L_TAE_Field_(field).csv", 0.59, 0.47),
    "Transformer (pixel)": ("predictions_transformer.csv", None, 0.58, 0.50),
    "TabNet (pixel)": ("predictions_tabnet.csv", "confusion_matrix_TabNet_(pixel).csv", 0.60, 0.54),
}
# Retrained L-TAE pixel (same harness as L-TAE-S frac_1.00); no published value to match.
RETRAIN_CSV = os.path.join(SCRIPT_DIR, "rerun", "predictions_ltae_retrain.csv")
if os.path.exists(RETRAIN_CSV):
    MODELS["L-TAE retrain (pixel)"] = (RETRAIN_CSV, None, None, None)

# (A, B, question). Delta = metric(A) - metric(B).
PAIRS = [
    ("L-TAE-S (pixel)", "L-TAE (pixel)", "sparse gate, pixel level"),
    ("L-TAE-S (field)", "L-TAE (field)", "sparse gate, field level"),
    ("Transformer (pixel)", "L-TAE (pixel)", "attention capacity, pixel level"),
    ("TabNet (pixel)", "L-TAE-S (pixel)", "TabNet vs. L-TAE-S parity"),
    ("L-TAE-S (pixel)", "Transformer (pixel)", "sparsity vs. attention capacity, pixel level"),
    ("L-TAE-S (pixel)", "L-TAE retrain (pixel)", "sparse gate, pixel level (retrained L-TAE)"),
    ("Transformer (pixel)", "L-TAE retrain (pixel)", "attention capacity, pixel level (retrained L-TAE)"),
    ("TabNet (pixel)", "L-TAE retrain (pixel)", "TabNet vs. retrained L-TAE"),
]
# NOTE: out_of_sample/predictions_ltae.csv does NOT reproduce the published L-TAE (pixel)
# row (F1m 0.54 vs 0.58 published); pairs involving it are flagged in the provenance check.


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
    return f1.mean(), (po - pe) / (1 - pe), po


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from compare_predictions import load_ground_truth
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]

    preds = {}
    for name, (csv, _, _, _) in MODELS.items():
        p = pd.read_csv(os.path.join(OOS, csv))[["fid", "crop_name"]].drop_duplicates("fid")
        preds[name] = p.set_index("fid")["crop_name"]

    classes = sorted(set(gt.unique()).union(*[set(p.unique()) for p in preds.values()]))
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)

    # ---- 1. provenance check ----
    rows = []
    for name, (csv, cmf, pub_f1, pub_k) in MODELS.items():
        m = pd.concat([gt.rename("t"), preds[name].rename("p")], axis=1, join="inner")
        cm = confusion(m["t"].map(cidx).values, m["p"].map(cidx).values, K)
        f1, k, acc = metrics(cm)
        cm_match = None
        if cmf and os.path.exists(os.path.join(CM_DIR, cmf)):
            canon = pd.read_csv(os.path.join(CM_DIR, cmf), index_col=0)
            ours = pd.DataFrame(cm.astype(int), index=classes, columns=classes)
            ours = ours.loc[canon.index, canon.columns]
            cm_match = bool((ours.values == canon.values).all())
        rows.append({"model": name, "n_fields": len(m), "f1m": round(f1, 4), "pub_f1m": pub_f1,
                     "kappa": round(k, 4), "pub_kappa": pub_k, "accuracy": round(acc, 4),
                     "matches_published_2dp": None if pub_f1 is None else (round(f1, 2) == pub_f1 and round(k, 2) == pub_k),
                     "matches_canonical_cm": cm_match})
    prov = pd.DataFrame(rows)
    prov.to_csv(os.path.join(OUT_DIR, "provenance_check.csv"), index=False)
    print("=== Provenance check ===")
    print(prov.to_string(index=False))

    # ---- 2. paired bootstrap + 3. McNemar ----
    rng = np.random.default_rng(SEED)
    out = []
    for a, b, q in PAIRS:
        if a not in preds or b not in preds:
            continue
        m = pd.concat([gt.rename("t"), preds[a].rename("a"), preds[b].rename("b")],
                      axis=1, join="inner")
        yt = m["t"].map(cidx).values
        ya = m["a"].map(cidx).values
        yb = m["b"].map(cidx).values
        N = len(m)
        pa = np.array(metrics(confusion(yt, ya, K)))
        pb = np.array(metrics(confusion(yt, yb, K)))
        d = np.empty((B, 3))
        for i in range(B):
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
            d[i] = np.array(metrics(confusion(yt, ya, K, w))) - np.array(metrics(confusion(yt, yb, K, w)))
        ca, cb = ya == yt, yb == yt
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_mcn = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        for j, met in enumerate(["macro_f1", "kappa", "accuracy"]):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            p = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
            out.append({"A": a, "B": b, "question": q, "metric": met, "n_fields": N,
                        "A_point": round(pa[j], 4), "B_point": round(pb[j], 4),
                        "delta": round(pa[j] - pb[j], 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "p_boot": round(p, 4), "sig_95": bool(lo > 0 or hi < 0),
                        "mcnemar_A_only_correct": n10, "mcnemar_B_only_correct": n01,
                        "mcnemar_p": round(p_mcn, 4)})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT_DIR, "paired_bootstrap.csv"), index=False)
    print("\n=== Paired bootstrap (B=%d) ===" % B)
    print(res.drop(columns=["question", "n_fields"]).to_string(index=False))


if __name__ == "__main__":
    main()
