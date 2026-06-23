"""
Bootstrap confidence intervals for spatial-transfer (holdout) macro-F1.

A macro-F1 field-bootstrap depends only on each model's per-field (true, pred)
outcome multiset, which is fully encoded by its holdout confusion matrix. We
therefore source every model's outcomes from the canonical World-A confusion
matrices in scoring_results/ (confusion_matrix_<Model>.csv), which reproduce the
published Table values exactly; the three models without a confusion-matrix file
(Transformer, both L-TAE-S variants) are read from their per-field prediction CSVs.
This keeps the CIs centered on the manuscript's reported point estimates regardless
of any later re-runs of the inference scripts.

For each model we resample its holdout field outcomes with replacement
(B = 10,000, fixed seed) and recompute macro-F1 and per-class F1, reporting the
2.5th-97.5th percentile interval. Between-model gap CIs use an independent
(unpaired) bootstrap, which is conservative relative to a paired test (it ignores
the positive correlation from the shared test set), so any gap it flags as
significant is robustly so.

Outputs to out_of_sample/scoring_results/:
  - bootstrap_ci.csv            per-model macro-F1 point + mean + 95% CI
  - bootstrap_per_class_ci.csv  per-model per-class F1 point + 95% CI
  - bootstrap_pairwise_gaps.csv Delta CIs + bootstrap p-values for key comparisons
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "scoring_results")

# Models whose outcomes come from per-field CSVs (no confusion-matrix file written).
CSV_FALLBACK = {
    "Transformer (pixel)": os.path.join(SCRIPT_DIR, "predictions_transformer.csv"),
    "L-TAE-S (pixel)": os.path.join(SCRIPT_DIR, "predictions_ltae_s.csv"),
    "L-TAE-S Field (field)": os.path.join(SCRIPT_DIR, "predictions_ltae_s_field.csv"),
}

# Key rank comparisons to test (modelA, modelB); Delta = F1(A) - F1(B).
PAIRWISE = [
    ("TabNet (pixel)", "CNN-BiLSTM (pixel)"),
    ("TabNet (pixel)", "L-TAE-S (pixel)"),
    ("TabNet (pixel)", "L-TAE (pixel)"),
    ("L-TAE-S Field (field)", "L-TAE (pixel)"),
    ("L-TAE-S (pixel)", "CNN-BiLSTM (pixel)"),
    ("L-TAE Field (field)", "L-TAE (pixel)"),
    ("CNN-BiLSTM Field (field)", "CNN-BiLSTM (pixel)"),
    ("TabNet (pixel)", "TabNet Temporal Field (field)"),
    ("XGBoost (field)", "Base XGBoost (pixel)"),
]

B = 10000
SEED = 42


def cm_filename(name):
    safe = name.replace(" ", "_").replace("-", "_")
    return os.path.join(RESULTS_DIR, f"confusion_matrix_{safe}.csv")


def outcomes_from_cm(path):
    """Return (true_idx array, pred_idx array, class list) from a confusion matrix CSV."""
    cm = pd.read_csv(path, index_col=0)
    classes = list(cm.index)
    M = cm.values.astype(int)
    yt, yp = [], []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if M[i, j]:
                yt.append(np.full(M[i, j], i))
                yp.append(np.full(M[i, j], j))
    return np.concatenate(yt), np.concatenate(yp), classes


def outcomes_from_csv(path, gt_s):
    """Return (true_idx, pred_idx, class list) by merging a per-field CSV with ground truth."""
    p = pd.read_csv(path)[["fid", "crop_name"]].drop_duplicates("fid")
    m = p.merge(gt_s.rename("true"), left_on="fid", right_index=True)
    classes = sorted(set(m["true"]) | set(m["crop_name"]))
    idx = {c: i for i, c in enumerate(classes)}
    return (m["true"].map(idx).values, m["crop_name"].map(idx).values, classes)


def macro_f1_from_cells(cells):
    """cells: KxK confusion counts (rows=true, cols=pred). Returns (macro_f1, per_class_f1)."""
    tp = np.diag(cells).astype(float)
    denom = cells.sum(axis=0) + cells.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return f1.mean(), f1


def bootstrap_model(yt, yp, classes, rng):
    """Independent field-resample bootstrap. Returns point/boot macro-F1 and per-class F1."""
    K = len(classes)
    N = len(yt)
    cell_id = yt * K + yp
    cells0 = np.bincount(cell_id, minlength=K * K).reshape(K, K)
    m0, pc0 = macro_f1_from_cells(cells0)
    bm = np.empty(B)
    bpc = np.empty((B, K))
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        w = np.bincount(idx, minlength=N).astype(float)
        cells = np.bincount(cell_id, weights=w, minlength=K * K).reshape(K, K)
        bm[b], bpc[b] = macro_f1_from_cells(cells)
    return m0, pc0, bm, bpc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Authoritative model list + point estimates from the canonical comparison CSV.
    canon = pd.read_csv(os.path.join(RESULTS_DIR, "f1_macro_train_vs_oos.csv"))
    model_names = list(canon["Model"])

    # Ground truth only needed for the CSV-fallback models.
    gt_s = None
    if any(n in CSV_FALLBACK for n in model_names):
        from compare_predictions import load_ground_truth
        gt = load_ground_truth()
        gt_s = gt.drop_duplicates("fid").set_index("fid")["true_label"]

    rng = np.random.default_rng(SEED)
    point_macro, boot_macro = {}, {}
    point_pc, boot_pc, classes_of = {}, {}, {}

    for name in model_names:
        cmf = cm_filename(name)
        if os.path.exists(cmf):
            yt, yp, classes = outcomes_from_cm(cmf)
        elif name in CSV_FALLBACK and os.path.exists(CSV_FALLBACK[name]) and gt_s is not None:
            yt, yp, classes = outcomes_from_csv(CSV_FALLBACK[name], gt_s)
        else:
            print(f"  (skip, no source) {name}")
            continue
        m0, pc0, bm, bpc = bootstrap_model(yt, yp, classes, rng)
        point_macro[name] = m0
        boot_macro[name] = bm
        point_pc[name] = pc0
        boot_pc[name] = bpc
        classes_of[name] = classes
        print(f"  {name:30s} F1m={m0:.4f}  CI[{np.percentile(bm,2.5):.4f},{np.percentile(bm,97.5):.4f}]  n={len(yt)}")

    # ---- Output 1: per-model macro-F1 CI ----
    rows = []
    for name in boot_macro:
        bm = boot_macro[name]
        rows.append({
            "Model": name,
            "OOS_F1_macro": round(point_macro[name], 4),
            "F1_mean": round(bm.mean(), 4),
            "F1_lo": round(np.percentile(bm, 2.5), 4),
            "F1_hi": round(np.percentile(bm, 97.5), 4),
            "se": round(bm.std(ddof=1), 4),
        })
    df_ci = pd.DataFrame(rows).sort_values("OOS_F1_macro", ascending=False)
    df_ci.to_csv(os.path.join(RESULTS_DIR, "bootstrap_ci.csv"), index=False)
    print("\n=== bootstrap_ci.csv ===")
    print(df_ci.to_string(index=False))

    # ---- Output 2: per-class CI ----
    pc_rows = []
    for name in boot_pc:
        for i, c in enumerate(classes_of[name]):
            col = boot_pc[name][:, i]
            pc_rows.append({
                "Model": name, "class": c,
                "f1_point": round(point_pc[name][i], 4),
                "f1_lo": round(np.percentile(col, 2.5), 4),
                "f1_hi": round(np.percentile(col, 97.5), 4),
            })
    pd.DataFrame(pc_rows).to_csv(
        os.path.join(RESULTS_DIR, "bootstrap_per_class_ci.csv"), index=False)
    print("\nwrote bootstrap_per_class_ci.csv")

    # ---- Output 3: pairwise Delta gaps (unpaired / independent bootstrap) ----
    gap_rows = []
    for a, c in PAIRWISE:
        if a not in boot_macro or c not in boot_macro:
            print(f"  (skip pair, missing) {a} vs {c}")
            continue
        d = boot_macro[a] - boot_macro[c]   # independent draws -> conservative
        point = point_macro[a] - point_macro[c]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
        gap_rows.append({
            "Model A": a, "Model B": c,
            "Delta": round(point, 4),
            "Delta_lo": round(lo, 4), "Delta_hi": round(hi, 4),
            "p_two_sided": round(p, 4),
            "significant_95": bool(lo > 0 or hi < 0),
        })
    df_gap = pd.DataFrame(gap_rows)
    df_gap.to_csv(os.path.join(RESULTS_DIR, "bootstrap_pairwise_gaps.csv"), index=False)
    print("\n=== bootstrap_pairwise_gaps.csv (unpaired, conservative) ===")
    print(df_gap.to_string(index=False))


if __name__ == "__main__":
    main()
