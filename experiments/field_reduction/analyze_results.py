"""Score OOS predictions and produce comparison CSV + line plot.

Loads ground truth from TEST_LABELS_GEOJSON, scores each prediction CSV,
pulls baseline (fraction=1.0) from the existing model_comparison.csv,
and outputs a consolidated results table and plot.

Usage:
    python analyze_results.py
"""

import json
import os
import sys
from collections import Counter

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from experiment_config import (
    FRACTIONS,
    MODELS_DIR,
    RESULTS_DIR,
    PREDICTIONS_DIR,
    TEST_LABELS_GEOJSON,
    OOS_COMPARISON_CSV,
    BASELINE_NAME_MAP,
    DISPLAY_NAME_MAP,
    MODEL_TRAINING_LEVEL,
    MODEL_FEATURE_TYPE,
    ML_ENSEMBLE_MODELS,
)

sys.stdout.reconfigure(line_buffering=True)

ALL_MODELS = list(DISPLAY_NAME_MAP.keys())


def load_ground_truth():
    """Load ground truth field labels from geojson."""
    print(f"Loading ground truth: {TEST_LABELS_GEOJSON}")
    gdf = gpd.read_file(TEST_LABELS_GEOJSON)
    gt = gdf[["fid", "crop_name"]].copy()
    gt["fid"] = gt["fid"].astype(int)
    print(f"  Ground truth fields: {len(gt)}")
    return gt


def score_predictions(pred_csv, gt):
    """Score a prediction CSV against ground truth."""
    pred = pd.read_csv(pred_csv)
    merged = gt.merge(pred, on="fid", suffixes=("_true", "_pred"))
    if len(merged) == 0:
        return None
    y_true = merged["crop_name_true"]
    y_pred = merged["crop_name_pred"]
    return {
        "fields": len(merged),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def score_df(pred_df, gt):
    """Score a prediction DataFrame against ground truth."""
    merged = gt.merge(pred_df, on="fid", suffixes=("_true", "_pred"))
    if len(merged) == 0:
        return None
    y_true = merged["crop_name_true"]
    y_pred = merged["crop_name_pred"]
    return {
        "fields": len(merged),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def majority_vote_ensemble(model_names, frac):
    """Build majority vote from individual model prediction CSVs."""
    frac_str = f"{frac:.2f}"
    dfs = []
    for m in model_names:
        csv_path = os.path.join(PREDICTIONS_DIR, f"{m}_frac_{frac_str}.csv")
        if not os.path.exists(csv_path):
            return None
        dfs.append(pd.read_csv(csv_path))

    # Merge all predictions on fid
    merged = dfs[0][["fid"]].copy()
    for i, df in enumerate(dfs):
        merged = merged.merge(df, on="fid", suffixes=("", f"_{i}"))

    # Get all crop_name columns
    crop_cols = [c for c in merged.columns if c.startswith("crop_name")]

    def _vote(row):
        votes = [row[c] for c in crop_cols]
        return Counter(votes).most_common(1)[0][0]

    merged["crop_name"] = merged.apply(_vote, axis=1)
    return merged[["fid", "crop_name"]]


def load_baselines():
    """Load fraction=1.0 baselines from existing OOS model_comparison.csv."""
    if not os.path.exists(OOS_COMPARISON_CSV):
        print(f"  Warning: baseline CSV not found: {OOS_COMPARISON_CSV}")
        return {}

    df = pd.read_csv(OOS_COMPARISON_CSV)
    baselines = {}
    for exp_name, csv_name in BASELINE_NAME_MAP.items():
        row = df[df["Model"] == csv_name]
        if len(row) == 0:
            print(f"  Warning: baseline not found for '{csv_name}'")
            continue
        row = row.iloc[0]
        baselines[exp_name] = {
            "fields": int(row.get("Fields", 0)),
            "accuracy": row.get("Accuracy", np.nan),
            "f1_macro": row.get("F1 (macro)", np.nan),
            "f1_weighted": row.get("F1 (weighted)", np.nan),
            "cohen_kappa": row.get("Cohen Kappa", np.nan),
        }
    return baselines


def load_train_metadata(model_name, frac):
    """Load training metadata.json for a model/fraction."""
    frac_str = f"{frac:.2f}"
    meta_path = os.path.join(MODELS_DIR, model_name, f"frac_{frac_str}", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Field Reduction Analysis ===")

    gt = load_ground_truth()
    baselines = load_baselines()

    rows = []

    # Score each fraction prediction
    for model_name in ALL_MODELS:
        display = DISPLAY_NAME_MAP[model_name]

        training_level = MODEL_TRAINING_LEVEL.get(model_name, "")
        feature_type = MODEL_FEATURE_TYPE.get(model_name, "")

        # Baseline (fraction=1.0)
        if model_name in baselines:
            bl = baselines[model_name]
            rows.append({
                "Model": display,
                "Training Level": training_level,
                "Feature Type": feature_type,
                "Fraction": 1.00,
                "Train Fields": "",
                "Train F1 (macro)": "",
                "OOS F1 (macro)": bl["f1_macro"],
                "OOS Accuracy": bl["accuracy"],
                "OOS Cohen Kappa": bl["cohen_kappa"],
                "Delta vs 1.0": 0.0,
            })

        baseline_f1 = baselines.get(model_name, {}).get("f1_macro", np.nan)

        for frac in FRACTIONS:
            frac_str = f"{frac:.2f}"
            pred_csv = os.path.join(PREDICTIONS_DIR, f"{model_name}_frac_{frac_str}.csv")

            if not os.path.exists(pred_csv):
                print(f"  [SKIP] Missing: {pred_csv}")
                continue

            metrics = score_predictions(pred_csv, gt)
            if metrics is None:
                print(f"  [SKIP] No matching fields for {model_name} frac={frac_str}")
                continue

            meta = load_train_metadata(model_name, frac)
            train_fields = meta.get("train_fields", "") if meta else ""
            train_f1 = ""
            if meta and "metrics" in meta:
                train_f1 = meta["metrics"].get("f1_macro", "")

            delta = metrics["f1_macro"] - baseline_f1 if not np.isnan(baseline_f1) else np.nan

            rows.append({
                "Model": display,
                "Training Level": training_level,
                "Feature Type": feature_type,
                "Fraction": frac,
                "Train Fields": train_fields,
                "Train F1 (macro)": round(train_f1, 4) if isinstance(train_f1, float) else train_f1,
                "OOS F1 (macro)": round(metrics["f1_macro"], 4),
                "OOS Accuracy": round(metrics["accuracy"], 4),
                "OOS Cohen Kappa": round(metrics["cohen_kappa"], 4),
                "Delta vs 1.0": round(delta, 4) if not np.isnan(delta) else "",
            })

            print(f"  {display} frac={frac_str}: OOS F1 macro={metrics['f1_macro']:.4f}, delta={delta:+.4f}" if not np.isnan(delta) else f"  {display} frac={frac_str}: OOS F1 macro={metrics['f1_macro']:.4f}")

    # Classical ML ensemble (majority vote)
    ensemble_display = "Ensemble ML (majority vote)"
    ensemble_baseline_f1 = baselines.get("ensemble_ml", {}).get("f1_macro", np.nan)

    # Baseline (fraction=1.0) from existing model_comparison.csv
    if "ensemble_ml" in baselines:
        bl = baselines["ensemble_ml"]
        rows.append({
            "Model": ensemble_display,
            "Training Level": "majority vote (ML models)",
            "Feature Type": "majority vote",
            "Fraction": 1.00,
            "Train Fields": "",
            "Train F1 (macro)": "",
            "OOS F1 (macro)": bl["f1_macro"],
            "OOS Accuracy": bl["accuracy"],
            "OOS Cohen Kappa": bl["cohen_kappa"],
            "Delta vs 1.0": 0.0,
        })

    for frac in FRACTIONS:
        frac_str = f"{frac:.2f}"
        ensemble_pred = majority_vote_ensemble(ML_ENSEMBLE_MODELS, frac)
        if ensemble_pred is None:
            print(f"  [SKIP] Missing predictions for ML ensemble frac={frac_str}")
            continue

        metrics = score_df(ensemble_pred, gt)
        if metrics is None:
            continue

        delta = metrics["f1_macro"] - ensemble_baseline_f1 if not np.isnan(ensemble_baseline_f1) else np.nan

        rows.append({
            "Model": ensemble_display,
            "Training Level": "majority vote (ML models)",
            "Feature Type": "majority vote",
            "Fraction": frac,
            "Train Fields": "",
            "Train F1 (macro)": "",
            "OOS F1 (macro)": round(metrics["f1_macro"], 4),
            "OOS Accuracy": round(metrics["accuracy"], 4),
            "OOS Cohen Kappa": round(metrics["cohen_kappa"], 4),
            "Delta vs 1.0": round(delta, 4) if not np.isnan(delta) else "",
        })

        print(f"  {ensemble_display} frac={frac_str}: OOS F1 macro={metrics['f1_macro']:.4f}, delta={delta:+.4f}" if not np.isnan(delta) else f"  {ensemble_display} frac={frac_str}: OOS F1 macro={metrics['f1_macro']:.4f}")

    if not rows:
        print("No results found. Run training and prediction first.")
        return

    # Save CSV
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "field_reduction_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(results_df.to_string(index=False))

    # Plot: fraction vs OOS F1 macro
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_models = results_df["Model"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_models)))

    for i, display in enumerate(plot_models):
        model_rows = results_df[results_df["Model"] == display].copy()
        if len(model_rows) == 0:
            continue
        model_rows = model_rows.sort_values("Fraction")
        fracs = model_rows["Fraction"].values
        f1s = pd.to_numeric(model_rows["OOS F1 (macro)"], errors="coerce").values
        style = '--' if "Ensemble" in display else '-'
        ax.plot(fracs, f1s, marker='o', label=display, color=colors[i],
                linewidth=2.5 if "Ensemble" in display else 2, linestyle=style)

    ax.set_xlabel("Fraction of Training Fields", fontsize=12)
    ax.set_ylabel("OOS F1 Macro", fontsize=12)
    ax.set_title("Field Reduction Experiment: OOS F1 Macro vs Training Data Fraction", fontsize=13)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "field_reduction_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
