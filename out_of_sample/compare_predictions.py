"""
Compare predictions across all models.

Reads prediction CSVs from each model and:
1. Shows prediction distribution per model
2. Computes pairwise agreement between models
3. Identifies fields where models disagree
4. Optionally copies a selected model's predictions to submissions/

Input: predictions_*.csv files
Output: Summary comparison + optional submission file
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix, log_loss,
)
from sklearn.preprocessing import label_binarize

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "deep_learn", "src"))
from config import TEST_LABELS_DIR, TEST_REGION, REPORTS_DIR

# Ground truth labels
TEST_LABELS_GEOJSON = os.path.join(
    TEST_LABELS_DIR,
    f"ref_fusion_competition_south_africa_test_labels_{TEST_REGION}",
    "labels.geojson",
)

# Map comparison display name -> feature type used for training
FEATURE_TYPE_MAP = {
    "XGBoost (field)": "xr_fresh time-series",
    "SMOTE Stacked (field)": "xr_fresh time-series",
    "Voting (field)": "xr_fresh time-series",
    "Stacking (field)": "xr_fresh time-series",
    "Base LR (pixel)": "raw pixel (band x month)",
    "Base RF (pixel)": "raw pixel (band x month)",
    "Base LightGBM (pixel)": "raw pixel (band x month)",
    "Base XGBoost (pixel)": "raw pixel (band x month)",
    "CNN-BiLSTM (pixel)": "raw pixel (band x month)",
    "TabNet (pixel)": "raw pixel (band x month)",
    "3D CNN (patch)": "raw pixel (spatial-temporal patch)",
    "Multi-Ch CNN (patch)": "raw pixel (spatial-temporal patch)",
    "Ensemble 3D CNN (patch)": "raw pixel (spatial-temporal patch)",
    "TabNet Field (field)": "xr_fresh time-series",
    "L-TAE (pixel)": "raw pixel (temporal sequence)",
    "TempCNN (pixel)": "raw pixel (temporal sequence)",
    "L-TAE Field (field)": "raw temporal (field-averaged)",
    "TempCNN Field (field)": "raw temporal (field-averaged)",
    "TabNet Temporal Field (field)": "raw temporal (field-averaged)",
    "LightGBM (field)": "xr_fresh time-series",
}

# Map comparison display name -> report model_name (for dynamic train_count lookup)
REPORT_NAME_MAP = {
    "XGBoost (field)": "XGBoost Field-Level",
    "SMOTE Stacked (field)": "SMOTE Stacked Ensemble",
    "Voting (field)": "Voting Ensemble",
    "Stacking (field)": "Stacking Ensemble",
    "Base LR (pixel)": "Logistic Regression (Pixel-Level)",
    "Base RF (pixel)": "Random Forest (Pixel-Level)",
    "Base LightGBM (pixel)": "LightGBM (Pixel-Level)",
    "Base XGBoost (pixel)": "XGBoost (Pixel-Level)",
    "CNN-BiLSTM (pixel)": "CNN-BiLSTM Ensemble (5-seed)",
    "TabNet (pixel)": "TabTransformer Ensemble (Field-Level)",
    "3D CNN (patch)": "3D CNN Patch-Level",
    "Multi-Ch CNN (patch)": "Multi-Channel CNN Patch-Level",
    "Ensemble 3D CNN (patch)": "Ensemble 3D CNN Patch-Level",
    "TabNet Field (field)": "TabNet Field-Level (xr_fresh)",
    "L-TAE (pixel)": "L-TAE Temporal Attention",
    "TempCNN (pixel)": "TempCNN Temporal Conv",
    "L-TAE Field (field)": "L-TAE Field-Level (temporal)",
    "TempCNN Field (field)": "TempCNN Field-Level (temporal)",
    "TabNet Temporal Field (field)": "TabNet Field-Level (temporal)",
    "LightGBM (field)": "LightGBM Field-Level",
}


def _load_report_metadata():
    """Scan reports/ for metadata.json and return {model_name: {train_count, training_time_seconds}} from latest report."""
    info = {}
    if not os.path.isdir(REPORTS_DIR):
        return info
    for entry in sorted(os.listdir(REPORTS_DIR)):
        meta_path = os.path.join(REPORTS_DIR, entry, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                name = meta.get("model_name", "")
                if not name:
                    continue
                tc = meta.get("split_info", {}).get("train_count")
                tt = meta.get("training_time_seconds")
                f1m = meta.get("metrics", {}).get("f1_macro")
                info[name] = {"train_count": tc, "training_time_seconds": tt, "f1_macro": f1m}
            except (json.JSONDecodeError, KeyError):
                continue
    return info


def _get_report_field(display_name, report_info, field):
    """Look up a field from report metadata, return None if not found."""
    report_name = REPORT_NAME_MAP.get(display_name)
    if report_name and report_name in report_info:
        return report_info[report_name].get(field)
    return None


# Prediction files to compare: (path, training_level)
# Training obs is looked up dynamically from reports/ metadata.json
_PREDICTION_FILES_BASE = {
    "XGBoost (field)": (os.path.join(SCRIPT_DIR, "predictions_xgboost.csv"), "field"),
    "SMOTE Stacked (field)": (os.path.join(SCRIPT_DIR, "predictions_smote_stacked.csv"), "field"),
    "Voting (field)": (os.path.join(SCRIPT_DIR, "predictions_voting.csv"), "field"),
    "Stacking (field)": (os.path.join(SCRIPT_DIR, "predictions_stacking.csv"), "field"),
    "Base LR (pixel)": (os.path.join(SCRIPT_DIR, "predictions_base_lr.csv"), "pixel"),
    "Base RF (pixel)": (os.path.join(SCRIPT_DIR, "predictions_base_rf.csv"), "pixel"),
    "Base LightGBM (pixel)": (os.path.join(SCRIPT_DIR, "predictions_base_lgbm.csv"), "pixel"),
    "Base XGBoost (pixel)": (os.path.join(SCRIPT_DIR, "predictions_base_xgb.csv"), "pixel"),
    "CNN-BiLSTM (pixel)": (os.path.join(SCRIPT_DIR, "predictions_cnn_bilstm.csv"), "pixel"),
    "TabNet (pixel)": (os.path.join(SCRIPT_DIR, "predictions_tabnet.csv"), "pixel"),
    "3D CNN (patch)": (os.path.join(SCRIPT_DIR, "predictions_3d_cnn.csv"), "patch"),
    "Multi-Ch CNN (patch)": (os.path.join(SCRIPT_DIR, "predictions_multi_channel_cnn.csv"), "patch"),
    "Ensemble 3D CNN (patch)": (os.path.join(SCRIPT_DIR, "predictions_ensemble_3d_cnn.csv"), "patch"),
    "TabNet Field (field)": (os.path.join(SCRIPT_DIR, "predictions_tabnet_field.csv"), "field"),
    "L-TAE (pixel)": (os.path.join(SCRIPT_DIR, "predictions_ltae.csv"), "pixel"),
    "TempCNN (pixel)": (os.path.join(SCRIPT_DIR, "predictions_tempcnn.csv"), "pixel"),
    "L-TAE Field (field)": (os.path.join(SCRIPT_DIR, "predictions_ltae_field.csv"), "field"),
    "TempCNN Field (field)": (os.path.join(SCRIPT_DIR, "predictions_tempcnn_field.csv"), "field"),
    "TabNet Temporal Field (field)": (os.path.join(SCRIPT_DIR, "predictions_tabnet_temporal_field.csv"), "field"),
    "LightGBM (field)": (os.path.join(SCRIPT_DIR, "predictions_lgbm.csv"), "field"),
}

# Build PREDICTION_FILES with dynamic train_obs from reports
_report_info = _load_report_metadata()
PREDICTION_FILES = {}
for _name, (_path, _level) in _PREDICTION_FILES_BASE.items():
    _train_obs = _get_report_field(_name, _report_info, "train_count")
    PREDICTION_FILES[_name] = (_path, _level, _train_obs)

# Output
SUBMISSION_PATH = os.path.join(REPO_ROOT, "submissions", "prediction.csv")


def load_predictions():
    """Load all available prediction files."""
    predictions = {}

    for name, (path, level, train_obs) in PREDICTION_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            predictions[name] = df
            print(f"Loaded {name}: {len(df)} predictions")
        else:
            print(f"Not found: {name} ({path})")

    return predictions


def show_distributions(predictions):
    """Show prediction distribution for each model."""
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTIONS")
    print("=" * 60)

    for name, df in predictions.items():
        print(f"\n{name}:")
        print(df["crop_name"].value_counts().to_string())


def compute_agreement(predictions):
    """Compute pairwise agreement between models."""
    if len(predictions) < 2:
        print("\nNeed at least 2 models to compute agreement")
        return

    print("\n" + "=" * 60)
    print("PAIRWISE AGREEMENT")
    print("=" * 60)

    names = list(predictions.keys())

    # Align predictions by fid
    merged = None
    for name, df in predictions.items():
        df = df.rename(columns={"crop_name": name})
        if merged is None:
            merged = df[["fid", name]]
        else:
            merged = merged.merge(df[["fid", name]], on="fid", how="inner")

    print(f"\nFields with predictions from all models: {len(merged)}")

    # Compute pairwise agreement
    print("\nAgreement matrix (%):")
    header = "            " + "  ".join(f"{n[:10]:>10}" for n in names)
    print(header)

    for name1 in names:
        row = f"{name1[:10]:<10}"
        for name2 in names:
            if name1 == name2:
                row += f"{'100.0':>12}"
            else:
                agreement = (merged[name1] == merged[name2]).mean() * 100
                row += f"{agreement:>12.1f}"
        print(row)

    return merged


def find_disagreements(merged, predictions):
    """Find fields where models disagree."""
    if merged is None or len(predictions) < 2:
        return

    print("\n" + "=" * 60)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 60)

    names = list(predictions.keys())
    pred_cols = [n for n in names if n in merged.columns]

    # Find rows where not all predictions match
    def all_same(row):
        vals = [row[c] for c in pred_cols]
        return len(set(vals)) == 1

    merged["all_agree"] = merged.apply(all_same, axis=1)

    n_agree = merged["all_agree"].sum()
    n_disagree = len(merged) - n_agree

    print(f"\nAll models agree: {n_agree} fields ({n_agree/len(merged)*100:.1f}%)")
    print(f"At least one disagrees: {n_disagree} fields ({n_disagree/len(merged)*100:.1f}%)")

    if n_disagree > 0:
        print("\nSample disagreements:")
        disagree_sample = merged[~merged["all_agree"]].head(10)
        print(disagree_sample[["fid"] + pred_cols].to_string(index=False))


# Model groupings for subset majority votes
ML_FIELD_MODELS = {
    "XGBoost (field)", "SMOTE Stacked (field)", "Voting (field)", "Stacking (field)",
    "TabNet Field (field)", "LightGBM (field)",
}
ML_PIXEL_MODELS = {
    "Base LR (pixel)", "Base RF (pixel)", "Base LightGBM (pixel)", "Base XGBoost (pixel)",
}
ML_MODELS = ML_FIELD_MODELS | ML_PIXEL_MODELS
DL_MODELS = {
    "CNN-BiLSTM (pixel)", "TabNet (pixel)",
    "3D CNN (patch)", "Multi-Ch CNN (patch)", "Ensemble 3D CNN (patch)",
    "L-TAE (pixel)", "TempCNN (pixel)",
    "L-TAE Field (field)", "TempCNN Field (field)",
}


def _majority_vote_subset(predictions, subset_names, label):
    """Create majority vote from a subset of models. Returns DataFrame or None."""
    subset = {k: v for k, v in predictions.items() if k in subset_names}
    if len(subset) < 2:
        return None

    merged = None
    names = list(subset.keys())
    for name, df in subset.items():
        df = df.rename(columns={"crop_name": name})
        if merged is None:
            merged = df[["fid", name]]
        else:
            merged = merged.merge(df[["fid", name]], on="fid", how="inner")

    pred_cols = [n for n in names if n in merged.columns]

    def majority_vote(row):
        votes = [row[c] for c in pred_cols]
        return Counter(votes).most_common(1)[0][0]

    merged["vote"] = merged.apply(majority_vote, axis=1)

    print(f"\n{label} ({len(names)} models: {', '.join(names)}):")
    print(merged["vote"].value_counts().to_string())

    return merged[["fid", "vote"]].rename(columns={"vote": "crop_name"})


def ensemble_vote(predictions):
    """Create ensemble predictions via majority vote: all, ML-only, DL-only."""
    print("\n" + "=" * 60)
    print("ENSEMBLE MAJORITY VOTES")
    print("=" * 60)

    results = {}

    # All models
    all_vote = _majority_vote_subset(predictions, set(predictions.keys()), "All Models")
    if all_vote is not None:
        results["Ensemble (all)"] = all_vote

    # ML only (field + pixel)
    ml_vote = _majority_vote_subset(predictions, ML_MODELS, "ML Models (all)")
    if ml_vote is not None:
        results["Ensemble (ML)"] = ml_vote

    # ML field-level only
    ml_field_vote = _majority_vote_subset(predictions, ML_FIELD_MODELS, "ML Models (field)")
    if ml_field_vote is not None:
        results["Ensemble (ML field)"] = ml_field_vote

    # ML pixel-level only
    ml_pixel_vote = _majority_vote_subset(predictions, ML_PIXEL_MODELS, "ML Models (pixel)")
    if ml_pixel_vote is not None:
        results["Ensemble (ML pixel)"] = ml_pixel_vote

    # DL only
    dl_vote = _majority_vote_subset(predictions, DL_MODELS, "DL Models")
    if dl_vote is not None:
        results["Ensemble (DL)"] = dl_vote

    return results


def load_ground_truth():
    """Load ground truth labels for the test region."""
    if not os.path.exists(TEST_LABELS_GEOJSON):
        print(f"\nGround truth not found: {TEST_LABELS_GEOJSON}")
        return None

    import geopandas as gpd
    gdf = gpd.read_file(TEST_LABELS_GEOJSON)
    gt = gdf[["fid", "crop_name"]].copy()
    gt = gt.rename(columns={"crop_name": "true_label"})
    print(f"\nGround truth loaded: {len(gt)} fields")
    return gt


def score_predictions(predictions, ground_truth):
    """Score each model's predictions against ground truth and save CSVs."""
    if ground_truth is None:
        return

    print("\n" + "=" * 60)
    print("SCORING AGAINST GROUND TRUTH")
    print("=" * 60)

    results_dir = os.path.join(REPO_ROOT, "out_of_sample", "scoring_results")
    os.makedirs(results_dir, exist_ok=True)

    results = []
    for name, df in predictions.items():
        merged = df.merge(ground_truth, on="fid", how="inner")
        if len(merged) == 0:
            print(f"\n{name}: No matching FIDs with ground truth")
            continue

        y_true = merged["true_label"]
        y_pred = merged["crop_name"]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)

        # Binary cross entropy per crop (one-vs-rest)
        labels = sorted(y_true.unique())
        y_true_bin = label_binarize(y_true, classes=labels)
        y_pred_bin = label_binarize(y_pred, classes=labels)
        # Clip hard predictions to avoid log(0)
        eps = 1e-7
        y_pred_prob = np.clip(y_pred_bin.astype(float), eps, 1 - eps)
        mean_ce = log_loss(y_true_bin, y_pred_prob)

        per_crop_ce = {}
        for i, crop in enumerate(labels):
            ce_i = log_loss(y_true_bin[:, i], y_pred_prob[:, i])
            per_crop_ce[crop] = ce_i

        # Look up training level, obs, feature type, and training time from PREDICTION_FILES
        info = PREDICTION_FILES.get(name, (None, "majority vote (all models)", None))
        level = info[1]
        train_obs = info[2] if len(info) > 2 else None
        feature_type = FEATURE_TYPE_MAP.get(name, "majority vote")
        train_time_s = _get_report_field(name, _report_info, "training_time_seconds")
        train_time_min = round(train_time_s / 60, 1) if train_time_s else None
        row = {"Model": name, "Training Level": level, "Feature Type": feature_type,
               "Training Obs": train_obs, "Training Time (min)": train_time_min,
               "Accuracy": acc, "F1 (weighted)": f1, "F1 (macro)": f1_macro,
               "Cohen Kappa": kappa, "Cross Entropy": mean_ce, "Fields": len(merged)}
        results.append(row)

        print(f"\n--- {name} ({len(merged)} fields) ---")
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"  F1 (macro):    {f1_macro:.4f}")
        print(f"  Cohen Kappa:   {kappa:.4f}")
        print(f"  Cross Entropy: {mean_ce:.4f}")
        print(f"  Per-crop CE:   {', '.join(f'{c}: {v:.4f}' for c, v in per_crop_ce.items())}")
        print(f"\n  Classification Report:")
        report_text = classification_report(y_true, y_pred)
        print(report_text)

        # Save per-class metrics CSV (includes cross entropy)
        safe_name = name.replace(" ", "_").replace("-", "_")
        per_class = pd.DataFrame(
            classification_report(y_true, y_pred, output_dict=True)
        ).T
        per_class.index.name = "class"
        # Add cross entropy column for per-crop rows
        per_class["cross_entropy"] = per_class.index.map(
            lambda c: per_crop_ce.get(c, np.nan)
        )
        per_class.to_csv(os.path.join(results_dir, f"per_class_{safe_name}.csv"))

        # Save confusion matrix CSV
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = "true_label"
        cm_df.to_csv(os.path.join(results_dir, f"confusion_matrix_{safe_name}.csv"))

    if results:
        print("\n" + "=" * 60)
        print("SCORE SUMMARY")
        print("=" * 60)
        summary = pd.DataFrame(results).sort_values("Cohen Kappa", ascending=False)
        print(summary.to_string(index=False, float_format="%.4f"))

        # Save combined summary CSV
        summary.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

        # Save split CSVs by training level
        field_levels = {"field"}
        pixel_levels = {"pixel", "patch"}
        field_rows = summary[summary["Training Level"].isin(field_levels)]
        pixel_rows = summary[summary["Training Level"].isin(pixel_levels)]

        if not field_rows.empty:
            field_rows.to_csv(os.path.join(results_dir, "model_comparison_field.csv"), index=False)
        if not pixel_rows.empty:
            pixel_rows.to_csv(os.path.join(results_dir, "model_comparison_pixel.csv"), index=False)

        print(f"\nCSVs saved to {results_dir}/")
        print(f"  model_comparison.csv       ({len(summary)} models)")
        if not field_rows.empty:
            print(f"  model_comparison_field.csv ({len(field_rows)} field-level models)")
        if not pixel_rows.empty:
            print(f"  model_comparison_pixel.csv ({len(pixel_rows)} pixel/patch-level models)")

    return results


def generate_f1_macro_comparison(oos_results):
    """Generate CSV comparing training F1 macro vs OOS F1 macro for each model."""
    if not oos_results:
        print("\nNo OOS results to compare.")
        return

    results_dir = os.path.join(REPO_ROOT, "out_of_sample", "scoring_results")

    rows = []
    for r in oos_results:
        display_name = r["Model"]
        oos_f1_macro = r.get("F1 (macro)")
        train_f1_macro = _get_report_field(display_name, _report_info, "f1_macro")

        delta = None
        if train_f1_macro is not None and oos_f1_macro is not None:
            delta = oos_f1_macro - train_f1_macro

        rows.append({
            "Model": display_name,
            "Training Level": r.get("Training Level", ""),
            "Feature Type": r.get("Feature Type", ""),
            "Train F1 (macro)": train_f1_macro,
            "OOS F1 (macro)": oos_f1_macro,
            "Delta (OOS - Train)": delta,
        })

    df = pd.DataFrame(rows).sort_values("OOS F1 (macro)", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("TRAINING vs OUT-OF-SAMPLE F1 MACRO")
    print("=" * 60)
    print(df.to_string(index=False, float_format="%.4f"))

    missing_train = df[df["Train F1 (macro)"].isna()]["Model"].tolist()
    if missing_train:
        print(f"\n  Note: No training F1 macro for: {', '.join(missing_train)}")
        print("  (These are ensemble/majority-vote models with no single training report.)")

    csv_path = os.path.join(results_dir, "f1_macro_train_vs_oos.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return df


def create_submission(df, model_name):
    """Create submission file from predictions."""
    print(f"\n" + "=" * 60)
    print(f"CREATING SUBMISSION")
    print("=" * 60)

    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Sort by fid (competition requirement)
    df = df.sort_values("fid").reset_index(drop=True)

    # Save just the crop_name column (competition format)
    submission = df[["crop_name"]]
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Model: {model_name}")
    print(f"Fields: {len(df)}")
    print(f"Saved: {SUBMISSION_PATH}")


def main():
    print("=== Compare Model Predictions ===")

    # Load predictions
    predictions = load_predictions()

    if not predictions:
        print("\nNo prediction files found. Run inference scripts first.")
        return

    # Show distributions
    show_distributions(predictions)

    # Compute agreement
    merged = compute_agreement(predictions)

    # Find disagreements
    find_disagreements(merged, predictions)

    # Create ensemble votes (all, ML-only, DL-only)
    ensemble_dfs = {}
    if len(predictions) >= 2:
        ensemble_dfs = ensemble_vote(predictions)

    # Score against ground truth
    gt = load_ground_truth()
    all_to_score = dict(predictions)
    all_to_score.update(ensemble_dfs)
    oos_results = score_predictions(all_to_score, gt)

    # Generate training vs OOS F1 macro comparison
    generate_f1_macro_comparison(oos_results)

    # Prompt for submission
    print("\n" + "=" * 60)
    print("SELECT MODEL FOR SUBMISSION")
    print("=" * 60)

    available = list(predictions.keys()) + list(ensemble_dfs.keys())

    print("\nAvailable models:")
    for i, name in enumerate(available, 1):
        print(f"  {i}. {name}")

    # Default to first available model
    default_model = available[0]
    print(f"\nUsing default: {default_model}")
    print("(Edit this script to change the default or add command-line args)")

    # Create submission
    all_preds = dict(predictions)
    all_preds.update(ensemble_dfs)
    create_submission(all_preds[default_model], default_model)


if __name__ == "__main__":
    main()
