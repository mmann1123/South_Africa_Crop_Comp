"""
Base ML models (pixel-level) inference on holdout test data (34S_20E_259N).

Loads saved LR, RF, LightGBM, XGBoost models trained at pixel level
on xr_fresh features. Predicts on field-level aggregated xr_fresh test
features (combined_test_features.parquet).

Note: Models were trained at pixel level, but test inference uses
field-level mean features since the test xr_fresh parquets have a B12
pixel grid mismatch that prevents clean pixel-level merging.

Input: combined_test_features.parquet (field-level xr_fresh features)
Output: predictions_base_lr.csv, predictions_base_rf.csv,
        predictions_base_lgbm.csv, predictions_base_xgb.csv
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from joblib import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import COMBINED_TEST_FEATURES_PATH, MODEL_DIR

# Input - field-level xr_fresh features (same feature space as training)
TEST_PARQUET = COMBINED_TEST_FEATURES_PATH

# Base ML model directory (timestamped subfolder under models/ml_base/)
BASE_ML_DIR = os.path.join(MODEL_DIR, "ml_base")

# Model name -> joblib filename
MODEL_FILES = {
    "base_lr": "logistic_regression.joblib",
    "base_rf": "random_forest.joblib",
    "base_lgbm": "lightgbm.joblib",
    "base_xgb": "xgboost.joblib",
}


def find_latest_run():
    """Find the most recent timestamped run directory."""
    if not os.path.isdir(BASE_ML_DIR):
        return None
    runs = sorted(glob.glob(os.path.join(BASE_ML_DIR, "*")))
    runs = [r for r in runs if os.path.isdir(r)]
    return runs[-1] if runs else None


def main():
    print("=== Base ML Models (Pixel-Level) Inference ===")

    if not os.path.exists(TEST_PARQUET):
        raise FileNotFoundError(f"Test parquet not found: {TEST_PARQUET}")

    run_dir = find_latest_run()
    if run_dir is None:
        raise FileNotFoundError(f"No ml_base run directories found in {BASE_ML_DIR}")
    print(f"Using model run: {os.path.basename(run_dir)}")

    # Load preprocessing artifacts
    scaler = load(os.path.join(run_dir, "scaler.joblib"))
    label_encoder = load(os.path.join(run_dir, "label_encoder.joblib"))
    feature_columns = load(os.path.join(run_dir, "feature_columns.joblib"))
    print(f"Classes: {label_encoder.classes_}")
    print(f"Features: {len(feature_columns)}")

    # Load test data
    print(f"\nLoading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    df = df.drop(columns=["May"], errors="ignore")
    print(f"Shape: {df.shape}, Fields: {df['fid'].nunique()}")

    # Prepare features (match training columns)
    exclude_cols = {"id", "point", "fid", "crop_id", "crop_name", "SHAPE_AREA", "SHAPE_LEN"}
    available = [c for c in feature_columns if c in df.columns]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} training features missing in test data, filling with 0")
        for col in missing:
            df[col] = 0.0

    X = df[feature_columns].copy()
    # Match training preprocessing: drop cols that are all NaN, fill remaining
    X = X.fillna(0)
    fids = df["fid"].values

    # Scale
    X_scaled = scaler.transform(X).astype(np.float32)

    # Run each model
    for short_name, model_file in MODEL_FILES.items():
        model_path = os.path.join(run_dir, model_file)
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {short_name}: {model_file} not found")
            continue

        print(f"\n--- {short_name} ---")
        model = load(model_path)

        # Field-level prediction (data is already aggregated to field means)
        print("  Predicting (field-level)...")
        y_pred_codes = model.predict(X_scaled)
        y_pred_labels = label_encoder.inverse_transform(y_pred_codes)

        field_df = pd.DataFrame({"fid": fids, "crop_name": y_pred_labels})

        output_csv = os.path.join(SCRIPT_DIR, f"predictions_{short_name}.csv")
        field_df.to_csv(output_csv, index=False)
        print(f"  Saved: {output_csv} ({len(field_df)} fields)")
        print(f"  {field_df['crop_name'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
