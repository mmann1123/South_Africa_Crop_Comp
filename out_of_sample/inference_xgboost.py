"""
XGBoost (Optuna-tuned) inference on holdout test data (34S_20E_259N).

Input: combined_test_features.parquet (field-level)
Output: predictions_xgboost.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import COMBINED_TEST_FEATURES_PATH, XGB_TUNER_DIR

# Input
TEST_PARQUET = COMBINED_TEST_FEATURES_PATH

# Model artifacts
MODEL_PATH = os.path.join(XGB_TUNER_DIR, "final_xgb_model.joblib")
IMPUTER_PATH = os.path.join(XGB_TUNER_DIR, "imputer.joblib")
SCALER_PATH = os.path.join(XGB_TUNER_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(XGB_TUNER_DIR, "label_encoder.joblib")

# Output
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_xgboost.csv")


def main():
    print("=== XGBoost (Optuna) Inference ===")

    for path, name in [
        (TEST_PARQUET, "Test parquet"),
        (MODEL_PATH, "XGBoost model"),
        (IMPUTER_PATH, "Imputer"),
        (SCALER_PATH, "Scaler"),
        (LABEL_ENCODER_PATH, "Label encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Load test data
    print(f"\nLoading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape: {df.shape}, Fields: {df['fid'].nunique()}")

    fids = df["fid"].to_numpy()
    X = df.drop(columns=["fid", "crop_name"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Load preprocessing + model
    print("Loading model artifacts...")
    imputer = load(IMPUTER_PATH)
    scaler = load(SCALER_PATH)
    le = load(LABEL_ENCODER_PATH)
    model = load(MODEL_PATH)
    print(f"Classes: {le.classes_}")

    # Align columns to what the imputer saw during training
    train_cols = imputer.feature_names_in_ if hasattr(imputer, "feature_names_in_") else None
    if train_cols is not None:
        missing = set(train_cols) - set(X.columns)
        extra = set(X.columns) - set(train_cols)
        if missing:
            print(f"  Adding {len(missing)} missing columns (zeros)")
            for col in missing:
                X[col] = 0.0
        if extra:
            print(f"  Dropping {len(extra)} extra columns")
        X = X[train_cols]

    print(f"Feature columns: {X.shape[1]}")

    # Preprocess
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # Predict
    print("Running XGBoost prediction...")
    codes = model.predict(X_scaled)
    labels = le.inverse_transform(codes)

    # Save
    result = pd.DataFrame({"fid": fids, "crop_name": labels})
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"\n{result['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()
