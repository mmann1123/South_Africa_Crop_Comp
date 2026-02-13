"""
TabNet field-level inference on holdout test data (34S_20E_259N).

Uses 5-seed TabNet ensemble trained on xr_fresh time-series features
(field-level averages from final_data.parquet).

Input: combined_test_features.parquet (field-level)
Output: predictions_tabnet_field.csv

Required artifacts in saved_models_tabnet_field/:
  - tabnet_field_seed_{42,101,202,303,404}.zip
  - tabnet_field_imputer.joblib
  - tabnet_field_scaler.joblib
  - tabnet_field_label_encoder.joblib
  - tabnet_field_feature_columns.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import COMBINED_TEST_FEATURES_PATH, TABNET_FIELD_DIR

TEST_PARQUET = COMBINED_TEST_FEATURES_PATH
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_tabnet_field.csv")
SEEDS = [42, 101, 202, 303, 404]


def main():
    print("=== TabNet Field-Level (xr_fresh) Inference ===")

    # Check artifacts exist
    if not os.path.exists(TABNET_FIELD_DIR):
        print(f"\nError: Model directory not found: {TABNET_FIELD_DIR}")
        print("Run tabnet_field.py first to train models.")
        return

    for artifact in [
        "tabnet_field_imputer.joblib",
        "tabnet_field_scaler.joblib",
        "tabnet_field_label_encoder.joblib",
        "tabnet_field_feature_columns.joblib",
    ]:
        path = os.path.join(TABNET_FIELD_DIR, artifact)
        if not os.path.exists(path):
            print(f"\nError: Missing artifact: {path}")
            return

    # Load preprocessing artifacts
    feature_columns = load(os.path.join(TABNET_FIELD_DIR, "tabnet_field_feature_columns.joblib"))
    imputer = load(os.path.join(TABNET_FIELD_DIR, "tabnet_field_imputer.joblib"))
    scaler = load(os.path.join(TABNET_FIELD_DIR, "tabnet_field_scaler.joblib"))
    le = load(os.path.join(TABNET_FIELD_DIR, "tabnet_field_label_encoder.joblib"))
    print(f"Classes: {list(le.classes_)}")
    print(f"Feature columns: {len(feature_columns)}")

    # Load test data (already field-level)
    print(f"\nLoading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape: {df.shape}, Fields: {df['fid'].nunique()}")

    fids = df["fid"].to_numpy()
    X = df.drop(columns=["fid", "crop_name"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Align columns to training
    train_cols = list(feature_columns)
    missing = set(train_cols) - set(X.columns)
    extra = set(X.columns) - set(train_cols)
    if missing:
        print(f"  Adding {len(missing)} missing columns (zeros)")
        for col in missing:
            X[col] = 0.0
    if extra:
        print(f"  Dropping {len(extra)} extra columns")
    X = X[train_cols]

    # Preprocess with training artifacts
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Load models
    print("\nLoading models...")
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        raise ImportError("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")

    models = []
    for seed in SEEDS:
        model_path = os.path.join(TABNET_FIELD_DIR, f"tabnet_field_seed_{seed}.zip")
        if not os.path.exists(model_path):
            print(f"\nError: Model not found: {model_path}")
            return
        model = TabNetClassifier()
        model.load_model(model_path)
        models.append(model)
        print(f"  Loaded: tabnet_field_seed_{seed}.zip")

    # Ensemble prediction
    print("\nRunning inference...")
    preds_all = []
    for i, model in enumerate(models):
        probs = model.predict_proba(X_scaled)
        preds_all.append(probs)
        print(f"  Model {i+1}/{len(models)} complete")

    pred_mean = np.mean(preds_all, axis=0)
    preds = np.argmax(pred_mean, axis=1)
    labels = le.inverse_transform(preds)

    # Save
    result = pd.DataFrame({"fid": fids, "crop_name": labels})
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"\n{result['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()
