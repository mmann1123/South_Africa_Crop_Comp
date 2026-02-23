"""
TabNet field-level (temporal) inference on holdout test data (34S_20E_259N).

Uses 5-seed TabNet ensemble trained on field-averaged raw temporal features
(6 bands x 10 months from merged_dl_train.parquet).

Input: merged_dl_test.parquet (pixel-level, aggregated to field here)
Output: predictions_tabnet_temporal_field.csv (field-level)

Required artifacts in saved_models_tabnet_temporal_field/:
  - tabnet_temporal_field_seed_{42,101,202,303,404}.zip
  - tabnet_temporal_field_scaler.joblib
  - tabnet_temporal_field_label_encoder.joblib
  - tabnet_temporal_field_feature_columns.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import MERGED_DL_TEST_PATH, TABNET_TEMPORAL_FIELD_DIR

TEST_PARQUET = MERGED_DL_TEST_PATH
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_tabnet_temporal_field.csv")
SEEDS = [42, 101, 202, 303, 404]


def main():
    print("=== TabNet Field-Level (temporal) Inference ===")

    # Check artifacts exist
    if not os.path.exists(TABNET_TEMPORAL_FIELD_DIR):
        print(f"\nError: Model directory not found: {TABNET_TEMPORAL_FIELD_DIR}")
        print("Run tabnet_temporal_field.py first to train models.")
        return

    for artifact in [
        "tabnet_temporal_field_scaler.joblib",
        "tabnet_temporal_field_label_encoder.joblib",
        "tabnet_temporal_field_feature_columns.joblib",
    ]:
        path = os.path.join(TABNET_TEMPORAL_FIELD_DIR, artifact)
        if not os.path.exists(path):
            print(f"\nError: Missing artifact: {path}")
            return

    # Load preprocessing artifacts
    feature_cols = load(os.path.join(TABNET_TEMPORAL_FIELD_DIR, "tabnet_temporal_field_feature_columns.joblib"))
    scaler = load(os.path.join(TABNET_TEMPORAL_FIELD_DIR, "tabnet_temporal_field_scaler.joblib"))
    le = load(os.path.join(TABNET_TEMPORAL_FIELD_DIR, "tabnet_temporal_field_label_encoder.joblib"))
    print(f"Classes: {list(le.classes_)}")
    print(f"Feature columns: {len(feature_cols)}")

    # Load test data (pixel-level)
    print(f"\nLoading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    df = df.drop(columns=['May'], errors='ignore')
    print(f"Pixel-level shape: {df.shape}, Fields: {df['fid'].nunique()}")

    # Ensure all training feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"  Warning: missing column '{col}', filling with 0")
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Aggregate to field level (same as training)
    print("Aggregating to field level...")
    df_field = df.groupby('fid')[feature_cols].mean().reset_index()
    print(f"Field-level shape: {df_field.shape}")

    fids = df_field["fid"].values
    X = scaler.transform(df_field[feature_cols].values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Load models
    print("\nLoading models...")
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        raise ImportError("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")

    models = []
    for seed in SEEDS:
        model_path = os.path.join(TABNET_TEMPORAL_FIELD_DIR, f"tabnet_temporal_field_seed_{seed}.zip")
        if not os.path.exists(model_path):
            print(f"\nError: Model not found: {model_path}")
            return
        model = TabNetClassifier()
        model.load_model(model_path)
        models.append(model)
        print(f"  Loaded: tabnet_temporal_field_seed_{seed}.zip")

    # Ensemble prediction
    print("\nRunning inference...")
    preds_all = []
    for i, model in enumerate(models):
        probs = model.predict_proba(X)
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
