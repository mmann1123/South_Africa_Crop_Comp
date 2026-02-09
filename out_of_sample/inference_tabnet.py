"""
TabNet ensemble inference on holdout test data (34S_20E_259N).

Uses saved TabNet models (5-seed ensemble) to generate predictions.

Input: merged_dl_test_259N.parquet (pixel-level)
Output: predictions_tabnet.csv (field-level)

Note: Requires pytorch-tabnet to be installed.
      Models must be trained first using TabTransformer_Final_Field.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Import config for shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import MERGED_DL_TEST_PATH, TABNET_DIR

# Input
TEST_PARQUET = MERGED_DL_TEST_PATH

# Output
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_tabnet.csv")

# Model seeds (must match training)
SEEDS = [42, 101, 202, 303, 404]

# Class mapping: LabelEncoder on crop_id [1,2,3,4,5] -> [0,1,2,3,4]
# crop_id mapping: 1=Wheat, 2=Barley, 3=Canola, 4=Lucerne/Medics, 5=Small grain grazing
CLASS_NAMES = ["Wheat", "Barley", "Canola", "Lucerne/Medics", "Small grain grazing"]


def load_test_data():
    """Load and prepare test data."""
    print(f"Loading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape: {df.shape}")
    print(f"Unique fields: {df['fid'].nunique()}")

    # Exclude metadata columns
    exclude_cols = {"id", "point", "fid", "crop_id", "crop_name"}
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    # Fill NaN with median (same as training)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # One-hot encode Type if present
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"])

    one_hot_cols = [col for col in df.columns if col.startswith("Type_")]
    feature_columns = numeric_cols + one_hot_cols

    # Standardize
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    features = df[feature_columns].astype(np.float32).values
    fids = df["fid"].values

    print(f"Feature columns: {len(feature_columns)}")
    return features, fids


def load_models():
    """Load all TabNet models."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        raise ImportError("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")

    models = []
    for seed in SEEDS:
        model_path = os.path.join(TABNET_DIR, f"tabnet_seed_{seed}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run TabTransformer_Final_Field.py first to train models."
            )

        model = TabNetClassifier()
        model.load_model(model_path)
        models.append(model)
        print(f"Loaded: tabnet_seed_{seed}.zip")

    return models


def aggregate_to_field_level(preds, fids):
    """Aggregate pixel predictions to field level by majority vote."""
    df = pd.DataFrame({"fid": fids, "pred_label": preds})

    # Majority vote per field
    field_preds = df.groupby("fid")["pred_label"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )

    return field_preds


def main():
    print("=== TabNet Ensemble Inference ===")

    # Check if TabNet directory exists
    if not os.path.exists(TABNET_DIR):
        print(f"\nError: TabNet model directory not found: {TABNET_DIR}")
        print("Run TabTransformer_Final_Field.py first to train models.")
        return

    # Load data
    features, fids = load_test_data()

    # Load models
    print("\nLoading models...")
    try:
        models = load_models()
    except FileNotFoundError as e:
        print(f"\n{e}")
        return

    print(f"Loaded {len(models)} models")

    # Run ensemble prediction
    print("\nRunning inference...")
    preds_all = []
    for i, model in enumerate(models):
        probs = model.predict_proba(features)
        preds_all.append(probs)
        print(f"  Model {i+1}/{len(models)} complete")

    # Average probabilities across ensemble
    pred_mean = np.mean(preds_all, axis=0)
    preds = np.argmax(pred_mean, axis=1)
    print(f"Total pixel predictions: {len(preds)}")

    # Aggregate to field level
    print("\nAggregating to field level...")
    field_preds = aggregate_to_field_level(preds, fids)
    print(f"Total fields: {len(field_preds)}")

    # Convert label indices to class names
    field_labels = [CLASS_NAMES[p] for p in field_preds.values]

    # Save predictions
    df_out = pd.DataFrame({
        "fid": field_preds.index,
        "crop_name": field_labels,
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Summary
    print("\n=== Prediction Distribution ===")
    print(df_out["crop_name"].value_counts())


if __name__ == "__main__":
    main()
