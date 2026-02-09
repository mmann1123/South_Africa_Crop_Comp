"""
Classical ML inference on holdout test data (34S_20E_259N).

Uses saved Voting and Stacking ensemble models to generate predictions.

Input: combined_test_features.parquet (field-level)
Output: predictions_voting.csv, predictions_stacking.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import load

# Import config for shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import COMBINED_TEST_FEATURES_PATH, MODEL_DIR

# Input
TEST_PARQUET = COMBINED_TEST_FEATURES_PATH

# Model files
VOTING_PIPE = os.path.join(MODEL_DIR, "ensemble_voting.pkl")
STACKING_PIPE = os.path.join(MODEL_DIR, "ensemble_stacking.pkl")
LABEL_ENCODER = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Output
OUTPUT_VOTING = os.path.join(SCRIPT_DIR, "predictions_voting.csv")
OUTPUT_STACKING = os.path.join(SCRIPT_DIR, "predictions_stacking.csv")


def main():
    print("=== Classical ML Inference ===")

    # Check files exist
    for path, name in [
        (TEST_PARQUET, "Test parquet"),
        (VOTING_PIPE, "Voting pipeline"),
        (STACKING_PIPE, "Stacking pipeline"),
        (LABEL_ENCODER, "Label encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Load test data
    print(f"\nLoading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape: {df.shape}")
    print(f"Fields: {df['fid'].nunique()}")

    # Prepare features (exclude non-numeric and metadata)
    fids = df["fid"].to_numpy()
    X = df.drop(columns=["fid", "crop_name"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    print(f"Feature columns: {X.shape[1]}")

    # Load models
    print("\nLoading models...")
    voting_pipe = load(VOTING_PIPE)
    stacking_pipe = load(STACKING_PIPE)
    le = load(LABEL_ENCODER)
    print(f"Classes: {le.classes_}")

    # Predict - Voting
    print("\nRunning Voting ensemble...")
    codes_v = voting_pipe.predict(X)
    labels_v = le.inverse_transform(codes_v)

    # Predict - Stacking
    print("Running Stacking ensemble...")
    codes_s = stacking_pipe.predict(X)
    labels_s = le.inverse_transform(codes_s)

    # Save predictions
    df_voting = pd.DataFrame({"fid": fids, "crop_name": labels_v})
    df_voting.to_csv(OUTPUT_VOTING, index=False)
    print(f"\nSaved: {OUTPUT_VOTING}")

    df_stacking = pd.DataFrame({"fid": fids, "crop_name": labels_s})
    df_stacking.to_csv(OUTPUT_STACKING, index=False)
    print(f"Saved: {OUTPUT_STACKING}")

    # Summary
    print("\n=== Voting Predictions ===")
    print(df_voting["crop_name"].value_counts())

    print("\n=== Stacking Predictions ===")
    print(df_stacking["crop_name"].value_counts())

    # Agreement
    agreement = (labels_v == labels_s).mean()
    print(f"\nVoting/Stacking agreement: {agreement:.2%}")


if __name__ == "__main__":
    main()
