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
import pandas as pd
import numpy as np
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Prediction files to compare
PREDICTION_FILES = {
    "Voting": os.path.join(SCRIPT_DIR, "predictions_voting.csv"),
    "Stacking": os.path.join(SCRIPT_DIR, "predictions_stacking.csv"),
    "CNN-BiLSTM": os.path.join(SCRIPT_DIR, "predictions_cnn_bilstm.csv"),
    "TabNet": os.path.join(SCRIPT_DIR, "predictions_tabnet.csv"),
    "3D CNN": os.path.join(SCRIPT_DIR, "predictions_3d_cnn.csv"),
}

# Output
SUBMISSION_PATH = os.path.join(REPO_ROOT, "submissions", "prediction.csv")


def load_predictions():
    """Load all available prediction files."""
    predictions = {}

    for name, path in PREDICTION_FILES.items():
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


def ensemble_vote(predictions):
    """Create ensemble prediction via majority vote."""
    if len(predictions) < 2:
        return None

    print("\n" + "=" * 60)
    print("ENSEMBLE MAJORITY VOTE")
    print("=" * 60)

    # Merge all predictions
    merged = None
    names = list(predictions.keys())

    for name, df in predictions.items():
        df = df.rename(columns={"crop_name": name})
        if merged is None:
            merged = df[["fid", name]]
        else:
            merged = merged.merge(df[["fid", name]], on="fid", how="inner")

    # Majority vote
    pred_cols = [n for n in names if n in merged.columns]

    def majority_vote(row):
        votes = [row[c] for c in pred_cols]
        return Counter(votes).most_common(1)[0][0]

    merged["ensemble"] = merged.apply(majority_vote, axis=1)

    print("\nEnsemble prediction distribution:")
    print(merged["ensemble"].value_counts().to_string())

    return merged[["fid", "ensemble"]].rename(columns={"ensemble": "crop_name"})


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

    # Create ensemble if multiple models
    if len(predictions) >= 2:
        ensemble_df = ensemble_vote(predictions)

    # Prompt for submission
    print("\n" + "=" * 60)
    print("SELECT MODEL FOR SUBMISSION")
    print("=" * 60)

    available = list(predictions.keys())
    if len(predictions) >= 2:
        available.append("Ensemble")

    print("\nAvailable models:")
    for i, name in enumerate(available, 1):
        print(f"  {i}. {name}")

    # Default to first available model
    default_model = available[0]
    print(f"\nUsing default: {default_model}")
    print("(Edit this script to change the default or add command-line args)")

    # Create submission
    if default_model == "Ensemble":
        create_submission(ensemble_df, "Ensemble")
    else:
        create_submission(predictions[default_model], default_model)


if __name__ == "__main__":
    main()
