#!/usr/bin/env python
"""
Step 4: Run out-of-sample inference on holdout region (34S_20E_259N).

Runs all available inference scripts, skipping models whose trained artifacts don't exist.

Output: out_of_sample/predictions_*.csv

Usage:
    python 4_run_inference.py              # Run all available models
    python 4_run_inference.py --force      # Re-run even if outputs exist
    python 4_run_inference.py --dry-run    # Show what would run
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_OF_SAMPLE = os.path.join(REPO_ROOT, "out_of_sample")
DEEP_LEARN_SRC = os.path.join(REPO_ROOT, "deep_learn", "src")

sys.path.insert(0, DEEP_LEARN_SRC)
from config import (
    MODEL_DIR, TABNET_DIR, XGB_TUNER_DIR, MERGED_DL_TEST_PATH,
    COMBINED_TEST_FEATURES_PATH, TEST_PATCH_DATA_PATH,
)

# (script, description, output_csv, required_data, required_models)
INFERENCE_STEPS = [
    (
        "inference_xgboost.py",
        "XGBoost (Optuna-tuned)",
        os.path.join(OUT_OF_SAMPLE, "predictions_xgboost.csv"),
        [COMBINED_TEST_FEATURES_PATH],
        [
            os.path.join(XGB_TUNER_DIR, "final_xgb_model.joblib"),
            os.path.join(XGB_TUNER_DIR, "imputer.joblib"),
            os.path.join(XGB_TUNER_DIR, "scaler.joblib"),
            os.path.join(XGB_TUNER_DIR, "label_encoder.joblib"),
        ],
    ),
    (
        "inference_smote_stacked.py",
        "SMOTE Stacked Ensemble",
        os.path.join(OUT_OF_SAMPLE, "predictions_smote_stacked.csv"),
        [COMBINED_TEST_FEATURES_PATH],
        [
            os.path.join(MODEL_DIR, "stacked_model_v1.joblib"),
            os.path.join(MODEL_DIR, "imputer.joblib"),
            os.path.join(MODEL_DIR, "scaler.joblib"),
            os.path.join(MODEL_DIR, "label_encoder.joblib"),
        ],
    ),
    (
        "inference_classical_ml.py",
        "Classical ML (Voting + Stacking)",
        os.path.join(OUT_OF_SAMPLE, "predictions_voting.csv"),
        [COMBINED_TEST_FEATURES_PATH],
        [
            os.path.join(MODEL_DIR, "ensemble_voting.pkl"),
            os.path.join(MODEL_DIR, "ensemble_stacking.pkl"),
            os.path.join(MODEL_DIR, "label_encoder.pkl"),
        ],
    ),
    (
        "inference_base_ml.py",
        "Base ML Models (Pixel-Level)",
        os.path.join(OUT_OF_SAMPLE, "predictions_base_xgb.csv"),
        [MERGED_DL_TEST_PATH],
        [os.path.join(MODEL_DIR, "ml_base")],  # just check dir exists
    ),
    (
        "inference_cnn_bilstm.py",
        "CNN-BiLSTM Ensemble (5-seed)",
        os.path.join(OUT_OF_SAMPLE, "predictions_cnn_bilstm.csv"),
        [MERGED_DL_TEST_PATH],
        [os.path.join(MODEL_DIR, f"new_model_seed_{i}_25epochs.pt") for i in range(5)],
    ),
    (
        "inference_tabnet.py",
        "TabNet Ensemble (5-seed)",
        os.path.join(OUT_OF_SAMPLE, "predictions_tabnet.csv"),
        [MERGED_DL_TEST_PATH],
        [os.path.join(TABNET_DIR, f"tabnet_seed_{s}.zip") for s in [42, 101, 202, 303, 404]],
    ),
    (
        "inference_3d_cnn.py",
        "3D CNN",
        os.path.join(OUT_OF_SAMPLE, "predictions_3d_cnn.csv"),
        [TEST_PATCH_DATA_PATH],
        [os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5")],
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Run out-of-sample inference")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 4: RUN INFERENCE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    for script, desc, output_csv, req_data, req_models in INFERENCE_STEPS:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")

        # Check output exists
        if not args.force and os.path.exists(output_csv):
            print(f"  [SKIP] Output exists: {os.path.basename(output_csv)}")
            results.append((desc, "skipped"))
            continue

        # Check required data
        missing_data = [f for f in req_data if not os.path.exists(f)]
        if missing_data:
            print(f"  [SKIP] Missing data: {[os.path.basename(f) for f in missing_data]}")
            results.append((desc, "missing_data"))
            continue

        # Check required models
        missing_models = [m for m in req_models if not os.path.exists(m)]
        if missing_models:
            print(f"  [SKIP] Missing models: {[os.path.basename(m) for m in missing_models]}")
            results.append((desc, "missing_models"))
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would execute: python {script}")
            results.append((desc, "dry_run"))
            continue

        script_path = os.path.join(OUT_OF_SAMPLE, script)
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=OUT_OF_SAMPLE,
            check=False,
        )
        if result.returncode == 0:
            print(f"  [SUCCESS] {desc}")
            results.append((desc, "success"))
        else:
            print(f"  [FAILED] {desc} (exit code {result.returncode})")
            results.append((desc, "failed"))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for desc, status in results:
        icon = {"success": "OK", "skipped": "SKIP", "dry_run": "DRY",
                "missing_data": "MISS", "missing_models": "MISS", "failed": "FAIL"}.get(status, "?")
        print(f"  [{icon}] {desc}: {status}")

    # Show available predictions
    print(f"\n--- Available Predictions ---")
    for name in ["xgboost", "smote_stacked", "voting", "stacking",
                  "base_lr", "base_rf", "base_lgbm", "base_xgb",
                  "cnn_bilstm", "tabnet", "3d_cnn"]:
        path = os.path.join(OUT_OF_SAMPLE, f"predictions_{name}.csv")
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path)
            print(f"  predictions_{name}.csv: {len(df)} fields")
        else:
            print(f"  predictions_{name}.csv: not found")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext: python 5_compare_and_submit.py")
    sys.exit(0)


if __name__ == "__main__":
    main()
