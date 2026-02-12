#!/usr/bin/env python
"""
Step 4: Run out-of-sample inference on holdout region (34S_20E_259N).

Runs all available inference scripts, skipping models whose trained artifacts don't exist.

Output: out_of_sample/predictions_*.csv

Usage:
    python 4_run_inference.py                      # Run all available models
    python 4_run_inference.py --force               # Re-run even if outputs exist
    python 4_run_inference.py --dry-run              # Show what would run
    python 4_run_inference.py --models tabnet        # Run only TabNet
    python 4_run_inference.py --models tabnet 3dcnn  # Run TabNet and 3D CNN
    python 4_run_inference.py --list                 # Show available model names
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

# (key, script, description, output_csv, required_data, required_models)
INFERENCE_STEPS = [
    (
        "xgboost",
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
        "smote",
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
        "classical",
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
        "baseml",
        "inference_base_ml.py",
        "Base ML Models (Pixel-Level)",
        os.path.join(OUT_OF_SAMPLE, "predictions_base_xgb.csv"),
        [COMBINED_TEST_FEATURES_PATH],
        [os.path.join(MODEL_DIR, "ml_base")],  # just check dir exists
    ),
    (
        "cnn",
        "inference_cnn_bilstm.py",
        "CNN-BiLSTM Ensemble (5-seed)",
        os.path.join(OUT_OF_SAMPLE, "predictions_cnn_bilstm.csv"),
        [MERGED_DL_TEST_PATH],
        [os.path.join(MODEL_DIR, f"new_model_seed_{i}_25epochs.pt") for i in range(5)],
    ),
    (
        "tabnet",
        "inference_tabnet.py",
        "TabNet Ensemble (5-seed)",
        os.path.join(OUT_OF_SAMPLE, "predictions_tabnet.csv"),
        [MERGED_DL_TEST_PATH],
        [os.path.join(TABNET_DIR, f"tabnet_seed_{s}.zip") for s in [42, 101, 202, 303, 404]],
    ),
    (
        "3dcnn",
        "inference_3d_cnn.py",
        "3D CNN",
        os.path.join(OUT_OF_SAMPLE, "predictions_3d_cnn.csv"),
        [TEST_PATCH_DATA_PATH],
        [os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5")],
    ),
    (
        "multicnn",
        "inference_multi_channel_cnn.py",
        "Multi-Channel CNN",
        os.path.join(OUT_OF_SAMPLE, "predictions_multi_channel_cnn.csv"),
        [TEST_PATCH_DATA_PATH],
        [os.path.join(MODEL_DIR, "patch_level_cnn.h5")],
    ),
    (
        "ensemble3d",
        "inference_ensemble_3d_cnn.py",
        "Ensemble 3D CNN",
        os.path.join(OUT_OF_SAMPLE, "predictions_ensemble_3d_cnn.csv"),
        [TEST_PATCH_DATA_PATH],
        [
            os.path.join(MODEL_DIR, "meta_model.joblib"),
            os.path.join(MODEL_DIR, "ensemble_3d_cnn_label_encoder.joblib"),
        ],
    ),
]

ALL_KEYS = [step[0] for step in INFERENCE_STEPS]


def main():
    parser = argparse.ArgumentParser(
        description="Run out-of-sample inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model keys: {', '.join(ALL_KEYS)}",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--models", nargs="+", metavar="KEY",
                        help=f"Only run these models (choices: {', '.join(ALL_KEYS)})")
    parser.add_argument("--list", action="store_true", help="List available model keys and exit")
    args = parser.parse_args()

    if args.list:
        print("Available model keys:")
        for key, script, desc, *_ in INFERENCE_STEPS:
            print(f"  {key:10s}  {desc}  ({script})")
        return

    if args.models:
        invalid = [m for m in args.models if m not in ALL_KEYS]
        if invalid:
            parser.error(f"Unknown model(s): {', '.join(invalid)}. Use --list to see options.")
        selected_keys = set(args.models)
    else:
        selected_keys = None  # run all

    print("=" * 60)
    print("STEP 4: RUN INFERENCE")
    if selected_keys:
        print(f"Models:  {', '.join(sorted(selected_keys))}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    for key, script, desc, output_csv, req_data, req_models in INFERENCE_STEPS:
        if selected_keys and key not in selected_keys:
            continue
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
                  "cnn_bilstm", "tabnet", "3d_cnn",
                  "multi_channel_cnn", "ensemble_3d_cnn"]:
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
