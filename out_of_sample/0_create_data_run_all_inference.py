#!/usr/bin/env python
"""
Run all out-of-sample inference steps.

Automatically skips steps if output already exists.
Checks for required model files before running inference.

Usage:
    python run_all_inference.py              # Run all available models
    python run_all_inference.py --force      # Re-run even if outputs exist
    python run_all_inference.py --dry-run    # Show what would run
    python run_all_inference.py --models cnn  # Run only CNN-BiLSTM
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import (
    MODEL_DIR, TABNET_DIR, MERGED_DL_TEST_PATH,
    COMBINED_TEST_FEATURES_PATH, TEST_PATCH_DATA_PATH, TEST_PATCHES_GEOJSON_PATH,
    DATA_OUTPUT_DIR,
)

# Define steps with their dependencies and outputs
STEPS = {
    "data_prep": {
        "name": "Prepare Classical ML Data",
        "script": "combine_test_parquets.py",
        "output": COMBINED_TEST_FEATURES_PATH,
        "required_files": [],  # Input parquets checked in script
        "required_models": [],
    },
    "cnn_bilstm": {
        "name": "CNN-BiLSTM Inference",
        "script": "inference_cnn_bilstm.py",
        "output": os.path.join(SCRIPT_DIR, "predictions_cnn_bilstm.csv"),
        "required_files": [MERGED_DL_TEST_PATH],
        "required_models": [
            os.path.join(MODEL_DIR, f"new_model_seed_{i}_25epochs.pt")
            for i in range(5)
        ],
    },
    "classical_ml": {
        "name": "Classical ML Inference",
        "script": "inference_classical_ml.py",
        "output": os.path.join(SCRIPT_DIR, "predictions_voting.csv"),
        "required_files": [COMBINED_TEST_FEATURES_PATH],
        "required_models": [
            os.path.join(MODEL_DIR, "ensemble_voting.pkl"),
            os.path.join(MODEL_DIR, "ensemble_stacking.pkl"),
            os.path.join(MODEL_DIR, "label_encoder.pkl"),
        ],
    },
    "tabnet": {
        "name": "TabNet Inference",
        "script": "inference_tabnet.py",
        "output": os.path.join(SCRIPT_DIR, "predictions_tabnet.csv"),
        "required_files": [MERGED_DL_TEST_PATH],
        "required_models": [
            os.path.join(TABNET_DIR, f"tabnet_seed_{seed}.zip")
            for seed in [42, 101, 202, 303, 404]
        ],
    },
    "patch_data": {
        "name": "Create Test Patches",
        "script": "create_test_patches.py",
        "output": TEST_PATCHES_GEOJSON_PATH,
        "required_files": [],  # Checked in script
        "required_models": [],
    },
    "3d_cnn": {
        "name": "3D CNN Inference",
        "script": "inference_3d_cnn.py",
        "output": os.path.join(SCRIPT_DIR, "predictions_3d_cnn.csv"),
        "required_files": [TEST_PATCH_DATA_PATH],
        "required_models": [
            os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5"),
        ],
    },
    "compare": {
        "name": "Compare Predictions",
        "script": "compare_predictions.py",
        "output": None,  # Always run
        "required_files": [],
        "required_models": [],
    },
}

# Run order
RUN_ORDER = ["data_prep", "cnn_bilstm", "classical_ml", "tabnet", "patch_data", "3d_cnn", "compare"]


def check_file_exists(path):
    """Check if file exists."""
    return os.path.exists(path)


def check_models_exist(model_paths):
    """Check if all required model files exist."""
    missing = [p for p in model_paths if not os.path.exists(p)]
    return missing


def run_step(step_id, step_config, force=False, dry_run=False):
    """Run a single inference step."""
    name = step_config["name"]
    script = step_config["script"]
    output = step_config["output"]
    required_files = step_config["required_files"]
    required_models = step_config["required_models"]

    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")

    # Check if output already exists
    if output and not force:
        if check_file_exists(output):
            print(f"[SKIP] Output exists: {os.path.basename(output)}")
            return "skipped"

    # Check required data files
    missing_files = [f for f in required_files if not check_file_exists(f)]
    if missing_files:
        print(f"[SKIP] Missing required data files:")
        for f in missing_files:
            print(f"  - {os.path.basename(f)}")
        return "missing_data"

    # Check required model files
    missing_models = check_models_exist(required_models)
    if missing_models:
        print(f"[SKIP] Missing trained models:")
        for m in missing_models:
            print(f"  - {os.path.basename(m)}")
        print(f"  Run training scripts first to create these models.")
        return "missing_models"

    # Run the script
    script_path = os.path.join(SCRIPT_DIR, script)
    print(f"Running: {script}")

    if dry_run:
        print(f"[DRY RUN] Would execute: python {script}")
        return "dry_run"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            check=False,
        )
        if result.returncode == 0:
            print(f"[SUCCESS] {name}")
            return "success"
        else:
            print(f"[FAILED] {name} (exit code {result.returncode})")
            return "failed"
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return "error"


def main():
    parser = argparse.ArgumentParser(description="Run out-of-sample inference pipeline")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--models", nargs="+", choices=list(STEPS.keys()),
                        help="Run only specific steps")
    args = parser.parse_args()

    print("=" * 60)
    print("OUT-OF-SAMPLE INFERENCE PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Determine which steps to run
    steps_to_run = args.models if args.models else RUN_ORDER

    # Check overall status
    print("\n--- Pre-flight Check ---")
    for step_id in steps_to_run:
        step = STEPS[step_id]
        model_status = "✓" if not check_models_exist(step["required_models"]) else "✗"
        output_status = ""
        if step["output"]:
            output_status = " [output exists]" if check_file_exists(step["output"]) else ""
        print(f"  {step_id}: {step['name']} - Models: {model_status}{output_status}")

    # Run steps
    results = {}
    for step_id in steps_to_run:
        step = STEPS[step_id]
        results[step_id] = run_step(step_id, step, force=args.force, dry_run=args.dry_run)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for step_id, status in results.items():
        name = STEPS[step_id]["name"]
        status_icon = {
            "success": "✓",
            "skipped": "○",
            "dry_run": "◇",
            "missing_data": "✗",
            "missing_models": "✗",
            "failed": "✗",
            "error": "✗",
        }.get(status, "?")
        print(f"  [{status_icon}] {name}: {status}")

    # Show available predictions
    print("\n--- Available Predictions ---")
    pred_files = [
        "predictions_voting.csv",
        "predictions_stacking.csv",
        "predictions_cnn_bilstm.csv",
        "predictions_tabnet.csv",
        "predictions_3d_cnn.csv",
    ]
    for f in pred_files:
        path = os.path.join(SCRIPT_DIR, f)
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path)
            print(f"  ✓ {f}: {len(df)} predictions")
        else:
            print(f"  ✗ {f}: not found")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
