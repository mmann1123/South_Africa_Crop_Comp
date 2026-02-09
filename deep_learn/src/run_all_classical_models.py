#!/usr/bin/env python
"""
Run all classical machine learning models sequentially.

This script executes:
  Stage 1a: Field-Level Classical ML
    - xg_boost_hyper.py (XGBoost with Optuna tuning)
    - SMOTE_meta.py (SMOTE + stacked meta-learner)
    - Ensemble - Voting and Stacking.py (voting & stacking ensembles)

  Stage 1b: Pixel-Level Classical ML
    - base_ml_models.py (LightGBM, XGBoost, RF, SVM baselines)

Usage:
    python run_all_classical_models.py                  # run all
    python run_all_classical_models.py --stage 1a       # only field-level
    python run_all_classical_models.py --stage 1b       # only pixel-level
    python run_all_classical_models.py --dry-run        # print plan only
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
from config import FINAL_DATA_PATH

STAGE_1A_SCRIPTS = [
    # (
    #     "Classical Machine Learning/Field Level/xg_boost_hyper.py",
    #     "XGBoost Optuna Hyperparameter Tuning",
    # ),
    (
        "Classical Machine Learning/Field Level/SMOTE_meta.py",
        "SMOTE + Stacked Meta-Learner",
    ),
    (
        "Classical Machine Learning/Field Level/Ensemble - Voting and Stacking.py",
        "Ensemble Voting & Stacking",
    ),
]

STAGE_1B_SCRIPTS = [
    (
        "Classical Machine Learning/pixel_level/base_ml_models.py",
        "Base ML Models (LightGBM, XGBoost, RF, SVM)",
    ),
]

# All classical ML scripts use FINAL_DATA_PATH from config
REQUIRED_DATA = FINAL_DATA_PATH


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return True if successful."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print(f"Script:  {script_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        )
        success = result.returncode == 0
        status = "SUCCESS" if success else f"FAILED (exit code {result.returncode})"
        print(f"\n[{status}] {description}")
        return success
    except Exception as e:
        print(f"\n[ERROR] {description}: {e}")
        return False


def check_data_exists(filepath: str) -> bool:
    """Check if required data file exists."""
    return os.path.exists(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Run all classical machine learning models sequentially"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1a", "1b"],
        help="Run only field-level (1a) or pixel-level (1b)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without executing"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check data exists
    if not check_data_exists(REQUIRED_DATA):
        print(f"[ERROR] Required data not found: {REQUIRED_DATA}")
        print("        Run create_final_data.py first.")
        sys.exit(1)

    # Determine which stages to run
    scripts_to_run = []
    if args.stage is None or args.stage == "1a":
        scripts_to_run.extend(STAGE_1A_SCRIPTS)
    if args.stage is None or args.stage == "1b":
        scripts_to_run.extend(STAGE_1B_SCRIPTS)

    if not scripts_to_run:
        print("[ERROR] No scripts to run.")
        sys.exit(1)

    # Print plan
    print("\n" + "=" * 70)
    print("CLASSICAL ML MODEL PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nScripts to run ({len(scripts_to_run)}):")
    for i, (script, desc) in enumerate(scripts_to_run, 1):
        print(f"  {i}. {desc}")
        print(f"     {script}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute the above scripts. Exiting.")
        sys.exit(0)

    # Run scripts
    results = []
    for script, desc in scripts_to_run:
        success = run_script(script, desc)
        results.append((desc, success))
        if not success:
            print(f"\n[STOPPING] {desc} failed. Fix the error and re-run.")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for desc, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {desc}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nCompleted: {passed}/{total} scripts")
    print(f"Finished:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nReports generated in: src/reports/")
    print("Run 'python compare_models.py' to compare all models.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
