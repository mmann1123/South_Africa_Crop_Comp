#!/usr/bin/env python
"""
Run all deep learning models sequentially.

This script executes:
  Stage 2: Pixel/Field-Level DL
    - cnn_bilstm.py (CNN+BiLSTM ensemble, best model)
    - field_acc_cnnlstm.py (field-level evaluation)
    - TabTransformer_Final_Field.py (TabNet ensemble)

  Stage 3: Patch-Level DL (requires patch data to exist)
    - 3D_CNN.py

Usage:
    python run_all_dl_models.py              # run all
    python run_all_dl_models.py --stage 2    # only pixel/field-level
    python run_all_dl_models.py --stage 3    # only patch-level
    python run_all_dl_models.py --skip-patch # skip patch models if data doesn't exist
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
from config import MERGED_DL_PATH, PATCH_DATA_PATH

# Scripts to run in order
STAGE_2_SCRIPTS = [
    (
        "Deep Learning/Pixel_Field_Level/cnn_bilstm.py",
        "CNN+BiLSTM Ensemble (5-seed, 25 epochs each)",
    ),
    (
        "Deep Learning/Pixel_Field_Level/field_acc_cnnlstm.py",
        "CNN+BiLSTM Field-Level Evaluation",
    ),
    # (
    #     "Deep Learning/Pixel_Field_Level/TabTransformer_Final_Field.py",
    #     "TabNet Ensemble (5 models)",
    # ),
]

STAGE_3_SCRIPTS = [
    # ("Deep Learning/Patch Level/3D_CNN.py", "3D CNN Patch-Level (20 epochs)"),
]

# Files required for each stage (from config)
STAGE_2_DATA = MERGED_DL_PATH
STAGE_3_DATA = PATCH_DATA_PATH


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
        description="Run all deep learning models sequentially"
    )
    parser.add_argument(
        "--stage", type=int, choices=[2, 3], help="Run only stage 2 or 3"
    )
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="Skip patch-level models if data doesn't exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without executing"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    scripts_to_run = []

    # Determine which stages to run
    run_stage_2 = args.stage is None or args.stage == 2
    run_stage_3 = args.stage is None or args.stage == 3

    # Check data and build script list
    if run_stage_2:
        if not check_data_exists(STAGE_2_DATA):
            print(f"[WARNING] Stage 2 data not found: {STAGE_2_DATA}")
            print(
                "         Run create_merged_dl_data.py first or download from Google Drive."
            )
            if args.stage == 2:
                sys.exit(1)
        else:
            scripts_to_run.extend(STAGE_2_SCRIPTS)

    if run_stage_3:
        if not check_data_exists(STAGE_3_DATA):
            msg = f"[WARNING] Stage 3 data not found: {STAGE_3_DATA}"
            if args.skip_patch:
                print(msg)
                print("         Skipping patch-level models (--skip-patch).")
            elif args.stage == 3:
                print(msg)
                print("         Run Create_Patches.py and Create Master Data.py first.")
                sys.exit(1)
            else:
                print(msg)
                print(
                    "         Skipping patch-level models. Use --stage 3 to require them."
                )
        else:
            scripts_to_run.extend(STAGE_3_SCRIPTS)

    if not scripts_to_run:
        print("[ERROR] No scripts to run. Check data files exist.")
        sys.exit(1)

    # Print plan
    print("\n" + "=" * 70)
    print("DEEP LEARNING MODEL PIPELINE")
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

    # Reports generated
    print("\nReports generated in: src/reports/")
    print("Run 'python compare_models.py' to compare all models.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
