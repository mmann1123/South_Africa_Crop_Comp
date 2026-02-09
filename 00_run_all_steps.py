#!/usr/bin/env python
"""
Master pipeline: run all steps 0 through 5 sequentially.

Steps:
  0  Create training data      (final_data.parquet, merged_dl_train/test.parquet, etc.)
  1  Train all models           (classical ML + deep learning)
  2  Compare training results   (model metrics comparison)
  3  Create test data           (out-of-sample feature datasets)
  4  Run inference              (predictions on holdout region)
  5  Compare & submit           (ensemble predictions, submission CSV)

Usage:
    python 00_run_all_steps.py                  # Run all steps
    python 00_run_all_steps.py --start 3        # Start from step 3
    python 00_run_all_steps.py --stop 2         # Run steps 0-2 only
    python 00_run_all_steps.py --start 1 --stop 1  # Run only step 1
    python 00_run_all_steps.py --dry-run        # Show what would run
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("0_create_training_data.py", "Create training data"),
    ("1_train_all_models.py", "Train all models"),
    ("2_compare_training_results.py", "Compare training results"),
    ("3_create_test_data.py", "Create test data"),
    ("4_run_inference.py", "Run inference"),
    ("5_compare_and_submit.py", "Compare & submit"),
]


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline (steps 0-5)")
    parser.add_argument("--start", type=int, default=0, choices=range(6),
                        help="First step to run (default: 0)")
    parser.add_argument("--stop", type=int, default=5, choices=range(6),
                        help="Last step to run (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    if args.start > args.stop:
        print(f"[ERROR] --start ({args.start}) must be <= --stop ({args.stop})")
        sys.exit(1)

    selected = STEPS[args.start : args.stop + 1]

    print("=" * 60)
    print("FULL PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps:   {args.start} -> {args.stop}")
    print("=" * 60)

    if args.dry_run:
        for script, desc in selected:
            print(f"  [DRY RUN] Step {STEPS.index((script, desc))}: {desc}  ({script})")
        sys.exit(0)

    results = []
    for script, desc in selected:
        step_num = STEPS.index((script, desc))
        print(f"\n{'=' * 60}")
        print(f"  STEP {step_num}: {desc}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        script_path = os.path.join(REPO_ROOT, script)
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=REPO_ROOT,
            check=False,
        )

        if result.returncode == 0:
            print(f"\n  [SUCCESS] Step {step_num}: {desc}")
            results.append((step_num, desc, "success"))
        else:
            print(f"\n  [FAILED] Step {step_num}: {desc} (exit code {result.returncode})")
            results.append((step_num, desc, "failed"))
            print(f"\nStopping pipeline due to failure at step {step_num}.")
            break

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for step_num, desc, status in results:
        icon = "OK" if status == "success" else "FAIL"
        print(f"  [{icon}] Step {step_num}: {desc}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    failed = any(s == "failed" for _, _, s in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
