#!/usr/bin/env python
"""
Step 0: Create all training and test datasets.

Runs:
  - create_final_data.py      -> data/final_data.parquet (xr_fresh features for classical ML)
  - create_merged_dl_data.py   -> data/merged_dl_train.parquet + data/merged_dl_test.parquet

Usage:
    python 0_create_training_data.py              # Run all
    python 0_create_training_data.py --dry-run    # Show what would run
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEP_LEARN_SRC = os.path.join(REPO_ROOT, "deep_learn", "src")
DATA_DIR = os.path.join(REPO_ROOT, "data")

SCRIPTS = [
    (
        os.path.join(DEEP_LEARN_SRC, "create_final_data.py"),
        "Create xr_fresh features (Classical ML)",
        os.path.join(DATA_DIR, "final_data.parquet"),
    ),
    (
        os.path.join(DEEP_LEARN_SRC, "create_merged_dl_data.py"),
        "Create merged DL data (training + test)",
        os.path.join(DATA_DIR, "merged_dl_train.parquet"),
    ),
]


def run_script(script_path, description, dry_run=False):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {os.path.relpath(script_path, REPO_ROOT)}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] Would execute: python {os.path.basename(script_path)}")
        return True

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=DEEP_LEARN_SRC,
        check=False,
    )
    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{status}] {description}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Create all training datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 0: CREATE TRAINING DATA")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:  {DATA_DIR}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    results = []
    for script_path, desc, output_path in SCRIPTS:
        success = run_script(script_path, desc, dry_run=args.dry_run)
        results.append((desc, success))
        if not success and not args.dry_run:
            print(f"\n[STOPPING] {desc} failed.")
            break

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for desc, success in results:
        icon = "OK" if success else "FAILED"
        print(f"  [{icon}] {desc}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0 if all(s for _, s in results) else 1)


if __name__ == "__main__":
    main()
