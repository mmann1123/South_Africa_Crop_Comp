#!/usr/bin/env python
"""
Step 3: Create out-of-sample test datasets for holdout region (34S_20E_259N).

Runs:
  - combine_test_parquets.py  -> data/combined_test_features.parquet (classical ML)
  - create_test_patches.py    -> data/test_patch_data.parquet (3D CNN)

Usage:
    python 3_create_test_data.py              # Run all
    python 3_create_test_data.py --dry-run    # Show what would run
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_OF_SAMPLE = os.path.join(REPO_ROOT, "out_of_sample")
DATA_DIR = os.path.join(REPO_ROOT, "data")

SCRIPTS = [
    (
        os.path.join(OUT_OF_SAMPLE, "combine_test_parquets.py"),
        "Combine test parquets (Classical ML features)",
        os.path.join(DATA_DIR, "combined_test_features.parquet"),
    ),
    (
        os.path.join(OUT_OF_SAMPLE, "create_test_patches.py"),
        "Create test patches (3D CNN)",
        os.path.join(DATA_DIR, "test_patch_data.parquet"),
    ),
]


def run_script(script_path, description, output_path, force=False, dry_run=False):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {os.path.relpath(script_path, REPO_ROOT)}")
    print(f"{'='*60}")

    if not force and os.path.exists(output_path):
        print(f"  [SKIP] Output exists: {os.path.relpath(output_path, REPO_ROOT)}")
        return True

    if dry_run:
        print(f"  [DRY RUN] Would execute: python {os.path.basename(script_path)}")
        return True

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=OUT_OF_SAMPLE,
        check=False,
    )
    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{status}] {description}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Create out-of-sample test datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3: CREATE TEST DATA")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:  {DATA_DIR}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    results = []
    for script_path, desc, output_path in SCRIPTS:
        success = run_script(script_path, desc, output_path,
                             force=args.force, dry_run=args.dry_run)
        results.append((desc, success))
        if not success and not args.dry_run:
            print(f"\n[WARNING] {desc} failed. Continuing with remaining steps...")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for desc, success in results:
        icon = "OK" if success else "FAILED"
        print(f"  [{icon}] {desc}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext: python 4_run_inference.py")
    sys.exit(0 if all(s for _, s in results) else 1)


if __name__ == "__main__":
    main()
