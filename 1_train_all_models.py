#!/usr/bin/env python
"""
Step 1: Train all models (Classical ML + Deep Learning).

Runs:
  - run_all_classical_models.py  (SMOTE, Ensemble Voting/Stacking, base ML)
  - run_all_dl_models.py         (TabNet, CNN-BiLSTM, 3D CNN)

Usage:
    python 1_train_all_models.py                    # Run all
    python 1_train_all_models.py --classical-only   # Only classical ML
    python 1_train_all_models.py --dl-only          # Only deep learning
    python 1_train_all_models.py --dry-run          # Show what would run
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEP_LEARN_SRC = os.path.join(REPO_ROOT, "deep_learn", "src")

SCRIPTS = {
    "classical": (
        os.path.join(DEEP_LEARN_SRC, "run_all_classical_models.py"),
        "Classical ML Models (SMOTE, Ensemble, base ML)",
    ),
    "dl": (
        os.path.join(DEEP_LEARN_SRC, "run_all_dl_models.py"),
        "Deep Learning Models (TabNet, CNN-BiLSTM, 3D CNN)",
    ),
}


def run_script(script_path, description, extra_args=None, dry_run=False):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {os.path.relpath(script_path, REPO_ROOT)}")
    print(f"{'='*60}")

    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, cwd=DEEP_LEARN_SRC, check=False)
    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{status}] {description}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--classical-only", action="store_true", help="Only classical ML")
    parser.add_argument("--dl-only", action="store_true", help="Only deep learning")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: TRAIN ALL MODELS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    scripts_to_run = []
    if not args.dl_only:
        scripts_to_run.append(SCRIPTS["classical"])
    if not args.classical_only:
        scripts_to_run.append(SCRIPTS["dl"])

    if not scripts_to_run:
        print("[ERROR] No scripts selected.")
        sys.exit(1)

    results = []
    for script_path, desc in scripts_to_run:
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
    print("\nNext: python 2_compare_training_results.py")
    sys.exit(0 if all(s for _, s in results) else 1)


if __name__ == "__main__":
    main()
