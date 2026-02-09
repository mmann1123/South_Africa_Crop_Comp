#!/usr/bin/env python
"""
Step 2: Compare training results across all models.

Runs:
  - compare_models.py  (scans reports/ for metadata.json, builds comparison table)

Output: deep_learn/src/reports/model_comparison.csv + model_comparison.pdf

Usage:
    python 2_compare_training_results.py
"""

import os
import sys
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEP_LEARN_SRC = os.path.join(REPO_ROOT, "deep_learn", "src")


def main():
    print("=" * 60)
    print("STEP 2: COMPARE TRAINING RESULTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    script_path = os.path.join(DEEP_LEARN_SRC, "compare_models.py")

    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=DEEP_LEARN_SRC,
        check=False,
    )

    if result.returncode == 0:
        print(f"\n[SUCCESS] Model comparison complete")
        print(f"Reports: {os.path.join(DEEP_LEARN_SRC, 'reports')}")
    else:
        print(f"\n[FAILED] Exit code {result.returncode}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext: python 3_create_test_data.py")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
