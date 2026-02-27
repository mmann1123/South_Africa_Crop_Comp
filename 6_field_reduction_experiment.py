#!/usr/bin/env python
"""
Step 6: Field reduction experiment.

Tests how model performance (OOS F1 macro) degrades as the number of
training fields is reduced to 75%, 50%, and 25%.

Runs the full pipeline:
  1. Train 6 models at each fraction (via correct conda envs)
  2. Generate OOS predictions on the holdout zone
  3. Score and compare results, produce CSV + plot

Models tested:
  - TabNet (pixel)          [deep_field env]
  - L-TAE Field (field)     [deep_field env]
  - L-TAE (pixel)           [deep_field env]
  - XGBoost (field)         [ml_field env]
  - Base LightGBM (pixel)   [ml_field env]
  - Base LR (pixel)         [ml_field env]

Output:
  - experiments/field_reduction/results/field_reduction_results.csv
  - experiments/field_reduction/results/field_reduction_plot.png

Usage:
    python 6_field_reduction_experiment.py                # all models
    python 6_field_reduction_experiment.py --skip-tabnet  # skip TabNet (slowest)
    python 6_field_reduction_experiment.py --dry-run       # show commands only
"""

import os
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "experiments", "field_reduction")


def main():
    print("=" * 60)
    print("STEP 6: FIELD REDUCTION EXPERIMENT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    script_path = os.path.join(EXPERIMENT_DIR, "run_experiment.py")

    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    # Forward all CLI arguments to run_experiment.py
    cmd = [sys.executable, script_path] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=EXPERIMENT_DIR, check=False)

    if result.returncode == 0:
        results_csv = os.path.join(EXPERIMENT_DIR, "results", "field_reduction_results.csv")
        if os.path.exists(results_csv):
            import pandas as pd
            df = pd.read_csv(results_csv)
            print(f"\n{'=' * 60}")
            print("RESULTS SUMMARY")
            print(f"{'=' * 60}")
            print(df.to_string(index=False))
        print(f"\n[SUCCESS] Experiment complete")
    else:
        print(f"\n[FAILED] Exit code {result.returncode}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
