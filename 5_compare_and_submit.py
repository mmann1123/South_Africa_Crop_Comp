#!/usr/bin/env python
"""
Step 5: Compare model predictions and create submission.

Runs:
  - compare_predictions.py  (pairwise agreement, ensemble vote, submission)

Output: submissions/prediction.csv

Usage:
    python 5_compare_and_submit.py
"""

import os
import sys
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_OF_SAMPLE = os.path.join(REPO_ROOT, "out_of_sample")


def main():
    print("=" * 60)
    print("STEP 5: COMPARE PREDICTIONS & CREATE SUBMISSION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    script_path = os.path.join(OUT_OF_SAMPLE, "compare_predictions.py")

    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=OUT_OF_SAMPLE,
        check=False,
    )

    if result.returncode == 0:
        submission = os.path.join(REPO_ROOT, "submissions", "prediction.csv")
        if os.path.exists(submission):
            import pandas as pd
            df = pd.read_csv(submission)
            print(f"\nSubmission: {submission}")
            print(f"Fields: {len(df)}")
            print(f"Distribution:\n{df['crop_name'].value_counts().to_string()}")
        print(f"\n[SUCCESS] Comparison complete")
    else:
        print(f"\n[FAILED] Exit code {result.returncode}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPipeline complete! Submit via: git add submissions/ && git push")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
