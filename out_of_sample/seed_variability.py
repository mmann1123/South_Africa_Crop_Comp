"""
Seed-to-seed variability of spatial-transfer macro-F1 for the 5-seed ensembles.

The deep inference scripts emit per-seed field predictions
(predictions_<base>_seed<seed>.csv). This scores each seed member against the
holdout ground truth and reports mean +/- std macro-F1 across seeds — quantifying
optimization noise, complementary to the field-resampling bootstrap CI.

Output: scoring_results/seed_f1_macro.csv
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from compare_predictions import load_ground_truth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "scoring_results")

# Map per-seed file base token -> manuscript display name (matches bootstrap_ci.py).
BASE_TO_DISPLAY = {
    "ltae": "L-TAE (pixel)",
    "ltae_field": "L-TAE Field (field)",
    "tempcnn": "TempCNN (pixel)",
    "cnn_bilstm": "CNN-BiLSTM (pixel)",
    "tabnet": "TabNet (pixel)",
}

SEED_RE = re.compile(r"^predictions_(?P<base>.+)_seed(?P<seed>\w+)\.csv$")


def main():
    gt = load_ground_truth()
    if gt is None:
        return
    gt_s = gt.drop_duplicates("fid").set_index("fid")["true_label"]

    groups = {}
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "predictions_*_seed*.csv"))):
        m = SEED_RE.match(os.path.basename(path))
        if not m:
            continue
        groups.setdefault(m.group("base"), []).append((m.group("seed"), path))

    rows = []
    for base, items in sorted(groups.items()):
        f1s = []
        for _, path in items:
            p = pd.read_csv(path)[["fid", "crop_name"]]
            merged = p.merge(gt_s.rename("true_label"), left_on="fid", right_index=True)
            f1s.append(f1_score(merged["true_label"], merged["crop_name"], average="macro"))
        f1s = np.array(f1s)
        rows.append({
            "Model": BASE_TO_DISPLAY.get(base, base),
            "base": base,
            "n_seeds": len(f1s),
            "F1_mean": round(f1s.mean(), 4),
            "F1_std": round(f1s.std(ddof=1), 4),
            "F1_min": round(f1s.min(), 4),
            "F1_max": round(f1s.max(), 4),
            "F1_seeds": ";".join(f"{v:.4f}" for v in f1s),
        })

    if not rows:
        print("No per-seed prediction files found "
              "(run the modified deep inference scripts first).")
        return

    df = pd.DataFrame(rows).sort_values("F1_mean", ascending=False)
    out = os.path.join(RESULTS_DIR, "seed_f1_macro.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
