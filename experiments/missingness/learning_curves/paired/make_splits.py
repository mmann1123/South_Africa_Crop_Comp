"""Write the shared FID split and subsample draws for the paired low-data sweep.

The earlier 5 to 15 percent learning curves drew training fields per model (each sweep script
rebuilt its own split: 64/16/20 for the deep models and XGBoost, 68/12/20 for logistic
regression, a 0.1 validation share for CNN-BiLSTM, and FID orders that differ between the pixel
and feature parquets), so draws did not pair across models. This script fixes one split and one
set of draws that every model then reads through PAIRED_FID_FILE
(experiments/field_reduction/subsample.py).

Split: get_fid_split_dl on data/merged_dl_train.parquet (the published deep-model split).
Draws: subsample_train_fids (stratified by crop) on the same frame, fractions 0.05, 0.10, 0.15,
seeds 42 101 202 303 404 505 606 707 808 909, so the deep models' draws equal the ones already run.
Checks that the classical feature table (final_data.parquet) holds the same fields.
Output: splits.json next to this script. Run in the deep_field env.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "field_reduction"))
from config import FINAL_DATA_PATH, MERGED_DL_PATH  # noqa: E402
from subsample import get_fid_split_dl, subsample_train_fids  # noqa: E402

FRACTIONS = [0.05, 0.10, 0.15]
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
OUT = os.path.join(HERE, "splits.json")


def main():
    if os.environ.get("PAIRED_FID_FILE"):
        sys.exit("unset PAIRED_FID_FILE before building the file")
    df = pd.read_parquet(MERGED_DL_PATH, columns=["fid", "crop_name"])
    tr, va, te = get_fid_split_dl(df)
    draws = {}
    for frac in FRACTIONS:
        draws[f"{frac:.2f}"] = {str(s): [int(f) for f in subsample_train_fids(df, tr, frac, seed=s)] for s in SEEDS}
        n = {len(v) for v in draws[f"{frac:.2f}"].values()}
        print(f"fraction {frac:.2f}: {n} fields per draw")
    feat = pd.read_parquet(FINAL_DATA_PATH, columns=["fid"])
    ff = set(feat["fid"].unique())
    allf = set(tr) | set(va) | set(te)
    print(f"fields: {len(allf)} in merged_dl_train, {len(ff)} in final_data, {len(allf - ff)} missing from final_data, "
          f"{len(ff - allf)} extra in final_data")
    out = {"source": os.path.relpath(MERGED_DL_PATH, REPO), "split": "get_fid_split_dl (test 0.2, then val 0.2, random_state 42)",
           "train": [int(f) for f in tr], "val": [int(f) for f in va], "test": [int(f) for f in te], "draws": draws}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"{len(tr)} train, {len(va)} val, {len(te)} test -> {OUT}")


if __name__ == "__main__":
    main()
