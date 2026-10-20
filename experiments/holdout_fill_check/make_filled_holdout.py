"""Holdout pixel file with missing values filled from TRAINING-tile medians.

The holdout tile (data/merged_dl_test.parquet) is missing September for ~41% of pixels;
the training tiles have no missing values in any month used. The published inference
fills these with 0 (raw-input deep models) or with the holdout's own medians (pixel
TabNet). This writes a copy in which every missing value in a feature column that the
training data does observe is replaced by that column's median over training pixels,
so no holdout statistic is used. Columns missing in all training pixels (May) are left
as they are; every model drops them.

Output: experiments/holdout_fill_check/data/merged_dl_test_trainmedian.parquet
        experiments/holdout_fill_check/data/train_medians.csv
"""
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "deep_learn", "src"))
from config import MERGED_DL_PATH, MERGED_DL_TEST_PATH  # noqa: E402

META = {"id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name"}
OUT = os.path.join(HERE, "data", "merged_dl_test_trainmedian.parquet")

test = pd.read_parquet(MERGED_DL_TEST_PATH)
feat = [c for c in test.columns if c not in META]
need = [c for c in feat if test[c].isna().any()]
train = pq.read_table(MERGED_DL_PATH, columns=need).to_pandas()
med = train.median()
fill = med.dropna()
print("Columns with missing holdout values:", {c: round(test[c].isna().mean(), 3) for c in need})
print("Filled from training medians:", fill.round(4).to_dict())
print("Left missing (no training data):", sorted(set(need) - set(fill.index)))
test[fill.index] = test[fill.index].fillna(fill)
test.to_parquet(OUT, index=False)
med.rename("train_median").to_csv(os.path.join(HERE, "data", "train_medians.csv"))
print("Wrote", OUT, test.shape)
