"""Raw holdout series filled by each HOLDOUT_FILL method, for rebuilding the xr_fresh features
of the classical models (experiments/holdout_fill_check/rebuild_xrfresh.py --variants ...).

Only months observed in the training tiles are filled. May is missing in every training and
holdout pixel, and the training xr_fresh features were computed with it missing (the NaN-aware
xr_fresh functions skip it), so it stays missing here; filling it would change the features
for a reason unrelated to the September gap.

Writes experiments/holdout_fill_check/data/merged_dl_test_fill_{method}.parquet (git-ignored).
Check: the train_median output must equal the re-analysis file merged_dl_test_trainmedian.parquet.
Run in the deep_field env: python experiments/missingness/make_filled_variants.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
import missing  # noqa: E402
from config import MERGED_DL_TEST_PATH  # noqa: E402

DATA = os.path.join(REPO, "experiments", "holdout_fill_check", "data")
METHODS = ["zero", "self_median", "train_mean", "train_median", "interp"]

if __name__ == "__main__":
    df = pd.read_parquet(MERGED_DL_TEST_PATH)
    stats = missing.train_stats("median")
    observed = [c for c in df.columns
                if missing._band_month(c) is not None and not c.endswith("_May") and c in stats.index]
    print(f"filling {len(observed)} band-month columns (May left missing)", flush=True)
    for method in METHODS:
        out = df.copy()
        out[observed] = missing.fill_missing(df, observed, published="zero", method=method)
        assert not out[observed].isna().any().any()
        assert out[[c for c in df.columns if c.endswith("_May")]].isna().all().all()
        path = os.path.join(DATA, f"merged_dl_test_fill_{method}.parquet")
        out.to_parquet(path, index=False)
        print("wrote", path, flush=True)
    ref = pd.read_parquet(os.path.join(DATA, "merged_dl_test_trainmedian.parquet"))
    new = pd.read_parquet(os.path.join(DATA, "merged_dl_test_fill_train_median.parquet"))
    same = ref[observed].equals(new[observed]) and ref.columns.equals(new.columns)
    print("train_median equals merged_dl_test_trainmedian.parquet:", same)
    sys.exit(0 if same else 1)
