"""Regression tests for missing.py on the real holdout data (skipped if the data are absent).

1. "published" reproduces the original inline code bit for bit, for both original rules
   (0 fill for raw-sequence models; holdout median then 0 for pixel TabNet).
2. "train_median" reproduces the September-filled holdout used by
   experiments/holdout_fill_check (make_filled_holdout.py), so results produced through the
   pipeline flag match the published re-analysis.

Uses the first N_ROWS holdout pixels plus every pixel of 50 fields with September missing,
to keep memory and time low. Run from the repo root (deep_field env):
  python -m unittest tests.test_fill_regression -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
import missing  # noqa: E402
from config import MERGED_DL_TEST_PATH, MODEL_DIR, TABNET_DIR  # noqa: E402

FILLED = os.path.join(REPO, "experiments", "holdout_fill_check", "data", "merged_dl_test_trainmedian.parquet")
N_ROWS = 300_000


def _sample(path, cols):
    t = pq.read_table(path, columns=["fid"] + cols).to_pandas()
    gap = t["fid"][t["B2_September"].isna()].unique()[:50]
    keep = (np.arange(len(t)) < N_ROWS) | t["fid"].isin(gap).to_numpy()
    return t[keep]


@unittest.skipUnless(os.path.exists(MERGED_DL_TEST_PATH), "holdout parquet not available")
class PublishedReproducesOriginalCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from joblib import load
        os.environ.pop("HOLDOUT_FILL", None)
        cls.seq_cols = load(os.path.join(MODEL_DIR, "ltae_feature_columns.joblib"))
        tab = load(os.path.join(TABNET_DIR, "tabnet_feature_columns.joblib"))
        names = set(pq.ParquetFile(MERGED_DL_TEST_PATH).schema.names)
        cls.tab_cols = [c for c in tab if not c.startswith("Type_") and c in names]
        cls.df = _sample(MERGED_DL_TEST_PATH, sorted(set(cls.seq_cols) | set(cls.tab_cols)))

    def test_sample_contains_september_gaps(self):
        self.assertGreater(self.df["B2_September"].isna().sum(), 1000)

    def test_zero_rule(self):
        old = self.df[self.seq_cols].fillna(0)
        new = missing.fill_missing(self.df, self.seq_cols, published="zero")
        pd.testing.assert_frame_equal(new, old)

    def test_tabnet_self_median_rule(self):
        x = self.df[self.tab_cols]
        old = x.fillna(x.median()).fillna(0)
        new = missing.fill_missing(self.df, self.tab_cols, published="self_median")
        pd.testing.assert_frame_equal(new, old)


@unittest.skipUnless(os.path.exists(MERGED_DL_TEST_PATH) and os.path.exists(FILLED),
                     "holdout or September-filled holdout not available")
class TrainMedianReproducesReanalysis(unittest.TestCase):
    def test_matches_filled_holdout(self):
        from joblib import load
        cols = load(os.path.join(MODEL_DIR, "ltae_feature_columns.joblib"))
        orig = _sample(MERGED_DL_TEST_PATH, cols)
        ref = pq.read_table(FILLED, columns=["fid"] + cols).to_pandas().loc[orig.index]
        new = missing.fill_missing(orig, cols, published="zero", method="train_median")
        np.testing.assert_array_equal(new.to_numpy(), ref[cols].fillna(0).to_numpy())
        self.assertFalse(new.isna().any().any())


if __name__ == "__main__":
    unittest.main()
