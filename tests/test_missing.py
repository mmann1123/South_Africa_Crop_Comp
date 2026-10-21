"""Unit tests for deep_learn/src/missing.py (no real data needed).

Run from the repo root, in any env with numpy/pandas/pyarrow:
  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deep_learn", "src"))
import missing  # noqa: E402

NAN = np.nan


class FillMethods(unittest.TestCase):
    def setUp(self):
        os.environ.pop("HOLDOUT_FILL", None)
        self.df = pd.DataFrame({
            "fid": [1, 1, 2],
            "B2_August": [0.2, 0.3, 0.4],
            "B2_September": [NAN, 0.5, NAN],
            "B2_October": [0.4, 0.6, 0.8],
        })
        self.cols = ["B2_August", "B2_September", "B2_October"]
        self.stats = pd.Series({"B2_August": 0.25, "B2_September": 0.9, "B2_October": 0.5})

    def test_zero(self):
        out = missing.fill_missing(self.df, self.cols, published="zero", method="zero")
        self.assertEqual(out["B2_September"].tolist(), [0.0, 0.5, 0.0])

    def test_input_not_modified(self):
        before = self.df.copy()
        missing.fill_missing(self.df, self.cols, published="zero", method="train_median", stats=self.stats)
        pd.testing.assert_frame_equal(self.df, before)

    def test_self_median_uses_frame_statistics(self):
        out = missing.fill_missing(self.df, self.cols, published="zero", method="self_median")
        self.assertEqual(out["B2_September"].tolist(), [0.5, 0.5, 0.5])

    def test_self_median_all_missing_column_falls_back_to_zero(self):
        df = self.df.assign(B2_September=NAN)
        out = missing.fill_missing(df, self.cols, published="zero", method="self_median")
        self.assertEqual(out["B2_September"].tolist(), [0.0, 0.0, 0.0])

    def test_train_median_uses_given_stats_not_frame(self):
        out = missing.fill_missing(self.df, self.cols, published="zero", method="train_median", stats=self.stats)
        self.assertEqual(out["B2_September"].tolist(), [0.9, 0.5, 0.9])
        self.assertEqual(out["B2_August"].tolist(), [0.2, 0.3, 0.4])  # observed values untouched

    def test_train_mean_same_mechanism(self):
        out = missing.fill_missing(self.df, self.cols, published="zero", method="train_mean", stats=self.stats)
        self.assertEqual(out["B2_September"].tolist(), [0.9, 0.5, 0.9])

    def test_missing_statistic_raises(self):
        with self.assertRaises(KeyError):
            missing.fill_missing(self.df, self.cols, published="zero", method="train_median",
                                 stats=self.stats.drop("B2_September"))

    def test_interp_between_neighbours(self):
        out = missing.fill_missing(self.df, self.cols, published="zero", method="interp", stats=self.stats)
        np.testing.assert_allclose(out["B2_September"].to_numpy(), [0.3, 0.5, 0.6])

    def test_interp_uses_calendar_positions(self):
        # April (3) and August (7) observed, July (6) missing; May/June absent from columns.
        df = pd.DataFrame({"B2_April": [0.0], "B2_July": [NAN], "B2_August": [0.4]})
        out = missing.fill_missing(df, list(df.columns), published="zero", method="interp",
                                   stats=pd.Series({"B2_July": 9.0}))
        self.assertAlmostEqual(out.loc[0, "B2_July"], 0.3)

    def test_interp_edge_falls_back_to_stats(self):
        df = pd.DataFrame({"B2_January": [NAN], "B2_February": [0.2], "B2_March": [0.3]})
        out = missing.fill_missing(df, list(df.columns), published="zero", method="interp",
                                   stats=pd.Series({"B2_January": 0.7}))
        self.assertAlmostEqual(out.loc[0, "B2_January"], 0.7)

    def test_interp_keeps_bands_separate(self):
        df = pd.DataFrame({"B2_August": [0.2], "B2_September": [NAN], "B2_October": [0.4],
                           "EVI_August": [10.0], "EVI_September": [NAN], "EVI_October": [30.0]})
        out = missing.fill_missing(df, list(df.columns), published="zero", method="interp", stats=pd.Series(dtype=float))
        self.assertAlmostEqual(out.loc[0, "B2_September"], 0.3)
        self.assertAlmostEqual(out.loc[0, "EVI_September"], 20.0)

    def test_interp_matches_rowwise_reference(self):
        # Random gap patterns (interior runs, edges, all-missing rows) against np.interp per row.
        rng = np.random.default_rng(0)
        months = ["January", "February", "March", "April", "July", "August", "September",
                  "October", "November", "December"]
        pos = np.array([missing.MONTHS.index(m) for m in months], dtype=float)
        cols = [f"B2_{m}" for m in months] + [f"EVI_{m}" for m in months]
        v = rng.normal(size=(500, len(cols)))
        v[rng.random(v.shape) < 0.35] = NAN
        v[0, :10] = NAN  # whole band missing in one row
        df = pd.DataFrame(v, columns=cols)
        stats = pd.Series(-99.0, index=cols)
        out = missing.fill_missing(df, cols, published="zero", method="interp", stats=stats)
        ref = v.copy()
        for b in range(2):
            sl = slice(10 * b, 10 * b + 10)
            for r in range(len(v)):
                row = v[r, sl]
                obs = ~np.isnan(row)
                if obs.sum() >= 2:
                    inside = (pos > pos[obs].min()) & (pos < pos[obs].max()) & ~obs
                    ref[r, sl][inside] = np.interp(pos[inside], pos[obs], row[obs])
        ref = np.where(np.isnan(ref), -99.0, ref)
        np.testing.assert_allclose(out.to_numpy(), ref, rtol=0, atol=1e-12)

    def test_interp_skips_months_never_observed_in_training(self):
        # May is missing in every training pixel; it must get the training value (0), not a
        # value interpolated between April and July.
        df = pd.DataFrame({"B2_April": [0.2], "B2_May": [NAN], "B2_July": [0.8], "B2_June": [NAN]})
        cols = ["B2_April", "B2_May", "B2_June", "B2_July"]
        out = missing.fill_missing(df, cols, published="zero", method="interp",
                                   stats=pd.Series({"B2_May": 0.0, "B2_June": 0.0}),
                                   observed={"B2_April", "B2_July", "B2_June"})
        self.assertEqual(out.loc[0, "B2_May"], 0.0)
        self.assertAlmostEqual(out.loc[0, "B2_June"], 0.2 + 0.6 * (5 - 3) / (6 - 3))

    def test_interp_non_band_month_columns_use_stats(self):
        df = self.df.assign(SHAPE_AREA=[NAN, 1.0, 2.0])
        out = missing.fill_missing(df, self.cols + ["SHAPE_AREA"], published="zero", method="interp",
                                   stats=pd.concat([self.stats, pd.Series({"SHAPE_AREA": 7.0})]))
        self.assertEqual(out["SHAPE_AREA"].tolist(), [7.0, 1.0, 2.0])
        np.testing.assert_allclose(out["B2_September"].to_numpy(), [0.3, 0.5, 0.6])


class MethodResolution(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"B2_September": [NAN, 0.2, 0.4]})

    def tearDown(self):
        os.environ.pop("HOLDOUT_FILL", None)

    def test_published_uses_the_scripts_rule(self):
        z = missing.fill_missing(self.df, ["B2_September"], published="zero", method="published")
        m = missing.fill_missing(self.df, ["B2_September"], published="self_median", method="published")
        self.assertEqual(z["B2_September"].tolist()[0], 0.0)
        self.assertAlmostEqual(m["B2_September"].tolist()[0], 0.3)

    def test_default_is_published(self):
        out = missing.fill_missing(self.df, ["B2_September"], published="self_median")
        self.assertAlmostEqual(out["B2_September"].tolist()[0], 0.3)

    def test_env_var_overrides_default(self):
        os.environ["HOLDOUT_FILL"] = "zero"
        out = missing.fill_missing(self.df, ["B2_September"], published="self_median")
        self.assertEqual(out["B2_September"].tolist()[0], 0.0)

    def test_explicit_argument_beats_env_var(self):
        os.environ["HOLDOUT_FILL"] = "zero"
        out = missing.fill_missing(self.df, ["B2_September"], published="zero", method="self_median")
        self.assertAlmostEqual(out["B2_September"].tolist()[0], 0.3)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            missing.fill_missing(self.df, ["B2_September"], published="zero", method="mode")

    def test_bad_published_raises(self):
        with self.assertRaises(ValueError):
            missing.fill_missing(self.df, ["B2_September"], published="train_median")


class TrainStats(unittest.TestCase):
    def test_compute_cache_and_reject_all_missing_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "train.parquet")
            pd.DataFrame({"fid": [1, 2, 3, 4], "B2_September": [1.0, 2.0, 3.0, 10.0],
                          "B2_May": [NAN] * 4, "crop_name": list("abcd")}).to_parquet(p)
            med = missing.train_stats("median", ["B2_September"], path=p, cache_dir=d)
            self.assertEqual(med["B2_September"], 2.5)
            self.assertTrue(os.path.exists(os.path.join(d, "train_medians.csv")))
            mean = missing.train_stats("mean", ["B2_September"], path=p, cache_dir=d)
            self.assertEqual(mean["B2_September"], 4.0)
            self.assertNotIn("fid", missing.train_stats("median", path=p, cache_dir=d).index)
            # never observed in training -> 0, the value training pipelines fed the models
            self.assertEqual(missing.train_stats("median", ["B2_May"], path=p, cache_dir=d)["B2_May"], 0.0)
            self.assertEqual(missing.train_observed(path=p, cache_dir=d), {"B2_September"})
            # the cached values are bit-identical to a fresh computation
            pd.DataFrame({"B2_September": [0.1 + 0.2, 1 / 3, 2 / 3, 0.7]}).to_parquet(p)
            for f in ("train_medians.csv", "train_means.csv"):
                os.remove(os.path.join(d, f))
            fresh = missing.train_stats("mean", ["B2_September"], path=p, cache_dir=d)["B2_September"]
            cached = missing.train_stats("mean", ["B2_September"], path=p, cache_dir=d)["B2_September"]
            self.assertEqual(fresh, cached)
            self.assertEqual(cached, pd.Series([0.1 + 0.2, 1 / 3, 2 / 3, 0.7]).mean())
            with self.assertRaises(KeyError):  # not in the training file at all
                missing.train_stats("median", ["B2_June"], path=p, cache_dir=d)


if __name__ == "__main__":
    unittest.main()
