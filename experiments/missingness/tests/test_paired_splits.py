"""PAIRED_FID_FILE mode of experiments/field_reduction/subsample.py: with the variable unset the
published splits are unchanged; set, every split function returns the file's FIDs (restricted to
the dataset's fields) and subsample_train_fids returns the stored draw."""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "field_reduction"))
import subsample  # noqa: E402


def toy(n_fields=40, px=3):
    rows = [{"fid": f, "crop_name": "abcde"[f % 5], "x": f * 1.0} for f in range(n_fields) for _ in range(px)]
    return pd.DataFrame(rows)


class PairedSplits(unittest.TestCase):
    def setUp(self):
        os.environ.pop("PAIRED_FID_FILE", None)
        subsample._PAIRED = None
        self.df = toy()

    def tearDown(self):
        os.environ.pop("PAIRED_FID_FILE", None)
        subsample._PAIRED = None

    def test_unset_reproduces_published_split(self):
        tr, va, te = subsample.get_fid_split_dl(self.df)
        self.assertEqual(len(tr) + len(va) + len(te), 40)
        tr2, _, _ = subsample.get_fid_split_base_ml(self.df)
        self.assertNotEqual(len(tr), len(tr2))  # 64/16/20 vs 68/12/20
        sub = subsample.subsample_train_fids(self.df, tr, 0.5, seed=1)
        self.assertEqual(len(sub), len(tr) // 2)

    def test_file_overrides_split_and_draw(self):
        d = {"train": list(range(0, 24)), "val": list(range(24, 32)), "test": list(range(32, 40)),
             "draws": {"0.25": {"7": [0, 5, 10, 15, 20, 23]}}}
        path = os.path.join(HERE, "_paired_toy.json")
        with open(path, "w") as f:
            json.dump(d, f)
        try:
            os.environ["PAIRED_FID_FILE"] = path
            for fn in (subsample.get_fid_split_dl, subsample.get_fid_split_base_ml):
                tr, va, te = fn(self.df)
                self.assertEqual(sorted(tr.tolist()), d["train"])
                self.assertEqual(sorted(va.tolist()), d["val"])
                self.assertEqual(sorted(te.tolist()), d["test"])
            sub = subsample.subsample_train_fids(self.df, tr, 0.25, seed=7)
            self.assertEqual(sub.tolist(), d["draws"]["0.25"]["7"])
            with self.assertRaises(KeyError):
                subsample.subsample_train_fids(self.df, tr, 0.5, seed=7)
            # a dataset lacking some fields gets the intersection, in file order
            small = self.df[self.df["fid"] < 20]
            tr, va, te = subsample.get_fid_split_dl(small)
            self.assertEqual(sorted(tr.tolist()), list(range(20)))
            self.assertEqual(len(va), 0)
            sub = subsample.subsample_train_fids(small, tr, 0.25, seed=7)
            self.assertEqual(sub.tolist(), [0, 5, 10, 15])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
