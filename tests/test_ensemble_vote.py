"""Unit tests for the majority vote in experiments/missingness/ensemble.py.
Run: python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "missingness"))
import ensemble as en  # noqa: E402

D, T = "L-TAE, pixel", "XGBoost, field"          # priority: L-TAE (0.78) before XGBoost (0.70)
R, C = "Random Forest, pixel", "CNN-BiLSTM, pixel"  # 0.65, 0.72


class Vote(unittest.TestCase):
    def test_majority(self):
        Y = np.array([[1, 1, 2], [0, 2, 2]])
        np.testing.assert_array_equal(en.vote(Y, [D, T, R], 4), [1, 2])

    def test_two_way_tie_goes_to_higher_priority(self):
        Y = np.array([[3, 1]])
        np.testing.assert_array_equal(en.vote(Y, [T, D], 4), [1])  # D wins although listed second
        np.testing.assert_array_equal(en.vote(Y, [D, T], 4), [3])

    def test_three_way_tie(self):
        Y = np.array([[0, 1, 2]])  # R, C, T: C has the highest in-region F1 of the three
        np.testing.assert_array_equal(en.vote(Y, [R, C, T], 4), [1])

    def test_tie_ignores_non_tied_priority(self):
        # D (top priority) votes for a minority class; the 2-2 tie is between C's and T's classes
        Y = np.array([[0, 1, 1, 2, 2]])
        np.testing.assert_array_equal(en.vote(Y, [D, C, R, T, "LightGBM, field"], 4), [1])

    def test_order_invariant(self):
        rng = np.random.default_rng(0)
        names = list(en.MEMBERS)[:6]
        Y = rng.integers(0, 5, (500, 6))
        perm = rng.permutation(6)
        np.testing.assert_array_equal(en.vote(Y, names, 5), en.vote(Y[:, perm], [names[i] for i in perm], 5))


if __name__ == "__main__":
    unittest.main()
