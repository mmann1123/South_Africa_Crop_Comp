import numpy as np
import pandas as pd
import pytest

from experiments.external_audits.stoian import corr


def test_composition_index_zero_for_proportional_and_excludes_water():
    areas = pd.DataFrame({"class": ["A", "B", "WAT"], "t1": [10, 30, 1000], "t2": [20, 60, 5], "t3": [40, 120, 1]})
    c = corr.composition_index(areas, ["t1", "t2", "t3"])
    assert c.max() < 1e-12


def test_composition_index_is_one_for_disjoint_classes():
    areas = pd.DataFrame({"class": ["A", "B"], "t1": [10, 0], "t2": [0, 10]})
    assert corr.composition_index(areas, ["t1", "t2"])["t1"] == pytest.approx(1.0)


def test_rank_and_linear_monotone():
    x = np.arange(8.0)
    r = corr.rank_and_linear(x, x**3)
    assert r["spearman_rho"] == pytest.approx(1.0) and r["pearson_r"] < 1.0 and r["n"] == 8


def test_partial_spearman_removes_common_driver():
    rng = np.random.default_rng(42)
    z = rng.normal(size=200)
    x = z + 0.1 * rng.normal(size=200)
    y = z + 0.1 * rng.normal(size=200)
    assert corr.partial_spearman(x, y, z) < 0.3
    assert corr.partial_spearman(x, y, rng.normal(size=200)) > 0.9


def test_leave_one_out_range_brackets_full():
    x = np.array([1, 2, 3, 4, 5, 6.0])
    y = np.array([2, 1, 4, 3, 6, 5.0])
    lo, hi = corr.leave_one_out_range(x, y)
    assert lo <= hi and hi <= 1.0


def test_ols_r2_perfect_fit_and_share():
    x = np.arange(10.0)
    assert corr.ols_r2(3 * x + 1, x) == pytest.approx(1.0)
    a = pd.DataFrame({"class": ["A", "B", "WAT"], "t": [1, 3, 100]})
    assert corr.share_of_classes(a, ["t"], ["A"])["t"] == pytest.approx(0.25)
