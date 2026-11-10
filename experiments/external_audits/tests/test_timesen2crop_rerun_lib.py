import numpy as np

from experiments.external_audits.timesen2crop_rerun import tslib as tl


def test_month_index_starts_in_september():
    assert tl.month_index(np.array([20170904, 20171201, 20180101, 20180831])).tolist() == [0, 3, 4, 11]


def test_composite_median_of_clear_only_and_empty_is_nan():
    dates = np.array([20170904, 20170914, 20170924, 20171004])
    X = np.array([[[100] * 9, [200] * 9, [9000] * 9, [400] * 9]], dtype=np.int16)
    flag = np.array([[0, 0, 1, 1]], dtype=np.int8)
    C = tl.composite(X, flag, dates)
    assert np.isclose(C[0, 0, 0], 150e-4)      # cloudy 9000 is excluded
    assert np.isnan(C[0, 1]).all()             # October's only acquisition is cloud
    assert np.isnan(C[0, 2]).all()             # no acquisition at all


def _one(values):
    return np.array(values, np.float32).reshape(1, 12, 1)


def test_fill_zero_and_median():
    C = _one([1, np.nan] + [2] * 10)
    assert tl.fill_zero(C)[0, 1, 0] == 0
    med = np.arange(12, dtype=np.float32).reshape(12, 1)
    assert tl.fill_median(C, med)[0, 1, 0] == 1


def test_fill_linear_interior_and_edges():
    n = np.nan
    C = _one([n, 1, n, n, 4, n, n, n, n, n, 6, n])
    out = tl.fill_linear(C)[0, :, 0]
    assert np.allclose(out[:5], [1, 1, 2, 3, 4])   # leading edge takes nearest, interior is linear
    assert np.isclose(out[7], 5)
    assert out[11] == 6                            # trailing edge takes nearest
    assert not np.isnan(out).any()


def test_fill_linear_all_empty_pixel_stays_zero():
    assert (tl.fill_linear(_one([np.nan] * 12)) == 0).all()


def test_fill_linear_keeps_present_values():
    C = _one(list(range(12)))
    assert np.array_equal(tl.fill_linear(C), C)


def test_apply_pattern():
    C = _one([1.0] * 12)
    keep = np.ones((1, 12), bool)
    keep[0, 3] = False
    assert np.isnan(tl.apply_pattern(C, keep)[0, 3, 0]) and C[0, 3, 0] == 1
