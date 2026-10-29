import numpy as np
import pytest

from experiments.external_audits.kerner import kernerlib as k


def test_pixel_to_xy_centres():
    x, y = k.pixel_to_xy(0, 0)
    assert (x, y) == (199980 + 15, 4600020 - 15)
    x, y = k.pixel_to_xy(3659, 3659)
    assert x == 199980 + 3659.5 * 30 and y == 4600020 - 3659.5 * 30


def test_parse_acquisition():
    assert k.parse_acquisition("S_T16TBL_20180720T163839") == ("S", "16TBL", 2018, 201)
    assert k.parse_acquisition("L_T16TBL_20190101T163839")[3] == 1
    assert k.parse_acquisition("S_T16TBL_20160301T000000")[3] == 61  # leap year
    with pytest.raises(ValueError):
        k.parse_acquisition("garbage")


def test_stratified_sample_is_seeded_and_class_pure():
    labels = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]])
    valid = np.ones_like(labels, dtype=bool)
    valid[0, 0] = False
    a = k.stratified_sample(labels, valid, 2, seed=42)
    b = k.stratified_sample(labels, valid, 2, seed=42)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    r, c, cls = a
    assert len(r) == 6 and (labels[r, c] == cls).all() and valid[r, c].all()
    assert len(set(zip(r, c))) == 6


def test_stratified_sample_caps_at_available():
    labels = np.zeros((2, 2), dtype=int)
    r, c, cls = k.stratified_sample(labels, np.ones((2, 2), bool), 10, classes=(0,))
    assert len(r) == 4


def test_window_counts():
    acq = np.array([100, 105, 110, 120])
    states = np.array(
        [
            [1, 2, 1, 1],  # clear 100, flagged 105, clear 110, clear 120
            [2, 2, 2, 0],
            [0, 0, 0, 0],
        ]
    )
    nc, nf, ex = k.window_counts(states, acq, 105, half=8)
    assert nc.tolist() == [2, 0, 0]  # 100 and 110 in window
    assert nf.tolist() == [1, 3, 0]
    assert ex.tolist() == [k.FLAGGED, k.FLAGGED, k.NODATA]
    # window 100..116 around 108: 100 (|8|, inclusive edge) and 110 count, 120 is 12 away
    nc, _, _ = k.window_counts(states, acq, 108, half=8)
    assert nc[0] == 2


def test_window_counts_per_pixel_target_and_exact_clear_wins():
    acq = np.array([100, 100, 150])
    states = np.array([[2, 1, 1], [2, 2, 1]])
    nc, nf, ex = k.window_counts(states, acq, np.array([100, 150]), half=8)
    assert nc.tolist() == [1, 1]
    assert ex.tolist() == [k.CLEAR, k.CLEAR]
    _, _, ex = k.window_counts(states, acq, np.array([100, 100]), half=8)
    assert ex.tolist() == [k.CLEAR, k.FLAGGED]  # clear beats flagged on the same day


def test_best_offset():
    off, share = k.best_offset(np.array([10, 20, 30, 40]), {11, 21, 31, 41})
    assert off == 1 and share[1] == 1.0 and share[0] == 0.0


def test_bootstrap_mean_ci_seeded_and_bounds():
    v = np.r_[np.ones(50), np.zeros(50)]
    m, lo, hi = k.bootstrap_mean_ci(v, n_boot=500, seed=42)
    assert m == 0.5 and lo < 0.5 < hi
    assert (m, lo, hi) == k.bootstrap_mean_ci(v, n_boot=500, seed=42)
    m2, lo2, hi2 = k.bootstrap_mean_ci(np.ones((30, 3)))
    assert (m2, lo2, hi2) == (1.0, 1.0, 1.0)
