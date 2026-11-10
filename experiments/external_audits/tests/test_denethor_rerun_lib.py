import importlib

import numpy as np

rf = importlib.import_module("experiments.external_audits.denethor_rerun.04_rf")
from experiments.external_audits.denethor_rerun import pflib  # noqa: E402


def test_authors_split_matches_randomstate0():
    tr, va = pflib.authors_split(100)
    idx = list(range(100))
    np.random.RandomState(0).shuffle(idx)
    assert va.tolist() == idx[:25] and tr.tolist() == idx[25:]


def test_qa_channel_and_permutation_preserve_values():
    qa = np.array([[[1, 100]], [[50, -1000]]], np.int16)  # T=2, F=1, P=2
    q = pflib.qa_channel(qa)
    assert np.allclose(q[:, 0, 0], [0.01, 0.5]) and q[1, 0, 1] == 1.0
    p = pflib.permuted_qa(np.random.default_rng(0).random((30, 3, 4)).astype(np.float32))
    assert p.shape == (30, 3, 4)


def test_thin_by_month_caps_counts():
    month = np.array([1, 1, 1, 2, 2, 3])
    clear = np.ones((2, 6), bool)
    target = np.zeros((2, 12), int)
    target[:, 0], target[:, 1], target[:, 2] = 1, 2, 0
    out = rf.thin_by_month(clear, month, target, np.random.default_rng(0))
    mc = rf.month_counts(out, month)
    assert mc[:, 0].tolist() == [1, 1] and mc[:, 1].tolist() == [2, 2] and mc[:, 2].tolist() == [0, 0]


def test_thin_random_drops_exact_number():
    clear = np.ones((3, 10), bool)
    out = rf.thin_random(clear, np.array([0, 3, 20]), np.random.default_rng(0))
    assert out.sum(1).tolist() == [10, 7, 0]


def test_features_shape_and_clear_selection():
    X = np.random.default_rng(0).uniform(100, 3000, (2, 6, 12)).astype(np.float32)
    clear = np.array([[1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0]], bool)
    f = rf.features(X, clear)
    assert f.shape == (2, 91)
    assert np.isclose(f[0, 13], (X[0, clear[0], 0] * 1e-4).min())  # min block starts after 13 max values
