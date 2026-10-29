import numpy as np
import pandas as pd

from experiments.external_audits.kerner import kernerlib as kl
from experiments.external_audits.kerner import kernersummary as ks


def test_decode_states():
    a = ks.decode_states(["012", "210"])
    assert a.shape == (2, 3) and a.tolist() == [[0, 1, 2], [2, 1, 0]]


def _sample(n=4):
    return pd.DataFrame({"pid": [f"p{i}" for i in range(n)], "class": ["corn", "corn", "soy", "other"][:n], "greenup": [100] * n, "peak": [150] * n, "senescence": [200] * n})


def test_input_table_growth_and_fixed():
    acq = np.array([100, 150, 200, 153])
    states = np.array([[1, 2, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]])
    t = ks.input_table(_sample(), states, acq, 2017, offset=0)
    g = t[(t.basis == "growth") & (t.stage == "greenup")].set_index("pid")
    assert g.loc["p0", "n_clear"] == 1 and g.loc["p0", "exact"] == kl.CLEAR
    assert g.loc["p1", "n_clear"] == 0 and g.loc["p1", "exact"] == kl.FLAGGED
    assert g.loc["p2", "exact"] == kl.NODATA
    f = t[(t.basis == "fixed") & (t.stage == "early")].set_index("pid")  # fixed DOY 153 in 2017, window 145-161 holds 150 and 153
    assert f.loc["p0", "n_clear"] == 1 and f.loc["p3", "n_clear"] == 2
    assert (f["target_doy"] == 153).all()
    # offset shifts the growth target
    t1 = ks.input_table(_sample(), states, acq, 2017, offset=1)
    assert (t1[(t1.basis == "growth") & (t1.stage == "peak")]["target_doy"] == 151).all()


def test_aggregate_shares_and_pooling():
    acq = np.array([100, 150, 200])
    rng = np.random.default_rng(0)
    s = pd.concat([_sample(4).assign(pid=lambda d: d.pid + f"_{i}") for i in range(10)], ignore_index=True)
    states = np.tile(np.array([[1, 2, 0]]), (len(s), 1))
    t = ks.input_table(s, states, acq, 2018, 0)
    agg = ks.aggregate(t, n_boot=50).set_index(["basis", "stage", "class"])
    r = agg.loc[("growth", "greenup", "all")]
    assert r["share_zero_clear"] == 0.0 and r["share_exact_clear"] == 1.0 and r["n_pixels"] == 40
    r = agg.loc[("growth", "peak", "corn")]
    assert r["share_zero_clear"] == 1.0
    assert agg.loc[("growth", "peak", "all"), "share_exact_cloudy"] == 1.0
    assert agg.loc[("growth", "senescence", "all"), "share_exact_no_acq"] == 1.0
    assert ("growth", "all", "all") in agg.index
    assert agg.loc[("growth", "all", "all"), "n_inputs"] == 120


def test_diff_ci_and_test_minus_train():
    d, lo, hi = ks.diff_ci(np.ones(100), np.zeros(100), n_boot=100)
    assert (d, lo, hi) == (1.0, 1.0, 1.0)
    acq = np.array([100])
    parts = []
    for y, st in ((2017, 1), (2018, 1), (2019, 0)):
        s = _sample(4).assign(pid=lambda d, y=y: d.pid + f"_{y}")
        parts.append(ks.input_table(s, np.full((4, 1), st), acq, y, 0))
    t = ks.test_minus_train(pd.concat(parts, ignore_index=True), n_boot=50).set_index(["basis", "stage", "class"])
    r = t.loc[("growth", "greenup", "all")]
    assert r["zero_clear_train"] == 0.0 and r["zero_clear_test2019"] == 1.0 and r["zero_clear_diff"] == 1.0
