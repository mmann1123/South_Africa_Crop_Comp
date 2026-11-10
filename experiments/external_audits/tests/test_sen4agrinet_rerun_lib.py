import numpy as np
import pandas as pd
import pytest

from experiments.external_audits.sen4agrinet_rerun import evalcore as ec
from experiments.external_audits.sen4agrinet_rerun import rerunlib as rl

scipy_interp = pytest.importorskip("scipy.interpolate")


def test_month_of_right_closed_bins():
    t = pd.DatetimeIndex(["2019-01-01T00:00", "2019-01-01T10:50", "2019-02-01T00:00", "2019-02-01T10:50", "2019-12-31T10:50"])
    assert rl.month_of(t, 2019).tolist() == [-1, 0, 0, 1, 11]


@pytest.mark.parametrize("valid", [
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])
def test_fill_weights_match_scipy_extrapolating_interp1d(valid):
    valid = np.array(valid, bool)
    x = rl.month_starts_days(2019)
    y = np.random.default_rng(0).normal(size=12)
    f = scipy_interp.interp1d(x[valid], y[valid], kind="linear", fill_value="extrapolate", bounds_error=False)
    want = np.where(valid, y, f(x))
    got = rl.fill_weights(valid, x) @ np.where(valid, y, 0)
    assert np.allclose(got, want)


def test_monthly_filled_deletion_and_presence():
    t = pd.DatetimeIndex([f"2019-{m:02d}-10" for m in range(1, 13)])
    data = np.arange(12, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    months = rl.month_of(t, 2019)
    keep = months != 4  # delete May
    out, has = rl.monthly_filled(data, months, 2019, keep=keep)
    assert not has[4].any() and has[3].all()
    assert out[4, 0, 0] == pytest.approx(4.0, abs=0.05)  # time-weighted line through April (3) and June (5)


def test_to_subpatches_matches_export_order():
    a = np.arange(366 * 366).reshape(366, 366)
    s = rl.to_subpatches(a)
    assert s.shape == (36, 61, 61)
    assert s[0, 0, 0] == a[0, 0] and s[1, 0, 0] == a[0, 61] and s[6, 0, 0] == a[61, 0]


def test_authors_confusion_truncation_and_background_ties():
    # two pixels; class 2 has p 0.5 (log p > -1), class 1 has p 0.45: both truncate to 0 -> exp 1 -> tie -> lowest index
    lp = np.log(np.array([[0.05, 0.45, 0.5], [0.6, 0.1, 0.3]]))  # second pixel: background wins on float argmax too
    logp = __import__("torch").tensor(lp.T[None, :, :, None], dtype=__import__("torch").float32)  # (1, K=3, 2, 1)
    torch = __import__("torch")
    label = torch.tensor([[[2], [1]]])
    parcels = torch.tensor([[[True], [True]]])
    cm = ec.authors_confusion(logp, label, parcels, 3)
    # pixel 1: classes 0 (log 0.05 = -3.0 -> -3) , 1 (-0.8 -> 0), 2 (-0.69 -> 0): tie between 1 and 2 -> predicted 1
    assert cm[2, 1] == 1
    # pixel 2: class 0 log 0.6 = -0.51 -> 0, the lowest index wins -> predicted background
    assert cm[1, 0] == 1
    cc = ec.corrected_confusion(logp, label, parcels, 3)
    assert cc[2, 2] == 1 and cc[1, 0] == 1


def test_weighted_f1_drops_background_in_authors_metric():
    cm = np.zeros((3, 3), int)
    cm[1, 1], cm[2, 2], cm[1, 0] = 10, 10, 10
    assert ec.authors_weighted_f1(cm) == pytest.approx(1.0)
    assert ec.corrected_weighted_f1(cm) < 1.0
