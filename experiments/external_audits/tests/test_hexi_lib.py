import datetime as dt

import numpy as np
import pytest

from experiments.external_audits.hexi import hexilib as hl


def test_week_bounds_are_contiguous_sevens():
    b = [hl.week_bounds(2022, k) for k in range(52)]
    assert b[0][0] == dt.date(2022, 1, 1) and b[-1][1] == dt.date(2022, 12, 31)
    assert all(e - s == dt.timedelta(days=7) for s, e in b)
    assert all(b[i][1] == b[i + 1][0] for i in range(51))


def test_week_of_matches_bounds_and_drops_last_day():
    for k in range(52):
        s, e = hl.week_bounds(2021, k)
        assert hl.week_of(s) == k and hl.week_of(e - dt.timedelta(days=1)) == k
    assert hl.week_of(dt.date(2022, 12, 31)) is None


def test_qa60_cloud_bits():
    qa = np.array([0, 1024, 2048, 3072, 512, 1 << 9, 1024 | 1])
    assert hl.qa60_cloud(qa).tolist() == [False, True, True, True, False, False, True]


def test_longest_run():
    assert hl.longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert hl.longest_run(np.zeros(5, bool)) == 0


def test_gap_metrics_splits_interior_edge():
    v = np.zeros((2, 52), bool)
    v[0, [3, 4, 10]] = True  # edge 3 + 41, interior 5 (weeks 5-9 empty)
    m = hl.gap_metrics(v, season=(0, 51))
    assert m["n_valid"][0] == 3 and m["n_edge"][0] == 3 + 41 and m["n_interp"][0] == 5
    assert m["longest_interp"][0] == 5 and m["longest_gap"][0] == 41
    assert m["n_no_data"][1] == 1 and m["n_edge"][1] == 52 and m["n_interp"][1] == 0
    assert m["n_valid"].sum() + m["n_interp"].sum() + m["n_edge"].sum() == 52 * 2


def test_gap_metrics_season_window():
    v = np.ones((1, 52), bool)
    v[0, 20:24] = False
    m = hl.gap_metrics(v, season=(13, 43))
    assert m["season_valid"][0] == 31 - 4 and m["season_interp"][0] == 4 and m["season_longest_gap"][0] == 4


def test_summary_shares_add_up():
    rng = np.random.default_rng(0)
    v = rng.random((50, 52)) < 0.5
    s = hl.summary(v)
    assert s["empty_share"] == pytest.approx(s["interp_share"] + s["edge_share"])
    assert s["valid_weeks_mean"] / 52 == pytest.approx(1 - s["empty_share"])


def test_residual_cloud():
    q = np.array([[2, 0, 3, 1]])
    c = np.array([[2, 0, 1, 0]])
    r = hl.residual_cloud(q, c)
    assert r["obs_share"] == pytest.approx(1 - 3 / 6)
    assert r["week_any"] == pytest.approx(2 / 3) and r["week_all"] == pytest.approx(1 / 3)
    assert np.isnan(hl.residual_cloud(np.zeros((1, 3)), np.zeros((1, 3)))["obs_share"])
