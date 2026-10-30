import datetime as dt

import numpy as np
import pandas as pd
import pytest

from experiments.external_audits.hao import catalogue, haolib


def test_window_dates_cover_doy_60_to_330_without_gaps():
    for year in (2017, 2018):
        w = catalogue.window_dates(year)
        assert len(w) == 18 and w[0][0] == dt.date(year, 3, 1)
        assert w[-1][1] == dt.date(year, 1, 1) + dt.timedelta(days=329)  # DOY 330 is the exclusive end
        assert all(w[i][1] == w[i + 1][0] for i in range(17))
    assert catalogue.WINDOW_ENDS[0] == 75 and catalogue.WINDOW_ENDS[-1] == 330


def test_window_index_edges():
    assert haolib.window_index(60) == 0 and haolib.window_index(74) == 0 and haolib.window_index(75) == 1
    assert haolib.window_index(329) == 17 and haolib.window_index(330) is None and haolib.window_index(59) is None


def test_runs_and_longest_run():
    f = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert haolib.runs(f) == [(1, 2), (4, 4), (7, 9)]
    assert haolib.longest_run(f) == 3 and haolib.longest_run(np.zeros(4, bool)) == 0


def test_fill_classes_partition_empty_windows():
    v = np.zeros((3, 8), bool)
    v[0, [2, 5]] = True  # edge 0,1 ; interior 3,4 ; edge 6,7
    v[1, [1, 2, 4]] = True  # empty 3 is strictly fillable (valid at 2 and 4)
    # row 2 has no valid window: everything is edge
    c = haolib.fill_classes(v)
    assert (c["interior"] | c["edge"] == c["empty"]).all() and not (c["interior"] & c["edge"]).any()
    assert c["interior"][0].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert c["edge"][0].tolist() == [1, 1, 0, 0, 0, 0, 1, 1]
    assert c["strict"][1, 3] and not c["strict"][0, 3] and not c["strict"][0, 4]
    assert c["edge"][2].all()


def test_window_shares_and_prefix_metrics():
    v = np.ones((4, 18), bool)
    v[:2, 0:3] = False  # two points lose the first three windows
    s = haolib.window_shares(v)
    assert s["empty_share"].iloc[0] == 0.5 and s["empty_share"].iloc[5] == 0.0
    m = haolib.prefix_metrics(v, 5)
    assert m["doy_end"] == 135 and m["empty_share"] == pytest.approx(6 / 20) and m["share_points_run_ge2"] == 0.5
    assert m["edge_share"] == m["empty_share"]  # the lost windows lie at the series start
    assert haolib.prefix_metrics(v, 3)["share_points_no_obs"] == 0.5


def test_cap_per_cell_and_subsample_are_seeded_and_bounded():
    df = pd.DataFrame({"cell": [1] * 50 + [2] * 5, "crop": ["a"] * 55})
    a = haolib.cap_per_cell(df, "cell", "crop", 30, 42)
    b = haolib.cap_per_cell(df, "cell", "crop", 30, 42)
    assert a.index.tolist() == b.index.tolist() and (a["cell"] == 1).sum() == 30 and (a["cell"] == 2).sum() == 5
    s = haolib.subsample_to_counts(a, "crop", {"a": 10}, 42)
    assert len(s) == 10 and len(haolib.subsample_to_counts(a, "crop", {"a": 999}, 42)) == len(a)


def test_grid_cell_and_utm_zone():
    c = haolib.grid_cell(np.array([50_000.0, 150_000.0, -50_000.0]), np.array([10.0, 10.0, 10.0]))
    assert len(set(c.tolist())) == 3
    assert haolib.utm_zone([-100.5, -111.0, 116.0, 179.9]).tolist() == [14, 12, 50, 60]


def test_assign_chunks_bounded_and_zone_pure():
    rng = np.random.default_rng(0)
    lon = np.r_[rng.uniform(-101, -99, 300), rng.uniform(-111, -109, 100)]
    lat = rng.uniform(40, 42, 400)
    ch = haolib.assign_chunks(lon, lat, max_n=120)
    assert np.bincount(ch).max() <= 120
    for c in np.unique(ch):
        assert len(set(haolib.utm_zone(lon[ch == c]).tolist())) == 1


def test_combine_valid_reads_columns():
    df = pd.DataFrame({f"w{k:02d}_fm": [k % 2, 1] for k in range(18)})
    v = haolib.combine_valid(df, "fm")
    assert v.shape == (2, 18) and v[0].tolist() == [bool(k % 2) for k in range(18)] and v[1].all()


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    x = np.random.default_rng(1).random(200)
    m, lo, hi = haolib.bootstrap_ci(x)
    assert lo < m < hi and haolib.bootstrap_ci(x) == (m, lo, hi)


def test_bar_values_linear_from_label_rows():
    rows = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])  # 100 % at row 100, 0 % at row 600
    assert haolib.bar_values(rows, [600, 350, 100]) == pytest.approx([0.0, 50.0, 100.0])


def test_orange_bars_ignore_small_blobs():
    img = np.full((200, 100, 3), 255, np.uint8)
    img[100:190, 10:40] = (255, 108, 0)
    img[10:20, 60:75] = (255, 108, 0)  # legend square
    assert haolib.orange_bars(img) == [(10, 40, 100, 190)]


def test_spearman_perfect():
    r = haolib.spearman([1, 2, 3, 4], [2, 4, 6, 8])
    assert r["spearman_rho"] == pytest.approx(1.0) and r["n"] == 4


def test_driver_domain_year_and_valid_matrix():
    import importlib

    m = importlib.import_module("experiments.external_audits.hao.03_ee_windows")
    assert m.domain_year("CONUS") == ("CONUS", 2017) and m.domain_year("NE") == ("NE", 2018)
    assert m.domain_year("CONUS@2018") == ("CONUS", 2018) and m.domain_year("AB@2017") == ("AB", 2017)
    s = importlib.import_module("experiments.external_audits.hao.04_summarise_windows")
    df = pd.DataFrame({f"w{k:02d}_{c}": [1 if k < 9 else 0] for k in range(18) for c in ("l8_clear", "l8_obs", "s2_obs", "s2_qa60", "s2_cs65")})
    assert s.valid_matrix(df, "l8s2", "paper")[0].sum() == 9 and s.valid_matrix(df, "l8s2", "cs65")[0].sum() == 9
