import numpy as np

from experiments.external_audits.maleki import catalogue, interplib as il

GRID = il.grid_days()


def test_grid_days():
    assert len(GRID) == 60 and GRID[0] == 1 and GRID[-1] == 355 and np.all(np.diff(GRID) == 6)


def test_counts_and_monthly():
    doy = np.array([10, 40, 41, 200])
    v = np.array([[1, 1, 0, 1], [0, 0, 0, 0]], dtype=bool)
    assert il.valid_counts(v).tolist() == [3, 0]
    m = il.monthly_counts(v, doy)
    assert m.shape == (2, 12) and m[0, 0] == 1 and m[0, 1] == 1 and m[0, 6] == 1 and m[1].sum() == 0


def test_observed_window_edges():
    # day 4 is within 3 days of grid day 1 and 7; day 10 is within 3 of grid day 7 and 13
    v = np.array([[True]])
    obs = il.observed_steps(v, np.array([4]), GRID)
    assert obs[0, 0] and obs[0, 1] and obs[0].sum() == 2
    obs = il.observed_steps(v, np.array([5]), GRID)  # 5 is 4 from day 1, 2 from day 7, 8 from 13
    assert obs[0, 1] and obs[0].sum() == 1


def test_step_kinds_partition():
    doy = np.array([100, 200, 300])
    v = np.ones((1, 3), dtype=bool)
    obs, interp, extra = il.step_kinds(v, doy, GRID)
    assert (obs.astype(int) + interp + extra).max() == 1 and (obs | interp | extra).all()
    assert extra[0, 0] and extra[0, -1] and interp[0, 20] and not interp[0, 0]


def test_no_valid_pixel_is_all_extrapolated():
    v = np.zeros((1, 3), dtype=bool)
    obs, interp, extra = il.step_kinds(v, np.array([100, 200, 300]), GRID)
    assert extra.all() and not obs.any() and not interp.any()


def test_longest_run_and_gap():
    m = np.array([[1, 1, 0, 1, 1, 1, 0]], dtype=bool)
    assert il.longest_run(m).tolist() == [3]
    v = np.array([[True, True]])
    assert il.longest_gap_days(v, np.array([100, 130]), 1, 365).tolist() == [235]
    assert il.longest_gap_days(np.zeros((1, 2), bool), np.array([100, 130]), 1, 365).tolist() == [364]


def test_pixel_metrics_dense():
    doy = np.arange(1, 366, 2)
    v = np.ones((1, len(doy)), dtype=bool)
    m = il.pixel_metrics(v, doy, GRID)
    assert m["share_not_observed"][0] == 0 and m["longest_gap_days"][0] == 2 and m["n_valid"][0] == len(doy)


def test_pixel_metrics_sparse_share():
    doy = np.array([100, 200])
    v = np.ones((1, 2), dtype=bool)
    m = il.pixel_metrics(v, doy, GRID)
    # day 100 lies within 3 days of grid days 97 and 103, day 200 of 199 only
    assert m["obs_steps"][0] == 3 and m["interp_steps"][0] + m["extrap_steps"][0] == 57


def test_drops_and_table_shapes():
    # the paper's quoted ranges (-3.8 to -16.9 and -11.0 to -23.8) reproduce only against the SOURCE site's baseline
    d = il.drops(catalogue.TABLE4[("Camargue", "Champagne")], catalogue.TABLE5["Camargue"])
    assert len(d) == 12 and min(d) == 3.8 and max(d) == 16.9
    e = il.drops(catalogue.TABLE4[("Champagne", "Camargue")], catalogue.TABLE5["Champagne"])
    assert min(e) == 11.0 and max(e) == 23.8


def test_summary_and_join():
    import importlib

    mod = importlib.import_module("experiments.external_audits.maleki.04_summarise")
    paper = mod.paper_table()
    assert paper.loc[0, "train"] == "Camargue" and paper.loc[0, "n_configs"] == 12
    assert paper.loc[1, "drop_vs_source_baseline_mean"] > paper.loc[0, "drop_vs_source_baseline_mean"]
    import pandas as pd

    row = lambda s, sh: {"site": s, "mask": "paper_masks", "stratum": "all", "n_valid_median": 10, "share_not_observed_mean": sh, "longest_unobserved_steps_median": 3, "longest_gap_days_median": 30}
    t = mod.join(paper, pd.DataFrame([row("Camargue", 0.3), row("Champagne", 0.6)]))
    assert t["source_more_interpolated"].tolist() == [False, True]
