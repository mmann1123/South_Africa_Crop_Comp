import math

import pandas as pd

from experiments.external_audits.masolele import masolele_lib as lib


def test_confluence_points_inclusive():
    pts = lib.confluence_points((0, 2), 1)
    assert len(pts) == 9 and (0, -1) in pts and (2, 1) in pts


def test_unit_box():
    assert lib.unit_box(10, -5, 0.05) == [9.95, -5.05, 10.05, -4.95]


def test_half_window():
    assert lib.half_window(2003, 1) == ("2003-01-01", "2003-07-01")
    assert lib.half_window(2003, 2) == ("2003-07-01", "2004-01-01")


def test_window_ids_and_columns():
    w = lib.window_ids([2000, 2001])
    assert w == ["2000h1", "2000h2", "2001h1", "2001h2"]
    assert lib.count_columns(w[:1]) == ["n_L5_2000h1", "n_L7_2000h1", "n_all_2000h1"]


def _wide():
    w = ["2000h1", "2000h2"]
    d = {"continent": ["A", "A", "B"], "point_id": [1, 2, 3]}
    d.update({"n_L5_2000h1": [1, 0, 0], "n_L7_2000h1": [1, 0, 0], "n_all_2000h1": [2, 0, 0]})
    d.update({"n_L5_2000h2": [0, 1, 0], "n_L7_2000h2": [3, 0, 2], "n_all_2000h2": [3, 1, 2]})
    return pd.DataFrame(d), w


def test_long_form_and_summary():
    df, w = _wide()
    lf = lib.long_form(df, w)
    assert len(lf) == 6 and lf["empty"].sum() == 2
    s = lib.continent_summary(lf).set_index("continent")
    assert s.loc["A", "empty_share"] == 0.25 and s.loc["A", "points_any_empty"] == 0.5
    assert s.loc["B", "mean_empty_windows"] == 1.0 and s.loc["B", "mean_obs_per_window"] == 1.0
    ws = lib.window_summary(lf)
    assert set(ws.columns) >= {"empty_share", "mean_obs"} and len(ws) == 4


def test_max_pairwise_ratio():
    assert lib.max_pairwise_ratio({"a": 0.1, "b": 0.4}) == 4.0
    assert math.isinf(lib.max_pairwise_ratio({"a": 0.0, "b": 0.4}))
