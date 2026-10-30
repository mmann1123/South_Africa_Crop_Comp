import pandas as pd
import pytest

from experiments.external_audits.wijesingha import wijesingha_lib as lib


def test_bimonth_window_crosses_no_year_for_march_to_october():
    assert lib.bimonth_window(2012, 3) == ("2012-03-01", "2012-05-01")
    assert lib.bimonth_window(2012, 9) == ("2012-09-01", "2012-11-01")
    assert lib.bimonth_window(2012, 11) == ("2012-11-01", "2013-01-01")


def test_count_columns_order():
    cols = lib.count_columns([3, 5])
    assert cols[:4] == ["n_L5_m03", "n_L7_m03", "n_L8_m03", "n_all_m03"] and len(cols) == 8


def _frame():
    d = {"n_all_m03": [2, 0, 1, 3], "n_all_m05": [1, 1, 0, 2]}
    for m in ("03", "05"):
        for s in lib.SENSOR_ORDER:
            d[f"n_{s}_m{m}"] = [1, 0, 1, 1]
    return pd.DataFrame(d)


def test_dropna_flags_any_empty_bimonth_drops_row():
    f = lib.dropna_flags(_frame(), [3, 5])
    assert f["dropped"].tolist() == [False, True, True, False]
    assert f["empty_m03"].sum() == 1 and f["empty_m05"].sum() == 1


def test_summarise_group_and_contrast():
    f = lib.dropna_flags(_frame(), [3, 5])
    s = lib.summarise_group(f, [3, 5])
    assert s["dropped_share"] == pytest.approx(0.5) and s["empty_share_m03"] == pytest.approx(0.25)
    assert s["mean_obs_m03"] == pytest.approx(1.5) and s["mean_obs_all_bimonths"] == pytest.approx(1.25)
    tab = pd.DataFrame(
        [{"region": "NH", "year": 2010, "dropped_share": 0.4, "mean_obs_all_bimonths": 1.0},
         {"region": "NH", "year": 2014, "dropped_share": 0.1, "mean_obs_all_bimonths": 3.0}]
    )
    c = lib.period_contrast(tab, [2014], [2010]).iloc[0]
    assert c["dropped_test_mean"] == 0.4 and c["obs_train_mean"] == 3.0
