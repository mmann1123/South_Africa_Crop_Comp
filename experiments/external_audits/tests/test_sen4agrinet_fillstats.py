import numpy as np
import pandas as pd

from experiments.external_audits.sen4agrinet import fillstats


def mk_stack(months_present, year=2019):
    ts = [f"{year}-{m:02d}-15T10:50:00" for m in months_present]
    return ";".join(ts)


def test_patch_fill_table_counts():
    sample = pd.DataFrame([{"scenario": 2, "split": "test", "country": "France", "year": 2019, "tile": "31TCJ", "stem": "p1"}])
    stacks = pd.DataFrame([{"stem": "p1", "n_obs": 6, "times": mk_stack([2, 3, 6, 7, 9, 10])}])
    t = fillstats.patch_fill_table(sample, stacks).iloc[0]
    assert t["PAD_Apr-Sep_filled"] == 3 and t["PAD_Apr-Sep_interpolated"] == 3 and t["PAD_Apr-Sep_extrapolated"] == 0
    assert t["OAD_May-Nov_filled"] == 3 and t["OAD_May-Nov_extrapolated"] == 1
    assert t["present_02"] == 1 and t["present_04"] == 0


def test_split_summary_weights_by_composition():
    base = {"scenario": 2, "split": "test", "country": "France", "year": 2019}
    a = {**base, "tile": "A"}
    b = {**base, "tile": "B"}
    cols = {"PAD_Apr-Sep_filled": 0, "PAD_Apr-Sep_interpolated": 0, "PAD_Apr-Sep_extrapolated": 0, "OAD_May-Nov_filled": 0, "OAD_May-Nov_interpolated": 0, "OAD_May-Nov_extrapolated": 0, "n_obs": 30}
    pf = pd.DataFrame([{**a, "stem": "a1", **cols}, {**a, "stem": "a2", **cols}, {**b, "stem": "b1", **cols, "PAD_Apr-Sep_filled": 6, "PAD_Apr-Sep_interpolated": 6}, {**b, "stem": "b2", **cols, "PAD_Apr-Sep_filled": 6, "PAD_Apr-Sep_interpolated": 6}])
    comp = pd.DataFrame([{**a, "n_patches": 300}, {**b, "n_patches": 100}])
    s = fillstats.split_summary(pf, comp, boot=20)
    pad = s[s["window"] == "PAD_Apr-Sep"].iloc[0]
    assert np.isclose(pad["share_filled"], 0.25)  # tile B is 1/4 of the split and fully filled
    assert np.isclose(pad["share_patches_any_fill"], 0.25)
    assert pad["share_filled_ci_low"] <= pad["share_filled"] <= pad["share_filled_ci_high"] + 1e-9


def test_month_missing_and_obs():
    base = {"scenario": 3, "split": "train", "country": "France", "year": 2019, "tile": "A"}
    row = {f"present_{m:02d}": 1 for m in range(1, 13)}
    pf = pd.DataFrame([{**base, "stem": "x", **row, "present_04": 0, "n_obs": 20}, {**base, "stem": "y", **row, "n_obs": 30}])
    comp = pd.DataFrame([{**base, "n_patches": 2}])
    mm = fillstats.month_missing(pf, comp, ["country", "year"])
    assert np.isclose(mm.loc[mm.month == 4, "share_missing"].item(), 0.5) and mm.loc[mm.month == 5, "share_missing"].item() == 0
    o = fillstats.obs_by_country_year(pf)
    assert o.loc[0, "mean"] == 25


def test_combine_tile_years():
    mm = pd.DataFrame([
        {"country": "France", "year": 2019, "tile": "A", "month": 4, "share_missing": 1.0},
        {"country": "France", "year": 2019, "tile": "B", "month": 4, "share_missing": 0.0},
    ])
    pop = pd.DataFrame([
        {"country": "France", "year": 2019, "tile": "A", "n_patches": 100},
        {"country": "France", "year": 2019, "tile": "B", "n_patches": 300},
    ])
    out = fillstats.combine_tile_years(mm, pop)
    assert np.isclose(out["share_missing"].item(), 0.25)
