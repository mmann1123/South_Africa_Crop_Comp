import numpy as np

from experiments.external_audits.gallo import gallolib


def test_candidate_days_leap_and_plain():
    assert len(gallolib.candidate_days(2016)) == 366
    assert len(gallolib.candidate_days(2017)) == 365
    assert gallolib.candidate_days(2016)[0] == "20160101"
    assert gallolib.candidate_days(2016)[-1] == "20161231"


def test_file_url():
    u = gallolib.file_url("lombardia2", 2018, 7, "20180105")
    assert u.endswith("fileName=lombardia2/data2018/7/20180105.tif")


def test_harmonise_pads_short_year():
    r = gallolib.harmonise_counts(35)
    assert r["pad_images"] == 36
    assert abs(r["pad_share"] - 36 / 71) < 1e-12
    assert r["subsample_keep"] == 1.0


def test_harmonise_subsamples_long_year():
    r = gallolib.harmonise_counts(142)
    assert r["pad_images"] == 0 and r["pad_share"] == 0
    assert r["subsample_keep"] == 0.5


def test_image_state_fractions():
    a = np.full((9, 2, 2), 500)
    a[:, 0, 0] = 0  # nodata pixel
    a[0, 0, 1] = 5000  # cloud pixel
    s = gallolib.image_state(a)
    assert s == {"nodata": 0.25, "cloud": 0.25, "clear": 0.5}


def test_effective_clear():
    assert gallolib.effective_clear(np.array([0.9, 0.5, 0.49, 0.0])) == 2


def test_zero_image_share():
    # 35 dates, 2 all-zero images: 36 pad + 2 zero slots of 71
    assert abs(gallolib.zero_image_share(35, 2) - 38 / 71) < 1e-12
    # 142 dates subsampled to 71: expected zero images halve
    assert abs(gallolib.zero_image_share(142, 4) - 2 / 71) < 1e-12


def test_parse_name():
    assert gallolib.parse_name("lombardia2/data2018/17/20180105.tif") == {"area": "L2", "year": 2018, "patch": 17, "kind": "image", "date": "20180105"}
    assert gallolib.parse_name("lombardia/data2016/3/y.tif")["kind"] == "label"
    assert gallolib.parse_name("lombardia/data2016/3/20160105_MSAVI.tif") is None
    assert gallolib.parse_name("lombardia-classes/classes.csv") is None


def test_date_stats():
    s = gallolib.date_stats({1: ["a"] * 35, 2: ["a"] * 35, 3: ["a"] * 30})
    assert s["n_patches"] == 3 and s["dates_median"] == 35 and s["dates_min"] == 30
    assert s["pad_images"] == 36


def test_slot_shift_identical_and_shifted():
    a = ["20190110", "20190120", "20190130"]
    assert gallolib.slot_shift(a, a)["median_abs_doy_shift"] == 0
    b = ["20190210", "20190220", "20190302", "20190312"]
    s = gallolib.slot_shift(a, b)
    assert s["slots_compared"] == 3 and s["median_abs_doy_shift"] == 31 and s["slots_beyond_other"] == 0


def test_classify_dates():
    lab = gallolib.classify_dates([0.0, 0.5, 0.49, float("nan"), 0.9], [3, 3, 3, 0, 0])
    assert lab == ["clear", "cloudy", "clear", "nodata", "nodata"]


def test_series_summary_pad_and_counts():
    s = gallolib.series_summary(["clear"] * 10 + ["cloudy"] * 20 + ["nodata"] * 4)
    assert s["pad_slots"] == 37 and s["n_dates"] == 34
    assert s["zero_slots"] == 41 and abs(s["zero_share"] - 41 / 71) < 1e-12
    assert abs(s["clear_share"] - 10 / 71) < 1e-12


def test_series_summary_subsample_scales_counts():
    s = gallolib.series_summary(["clear"] * 72)
    assert s["pad_slots"] == 0 and abs(s["clear_slots"] - 71) < 1e-9


def test_pool_mean_weighted():
    import pandas as pd

    df = pd.DataFrame({"x": [0.0, 1.0]})
    assert gallolib.pool_mean(df, ["x"])["x"] == 0.5
    assert gallolib.pool_mean(df, ["x"], weights=[1, 3])["x"] == 0.75


def test_classify_dates_raster_nodata_overrides():
    lab = gallolib.classify_dates([0.0, 0.0], [6, 6], raster_nodata=[1.0, 0.0])
    assert lab == ["nodata", "clear"]
