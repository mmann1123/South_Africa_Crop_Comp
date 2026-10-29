import numpy as np
import pandas as pd

from experiments.external_audits.common.tables import longest_gap_days
from experiments.external_audits.stoian import sclstats


def test_clear_mask_keeps_snow_and_water():
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert sclstats.clear_mask(scl).tolist() == [False, False, True, False, True, True, True, True, False, False, False, True]


def test_merge_same_date_prefers_clear_then_nonzero():
    a = np.array([[0, 9, 4, 8]], np.uint8)
    b = np.array([[5, 0, 9, 8]], np.uint8)
    out = sclstats.merge_same_date([a, b])
    assert out.tolist() == [[5, 9, 4, 8]]


def test_land_mask_drops_sea_and_never_clear():
    T = 6
    stack = np.full((T, 1, 3), 4, np.uint8)
    stack[:, 0, 1] = 6  # sea: water on every clear date
    stack[:, 0, 2] = 9  # always cloud: no clear observation
    assert sclstats.land_mask(stack).tolist() == [[True, False, False]]


def test_per_pixel_gap_matches_scalar_definition():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-03", "2016-12-28", freq="5D")
    clear = rng.random((len(dates), 4, 5)) < 0.3
    gap = sclstats.per_pixel_longest_gap(clear, dates.values, "2016-03-01", "2016-10-31")
    for i in range(4):
        for j in range(5):
            assert gap[i, j] == longest_gap_days(pd.Series(dates), pd.Series(clear[:, i, j]), "2016-03-01", "2016-10-31")


def test_no_clear_date_gives_whole_window():
    dates = pd.date_range("2016-03-05", "2016-10-01", freq="10D").values
    gap = sclstats.per_pixel_longest_gap(np.zeros((len(dates), 1, 1), bool), dates, "2016-03-01", "2016-10-31")
    assert gap[0, 0] == (pd.Timestamp("2016-10-31") - pd.Timestamp("2016-03-01")).days


def test_tile_stats_counts():
    dates = np.array(["2016-02-01", "2016-04-01", "2016-06-01", "2016-12-01"], dtype="datetime64[D]")
    stack = np.full((4, 2, 2), 4, np.uint8)
    stack[1, 0, :] = 9
    st = sclstats.tile_stats(stack, dates, ("2016-03-01", "2016-10-31"))
    assert st["n_land_pixels"] == 4 and st["median_clear_year"] == 3.5 and st["median_clear_marOct"] == 1.5
