import numpy as np
import pandas as pd

from experiments.external_audits.sen4agrinet import fillrule as fr


def ts(*s):
    return pd.DatetimeIndex(pd.to_datetime(list(s), format="mixed"))


def test_month_bins_right_closed():
    t = ts("2019-01-01 00:00:00", "2019-01-01 00:00:01", "2019-02-01 00:00:00", "2020-01-01 00:00:00", "2019-03-15 10:40")
    assert list(fr.month_bins(t, 2019)) == [-1, 1, 1, 12, 3]


def test_present_months_and_masks_interior_and_edges():
    t = ts("2019-02-10", "2019-03-05", "2019-06-20", "2019-09-01 10:50", "2019-10-12")
    p = fr.present_months(t, 2019)
    assert list(np.flatnonzero(p) + 1) == [2, 3, 6, 9, 10]
    interp, extrap = fr.fill_masks(p)
    assert list(np.flatnonzero(interp) + 1) == [4, 5, 7, 8]
    assert list(np.flatnonzero(extrap) + 1) == [1, 11, 12]


def test_window_stats_pad_and_oad():
    p = np.zeros(12, bool)
    p[[1, 2, 5, 6, 8, 9]] = True  # Feb, Mar, Jun, Jul, Sep, Oct
    pad = fr.window_stats(p, fr.PAD_MONTHS)  # Apr..Sep
    assert pad == {"months": 6, "interpolated": 3, "extrapolated": 0, "filled": 3}  # Apr, May, Aug
    oad = fr.window_stats(p, fr.OAD_MONTHS)  # May..Nov
    assert oad == {"months": 7, "interpolated": 2, "extrapolated": 1, "filled": 3}  # May, Aug, Nov(extrap)


def test_no_data_and_full_data():
    i, e = fr.fill_masks(np.zeros(12, bool))
    assert not i.any() and not e.any()
    i, e = fr.fill_masks(np.ones(12, bool))
    assert not i.any() and not e.any()


def test_single_acquisition_extrapolates_everything_else():
    p = np.zeros(12, bool)
    p[3] = True
    i, e = fr.fill_masks(p)
    assert not i.any() and e.sum() == 11


def test_linear_fill_matches_xarray_style_rule():
    x = fr.month_starts(2019)
    p = np.ones(12, bool)
    p[[0, 3, 4, 11]] = False
    v = np.where(p, 10 + 2.0 * np.arange(12) ** 1.5, np.nan)
    f = fr.linear_fill(v, p, x)
    # interior months lie on the chord between the neighbouring valid months, weighted by days
    k = 3
    exp = v[2] + (v[5] - v[2]) * (x[3] - x[2]) / (x[5] - x[2])
    assert np.isclose(f[k], exp)
    # leading month extrapolated from months 2 and 3 (Feb, Mar are valid at idx 1,2)
    assert np.isclose(f[0], v[1] + (v[2] - v[1]) * (x[0] - x[1]) / (x[2] - x[1]))
    assert np.isclose(f[11], v[10] + (v[10] - v[9]) * (x[11] - x[10]) / (x[10] - x[9]))


def _simulate(p, rng, n=1, bands=13):
    x = fr.month_starts(2019)
    W = fr.fill_weights(p, x)
    base = rng.uniform(100, 5000, (n, 12, bands))
    return np.einsum("kj,njb->nkb", W, base)


def test_fill_weights_match_linear_fill():
    rng = np.random.default_rng(42)
    x = fr.month_starts(2019)
    p = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], bool)
    v = rng.uniform(0, 1, 12)
    assert np.allclose(fr.fill_weights(p, x) @ np.where(p, v, 0), fr.linear_fill(np.where(p, v, np.nan), p, x))


def test_infer_filled_random_patterns_count_and_interior_exact():
    rng = np.random.default_rng(42)
    x = fr.month_starts(2019)
    interior_exact = interior_total = 0
    for _ in range(300):
        p = rng.random(12) > 0.35
        if p.sum() < 3:
            continue
        data = _simulate(p, rng)
        filled, _ = fr.infer_filled(data, 2019)
        # the fill count is identifiable: never fewer than generated, and the fill reproduces the data
        assert filled[0].sum() >= (~p).sum()
        assert np.allclose(fr.fill_weights(~filled[0], x) @ data[0], data[0], rtol=1e-5)
        if p[[0, 1, 10, 11]].all():  # no collinear triple at either edge, so the set is identifiable
            interior_total += 1
            interior_exact += bool((filled[0] == ~p).all())
    assert interior_total > 20 and interior_exact == interior_total


def test_infer_filled_unambiguous_edge_cases():
    rng = np.random.default_rng(1)
    cases = {
        "interior_single": [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "interior_runs": [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        "all_valid": [1] * 12,
    }
    for name, p in cases.items():
        p = np.array(p, bool)
        filled, extrap = fr.infer_filled(_simulate(p, rng), 2019)
        assert (filled[0] == ~p).all() and not extrap.any(), name


def test_infer_filled_trailing_and_leading_counts():
    rng = np.random.default_rng(1)
    for p in ([1] * 8 + [0, 0, 0, 0], [0, 0, 0] + [1] * 9, [1] * 10 + [0, 1]):
        p = np.array(p, bool)
        filled, extrap = fr.infer_filled(_simulate(p, rng), 2019)
        assert filled[0].sum() == (~p).sum()  # count identifiable; which edge month is "valid" is not
        assert extrap[0].sum() >= 1 or p[[0, 11]].all()


def test_infer_ignores_zero_rows():
    filled, _ = fr.infer_filled(np.zeros((1, 12, 13)), 2019)
    assert not filled.any()
