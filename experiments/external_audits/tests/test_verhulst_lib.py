import datetime as dt

import numpy as np
import pytest

from experiments.external_audits.verhulst import verhulstlib as vl


def test_dekad_bounds_cover_the_year_without_gaps():
    for year in (2018, 2020):
        b = [vl.dekad_bounds(year, k) for k in range(36)]
        assert b[0][0] == dt.date(year, 1, 1) and b[-1][1] == dt.date(year + 1, 1, 1)
        assert all(b[i][1] == b[i + 1][0] for i in range(35))
        assert b[1] == (dt.date(year, 1, 11), dt.date(year, 1, 21))
        assert b[2][1] == dt.date(year, 2, 1)


def test_dekad_of_matches_bounds():
    for k in range(36):
        a, b = vl.dekad_bounds(2019, k)
        assert vl.dekad_of(a) == k and vl.dekad_of(b - dt.timedelta(days=1)) == k


def test_gaussian_kernel_is_normalised_and_peaked():
    k = vl.gaussian_kernel()
    assert k.shape == (11, 11) and k.sum() == pytest.approx(1.0) and k[5, 5] == k.max()
    assert k[5, 6] / k[5, 5] == pytest.approx(np.exp(-1 / 4.5))


def test_dilation_none_keeps_mask_and_threshold_grows_reach():
    usable = np.ones((21, 21), bool)
    usable[10, 10] = False
    assert (vl.dilate_unusable(usable, None) == usable).all()
    d1 = vl.dilate_unusable(usable, 0.05)  # adjacent pixels (weight 0.057) go, distance 2 stay
    d3 = vl.dilate_unusable(usable, 0.01)
    assert not d1[10, 11] and d1[10, 12] and not d3[10, 12] and d3[10, 14]
    assert (~d1).sum() < (~d3).sum()


def test_dilation_never_adds_usable_pixels():
    rng = np.random.default_rng(42)
    usable = rng.random((30, 30)) > 0.2
    assert not (vl.dilate_unusable(usable, 0.02) & ~usable).any()


def test_longest_run():
    assert vl.longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert vl.longest_run(np.zeros(5, bool)) == 0


def test_gap_metrics_partitions_dekads():
    v = np.zeros((3, 36), bool)
    v[0, 4:30] = True  # edges: 4 + 6 empty, no interior gaps
    v[1, [3, 6, 20]] = True  # interior gaps of 2 and 13, edges 3 and 15
    # row 2: no valid dekad
    m = vl.gap_metrics(v)
    assert list(m["n_valid"]) == [26, 3, 0]
    assert list(m["n_edge"]) == [10, 18, 36]
    assert list(m["n_interp"]) == [0, 15, 0]
    assert (m["n_valid"] + m["n_edge"] + m["n_interp"] == 36).all()
    assert list(m["longest_gap"]) == [6, 15, 36]
    assert list(m["longest_interp"]) == [0, 13, 0]
    assert list(m["n_no_data"]) == [0, 0, 1]
    assert list(m["longest_gap_jj"]) == [4, 11, 18]


def test_year_summary_and_monthly_share():
    v = np.zeros((2, 36), bool)
    v[:, ::3] = True  # first dekad of every month
    s = vl.year_summary(v)
    assert s["valid_dekads_mean"] == 12 and s["valid_share"] == pytest.approx(1 / 3)
    assert s["interp_share"] + s["edge_share"] == pytest.approx(s["filled_share"])
    assert np.allclose(vl.monthly_valid_share(v), 1 / 3)


def test_scenarios_loss():
    oa = {(a, b): 70.0 + (a == b) * 5 for a in (1, 2, 3) for b in (1, 2, 3)}
    sc = vl.scenarios(oa, [1, 2, 3])
    assert len(sc) == 6 and all(s["oa_loss"] == pytest.approx(5.0) for s in sc)


def test_spearman_monotone():
    assert vl.spearman([1, 2, 3, 4], [1, 4, 9, 16])["spearman_rho"] == pytest.approx(1.0)


def test_box_from_mask_finds_median_line():
    m = np.zeros((40, 12), bool)
    m[10:30, 2:10] = True
    m[19, 2:10] = False  # darker median line replaces the fill
    top, med, bot = vl.box_from_mask(m)
    assert (top, med, bot) == (10.0, 19.0, 29.0)
    assert vl.box_from_mask(np.zeros((5, 5), bool)) is None


def test_blob_centroid_and_calibration():
    m = np.zeros((20, 20), bool)
    m[4:7, 8:11] = True
    m[15, 1] = True
    assert vl.blob_centroid(m) == (5.0, 9.0)
    slope, icpt = vl.calibrate_linear([0, 100], [90, 50])
    assert slope * 50 + icpt == pytest.approx(70)
