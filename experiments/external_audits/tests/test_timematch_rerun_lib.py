import numpy as np

from experiments.external_audits.timematch_rerun import rerunlib as r


def test_folds_are_disjoint_and_sized():
    f = r.make_folds(100, 50, "src", "tgt")
    assert len(f) == 3
    for s in f:
        for ds, n in (("src", 100), ("tgt", 50)):
            a = s[ds]
            assert not (a["train"] & a["val"]) and not (a["train"] & a["test"]) and not (a["val"] & a["test"])
            assert len(a["train"]) + len(a["val"]) + len(a["test"]) == n
        assert len(s["tgt"]["test"]) == 10


def test_folds_deterministic_and_differ_by_fold():
    a, b = r.make_folds(100, 50, "s", "t"), r.make_folds(100, 50, "s", "t")
    assert a[0]["t"]["test"] == b[0]["t"]["test"]
    assert a[0]["t"]["test"] != a[1]["t"]["test"]


def test_cloud_flags_nan_is_clear():
    f = r.cloud_flags(np.array([[0.2, 0.5, np.nan, 0.9]]))
    assert f.tolist() == [[False, True, False, True]]


def test_keep_mask_cloud_basic_and_floor():
    flags = np.array([[1, 0, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
    score = np.array([[.9, .1, .8, .2, .0], [.5, .9, .1, .7, .3]])
    keep = r.keep_mask_cloud(flags, score, min_keep=3)
    assert keep[0].tolist() == [False, True, False, True, True]
    assert keep[1].sum() == 3
    assert keep[1].tolist() == [True, False, True, False, True] or keep[1].tolist() == [True, False, True, False, True]
    # the two highest-score dates (1 and 3) are the ones removed
    assert (~keep[1]).nonzero()[0].tolist() == [1, 3]


def test_n_to_remove_matches_cloud_removal():
    rng = np.random.default_rng(0)
    flags = rng.random((50, 12)) < 0.6
    score = rng.random((50, 12))
    keep = r.keep_mask_cloud(flags, score)
    assert (12 - keep.sum(axis=1) == r.n_to_remove(flags)).all()
    assert (keep.sum(axis=1) >= r.MIN_KEEP).all()


def test_random_removal_same_count_and_seeded():
    nrem = np.array([0, 2, 5, 1])
    k1 = r.keep_mask_random(nrem, 8, np.random.default_rng(1))
    k2 = r.keep_mask_random(nrem, 8, np.random.default_rng(1))
    assert (k1 == k2).all()
    assert ((8 - k1.sum(axis=1)) == nrem).all()
    k3 = r.keep_mask_random(nrem, 8, np.random.default_rng(2))
    assert not (k1 == k3).all()


def test_group_by_shape_batches():
    b = r.group_by_shape([5, 5, 5, 6, 5], [10, 10, 10, 10, 20], 2)
    flat = sorted(i for x in b for i in x)
    assert flat == [0, 1, 2, 3, 4]
    assert all(len(x) <= 2 for x in b)
    assert [0, 1] in b and [2] in b


def test_metrics_perfect_and_wrong():
    m = r.metrics([0, 1, 2], [0, 1, 2], 3)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1
    m = r.metrics([0, 0, 1], [1, 1, 0], 2)
    assert m["accuracy"] == 0


def test_stratified_sample_seeded_and_capped():
    y = np.array([0] * 100 + [1] * 5)
    a, b = r.stratified_sample(y, 10), r.stratified_sample(y, 10)
    assert (a == b).all() and (y[a] == 0).sum() == 10 and (y[a] == 1).sum() == 5


def test_effect_and_share():
    assert r.cloud_effect(0.5, 0.4) == np.float64(0.5) - 0.4
    assert np.isclose(r.residual_share(0.05, 0.5, 0.9), 0.125)
    assert np.isnan(r.residual_share(0.05, 0.95, 0.9))


def test_block_flags_to_parcels():
    bf = np.array([[0.9, 0.1], [0.2, 0.7]])
    out = r.block_flags_to_parcels([1, 0, 1], bf)
    assert out.tolist() == [[False, True], [True, False], [False, True]]


def _toy():
    import pandas as pd

    rows = []
    for fold in (0, 1):
        base = {"direction": "X", "fold": fold, "mean_dates_kept": 40.0, "mean_removed": np.nan}
        rows.append({**base, "cloud_source": "none", "threshold": np.nan, "condition": "a", "draw": -1, "macro_f1": 0.30, "accuracy": 0.5})
        rows.append({**base, "cloud_source": "none", "threshold": np.nan, "condition": "c", "draw": -1, "macro_f1": 0.50, "accuracy": 0.7})
        rows.append({**base, "cloud_source": "s", "threshold": 0.6, "condition": "b", "draw": -1, "macro_f1": 0.33, "accuracy": 0.52})
        rows.append({**base, "cloud_source": "s", "threshold": 0.6, "condition": "d", "draw": -1, "macro_f1": 0.58, "accuracy": 0.72})
        for k, (e, es) in enumerate([(0.30, 0.50), (0.32, 0.54)]):
            rows.append({**base, "cloud_source": "s", "threshold": 0.6, "condition": "e", "draw": k, "macro_f1": e, "accuracy": 0.5})
            rows.append({**base, "cloud_source": "s", "threshold": 0.6, "condition": "e_shift", "draw": k, "macro_f1": es, "accuracy": 0.7})
    return pd.DataFrame(rows)


def test_fold_table_and_effects():
    from experiments.external_audits.timematch_rerun import rsummary as s

    ft = s.fold_table(_toy())
    e = ft[(ft.condition == "e") & (ft.fold == 0)].iloc[0]
    assert np.isclose(e.macro_f1, 0.31) and np.isclose(e.macro_f1_draw_sd, 0.01)
    ef = s.effects(ft, {("X", 0): 0.9, ("X", 1): 0.9}, {("X", 0): 0.7, ("X", 1): 0.7})
    r = ef.iloc[0]
    assert np.isclose(r.effect_b, 0.02) and np.isclose(r.effect_d, 0.06)
    assert np.isclose(r.share_of_shifted_gap, 0.06 / 0.4)
    assert np.isclose(r.share_of_unshifted_gap, 0.02 / 0.6)
    assert np.isclose(r.share_of_timematch_gap, 0.06 / 0.2)
    sm = s.effects_summary(ef)
    assert len(sm) == 1 and np.isclose(sm.effect_d_mean.iloc[0], 0.06)
    cs = s.condition_summary(ft)
    assert set(cs.condition) == {"a", "b", "c", "d", "e", "e_shift"}


def test_add_nodata_and_nodata_dates():
    a0 = np.zeros((2, 3, 4), dtype=np.uint16)
    a1 = np.ones((2, 3, 4), dtype=np.uint16)
    a1[1] = 0
    nd = r.nodata_dates([a0, a1])
    assert nd.tolist() == [[True, True], [False, True]]
    flags = np.array([[False, False], [True, False]])
    f, s = r.add_nodata(flags, np.array([[np.nan, 0.2], [0.9, 0.1]]), nd)
    assert f.tolist() == [[True, True], [True, True]]
    assert s[0, 0] == 2.0 and s[1, 0] == 0.9
