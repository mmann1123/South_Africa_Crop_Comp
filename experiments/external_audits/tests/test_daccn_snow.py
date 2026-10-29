import numpy as np
import pandas as pd

from experiments.external_audits.daccn import snowrule as sr


def test_snow_flag_rule():
    # bright, high NDSI, bright NIR -> snow
    assert bool(sr.snow_flag(0.6, 0.1, 0.5))
    # NDSI just below 0.42: (0.5-0.2)/(0.7)=0.4286 above; (0.5-0.22)/(0.72)=0.389 below
    assert bool(sr.snow_flag(0.5, 0.2, 0.3))
    assert not bool(sr.snow_flag(0.5, 0.22, 0.3))
    # low NIR fails even with high NDSI (water/shadow)
    assert not bool(sr.snow_flag(0.5, 0.1, 0.10))
    # dark green fails
    assert not bool(sr.snow_flag(0.09, 0.01, 0.2))
    # cloud: NDSI ~ 0
    assert not bool(sr.snow_flag(0.5, 0.5, 0.5))
    # vectorised and zero-denominator safe
    out = sr.snow_flag(np.array([0.6, 0.0]), np.array([0.1, 0.0]), np.array([0.5, 0.5]))
    assert out.tolist() == [True, False]


def _df():
    rows = []
    spec = [  # region, year, variant, cell, snow steps, invalid steps
        ("Minnesota", 2018, "B", 0, [0, 1], []),
        ("Minnesota", 2018, "B", 0, [], []),
        ("Minnesota", 2018, "B", 1, [20], []),
        ("Minnesota", 2018, "B", 1, [0], [0]),  # snow flag on an invalid step must be ignored
        ("Wisconsin", 2018, "B", 0, [], []),
    ]
    for region, year, variant, cell, snow, invalid in spec:
        r = {"region": region, "year": year, "variant": variant, "cell": cell, "class": "corn"}
        for k in range(22):
            r[f"valid_{k:02d}"] = 0 if k in invalid else 1
            r[f"snow_{k:02d}"] = int(k in snow)
            r[f"scl_{k:02d}"] = 1 if (k in snow and k == 0) else (-1 if k == 21 else 0)
            r[f"modis_{k:02d}"] = 80 if k in snow else (-1 if k == 21 else 0)
        rows.append(r)
    return pd.DataFrame(rows)


def test_region_year_snow():
    s = sr.region_year_snow(_df()).set_index("region")
    m = s.loc["Minnesota"]
    assert m["n_points"] == 4
    assert m["share_any_snow"] == 0.5  # rows 1,3 have valid snow; row 4 snow only on invalid step
    assert np.isclose(m["mean_n_snow"], 0.75)  # (2+0+1+0)/4
    assert m["share_any_snow_apr"] == 0.25
    assert m["share_any_snow_oct"] == 0.25
    assert s.loc["Wisconsin", "share_any_snow"] == 0.0
    by = [float(x) for x in m["share_snow_by_step"].split(";")]
    assert len(by) == 22 and np.isclose(by[0], 1 / 3, atol=1e-4)  # step 1: 3 valid rows, 1 with snow


def test_crosscheck():
    c = sr.crosscheck(_df()).set_index(["region", "year"])
    m = c.loc[("Minnesota", 2018)]
    assert m["n_flagged"] == 3
    assert m["scl_snow_given_flag"] == 1 / 3  # only step 0 of row 0 has SCL snow among 3 flagged
    assert m["modis_snow_given_flag"] == 1.0
    assert m["modis_snow_given_noflag"] == 0.0
    assert 0 < m["scl_coverage"] < 1  # step 22 has no L2A


def test_pairs_snow():
    s = sr.region_year_snow(_df())
    s2 = s.copy()
    s2["year"] = 2019
    s2["share_any_snow"] = 0.1
    both = pd.concat([s, s2], ignore_index=True)
    p = sr.pairs_snow(both, [("Minnesota", "Wisconsin")])
    sp = p[p["kind"] == "spatial"]
    assert len(sp) == 2
    r = sp[sp["year"] == 2018].iloc[0]
    assert np.isclose(r["difference"], 0.0 - 0.5)
    t = p[(p["kind"] == "temporal") & (p["source"] == "Minnesota 2018")].iloc[0]
    assert np.isclose(t["difference"], 0.1 - 0.5)
