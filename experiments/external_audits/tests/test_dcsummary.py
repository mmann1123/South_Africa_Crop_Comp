import numpy as np
import pandas as pd

from experiments.external_audits.daccn import dcsummary


def _counters():
    rows = []
    for region, year, variant, nz in [
        ("Minnesota", 2018, "A", [0, 0, 1, 2]),
        ("Wisconsin", 2018, "A", [0, 1, 1, 3]),
        ("Minnesota", 2019, "A", [1, 1, 2, 2]),
        ("Wisconsin", 2019, "A", [0, 0, 0, 0]),
    ]:
        for i, z in enumerate(nz):
            r = {"region": region, "country": "United States of America", "year": year, "variant": variant, "frame": "cdl_conf90", "class": "corn", "n_valid": 22 - z, "n_zero": z}
            for k in range(22):
                r[f"zero_{k:02d}"] = int(k < z)  # zeros at the first z steps
                r[f"valid_{k:02d}"] = int(k >= z)
            rows.append(r)
    return pd.DataFrame(rows)


def test_region_year_summary():
    s = dcsummary.region_year_summary(_counters()).set_index(["region", "year", "variant"])
    r = s.loc[("Minnesota", 2018, "A")]
    assert r["n_points"] == 4
    assert r["share_any_zero"] == 0.5
    assert np.isclose(r["mean_n_zero"], 0.75)
    assert r["share_zero_step01"] == 0.5 and r["share_zero_step22"] == 0.0
    assert np.isclose(r["share_zero_apr"], (0.5 + 0.25 + 0.0) / 3)  # steps 1..3 mean


def test_pairs_table():
    p = dcsummary.pairs_table(_counters(), spatial_pairs=[("Minnesota", "Wisconsin")]).set_index(["source", "target", "year", "variant"])
    r = p.loc[("Minnesota", "Wisconsin", 2018, "A")]
    assert r["source_share_any_zero"] == 0.5 and r["target_share_any_zero"] == 0.75
    assert np.isclose(r["difference"], 0.25)
    t = p.loc[("Minnesota 2018", "Minnesota 2019", 2019, "A")]
    assert t["kind"] == "temporal" and t["source_share_any_zero"] == 0.5 and t["target_share_any_zero"] == 1.0


def test_region_year_summary_cell_spread():
    df = _counters()
    df["cell"] = [0, 0, 1, 1] * 4
    s = dcsummary.region_year_summary(df).set_index(["region", "year", "variant"])
    r = s.loc[("Minnesota", 2018, "A")]
    # cells: cell 0 shares any-zero [0,0] -> 0.0 ; cell 1 [1,1] -> 1.0 ; sd (ddof=1) of [0,1] = 0.7071
    assert r["n_cells"] == 2
    assert np.isclose(r["cell_sd_share_any_zero"], 0.70710678)
