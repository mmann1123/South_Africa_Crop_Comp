import numpy as np
import pandas as pd

from experiments.external_audits.denethor import dnsummary


def _summary():
    rows = []
    for tile, year, vals in [
        ("18E-242N", 2018, [50, 54, 58]),
        ("17E-243N", 2018, [53, 55, 57]),
        ("18E-242N", 2019, [44, 48, 52]),
        ("17E-243N", 2019, [46, 48, 50]),
    ]:
        for i, v in enumerate(vals):
            rows.append(
                {
                    "tile_year": f"s2-utm-33N-{tile}-{year}",
                    "tile": tile,
                    "year": year,
                    "field_id": i,
                    "n_clear_literal40": v - 10,
                    "n_clear_s2cp40": v,
                    "n_clear_scl": v + 1,
                    "n_clear_s2cp40_aprOct": v // 2,
                    "longest_gap_days_s2cp40_aprOct": 20,
                    "clear_by_month_s2cp40": ";".join(["4"] * 12),
                }
            )
    return pd.DataFrame(rows)


def test_denethor_2x2_medians_and_margins():
    t = dnsummary.two_by_two(_summary())
    cell = t.set_index(["tile", "year"])["median_n_clear_s2cp40"]
    assert cell[("18E-242N", 2018)] == 54 and cell[("17E-243N", 2019)] == 48
    m = dnsummary.margins(t)
    assert m["year_effect_18E-242N"] == 48 - 54
    assert m["year_effect_17E-243N"] == 48 - 55
    assert m["tile_effect_2018"] == 55 - 54
    assert m["tile_effect_2019"] == 48 - 48
    assert m["benchmark_difference"] == 48 - 54  # test tile-year minus train tile-year


def test_tile_year_summary_columns():
    s = dnsummary.tile_year_summary(_summary())
    assert len(s) == 4
    r = s.set_index("tile_year").loc["s2-utm-33N-18E-242N-2018"]
    assert r["n_fields"] == 3 and r["median_n_clear_s2cp40"] == 54
    assert r["share_fields_under_20_clear_aprOct_s2cp40"] == 0.0
    assert np.isclose(r["iqr_n_clear_s2cp40"], 4.0)
    assert r["mean_clear_by_month"] == ";".join(["4.0"] * 12)
