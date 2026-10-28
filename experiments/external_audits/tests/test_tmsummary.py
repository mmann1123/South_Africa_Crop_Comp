import numpy as np
import pandas as pd

from experiments.external_audits.timematch import tmsummary


def _rows():
    return pd.DataFrame(
        {
            "country": ["denmark"] * 3 + ["france"] * 2,
            "tile": ["32VNH"] * 3 + ["30TXT"] * 2,
            "date": ["2017-03-01", "2017-03-11", "2017-06-01", "2017-05-01", "2017-05-11"],
            "tile_cloudy_pct": [10.0, 60.0, 80.0, 5.0, 55.0],
            "n_images": [1, 1, 1, 1, 0],
            "n_pixels": [100] * 5,
            "frac_cloudy_csplus": [0.2, 0.5, 1.0, 0.1, np.nan],
            "frac_cloudy_s2cp40": [0.1, 0.6, 0.9, 0.2, np.nan],
            "frac_cloudy_s2cp16": [0.3, 0.7, 1.0, 0.3, np.nan],
            "frac_nodata": [0.0, 0.0, 0.1, 0.0, np.nan],
        }
    )


def test_tile_summary_expected_cloudy_obs():
    s = tmsummary.tile_summary(_rows()).set_index("tile")
    dk = s.loc["32VNH"]
    assert dk["n_dates"] == 3
    assert dk["n_dates_with_images"] == 3
    assert np.isclose(dk["expected_cloudy_obs_csplus"], 1.7)
    assert dk["dates_field_over50_csplus"] == 1  # strictly greater than 0.5
    assert dk["dates_field_over80_csplus"] == 1
    assert dk["dates_tile_over50"] == 2
    assert np.isclose(dk["mean_field_cloudy_csplus"], 1.7 / 3)
    fr = s.loc["30TXT"]
    assert fr["n_dates"] == 2 and fr["n_dates_with_images"] == 1
    assert np.isclose(fr["expected_cloudy_obs_s2cp40"], 0.2)  # NaN rows do not count


def test_monthly_summary_expected_clear():
    m = tmsummary.monthly_summary(_rows()).set_index(["tile", "month"])
    assert m.loc[("32VNH", 3), "n_dates"] == 2
    assert np.isclose(m.loc[("32VNH", 3), "expected_cloudy_obs_csplus"], 0.7)
    assert np.isclose(m.loc[("32VNH", 3), "expected_clear_obs_csplus"], 1.3)
