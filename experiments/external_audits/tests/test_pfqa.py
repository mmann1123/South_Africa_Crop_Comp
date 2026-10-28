import numpy as np
import pandas as pd

from experiments.external_audits.denethor import pfqa


def test_missing_day_excluded():
    daily = pd.DataFrame(
        {
            "field_pos": [0, 0, 0],
            "date": pd.to_datetime(["2018-04-01", "2018-04-02", "2018-04-03"]),
            "frac_synthetic": [0.9, np.nan, 0.2],
            "mean_percent_synthetic": [80.0, np.nan, 10.0],
            "mean_days_to_real": [3.0, np.nan, 0.0],
        }
    )
    out = pfqa.summarise(daily)
    row = out.iloc[0]
    assert row["month"] == 4
    assert row["n_days_with_qa"] == 2
    assert row["n_days_missing"] == 1
    assert row["share_days_mostly_synthetic"] == 0.5  # one of two QA days has frac_synthetic > 0.5
    assert row["mean_percent_synthetic"] == 45.0
    assert row["mean_days_to_real"] == 1.5


def test_field_day_stats_threshold():
    pct = np.array([[1, 60], [100, 1]], dtype=np.int16)
    days = np.array([[0, 2], [5, 0]], dtype=np.int16)
    idx = np.zeros((2, 2), dtype=np.int32)  # one field covering all four pixels
    out = pfqa.field_day_stats(pct, days, idx, 1)
    assert out.loc[0, "frac_synthetic"] == 0.5
    assert out.loc[0, "mean_percent_synthetic"] == 40.5
    assert out.loc[0, "mean_days_to_real"] == 1.75


def test_read_day_returns_none_on_missing(tmp_path):
    from rasterio.windows import Window

    assert pfqa.read_day(str(tmp_path / "nope.tif"), Window(0, 0, 10, 10)) is None
