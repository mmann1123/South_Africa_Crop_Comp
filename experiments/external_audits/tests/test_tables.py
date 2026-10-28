import json

import pandas as pd

from experiments.external_audits.common import tables


def test_write_csv_writes_sidecar(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    out = tables.write_csv(df, tmp_path / "t.csv", script="s", inputs=["x.csv"])
    assert out.exists()
    meta = json.loads((tmp_path / "t.meta.json").read_text())
    assert meta["script"] == "s" and meta["inputs"] == ["x.csv"]
    assert "git_commit" in meta and "run_date" in meta
    assert pd.read_csv(out)["a"].tolist() == [1, 2]


def test_longest_gap_days_simple():
    dates = pd.Series(pd.to_datetime(["2018-04-10", "2018-05-20", "2018-06-01"]))
    clear = pd.Series([True, True, False])
    assert tables.longest_gap_days(dates, clear, "2018-04-01", "2018-10-31") == 164


def test_longest_gap_days_no_clear():
    dates = pd.Series(pd.to_datetime(["2018-04-10"]))
    clear = pd.Series([False])
    assert tables.longest_gap_days(dates, clear, "2018-04-01", "2018-10-31") == 213
