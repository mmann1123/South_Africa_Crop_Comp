import pandas as pd

from experiments.external_audits.stoian import table1

TILES = ["T1", "T2"]
TEXT = "\n".join(["Table 1.", "Tile Name", "T1", "T2", "Total", "ASC", "10", "2,000", "2,010 (50.00%)", "AWC", "5", "1,995", "2,000 (50.00%)", "Total", "15", "3,995", "4,010"])


def test_parse_and_totals():
    df = table1.parse_table1(TEXT, TILES)
    assert df["class"].tolist() == ["ASC", "AWC"]
    assert df.loc[0, "T2"] == 2000 and df.loc[1, "total_ha"] == 2000 and df.loc[0, "total_pct"] == 50.0
    rows, cols = table1.check_totals(df, TILES)
    assert rows == [] and cols == {"T1": 0, "T2": 0, "total_ha": 0}


def test_row_mismatch_is_reported():
    df = table1.parse_table1(TEXT, TILES)
    df.loc[0, "T1"] = 11
    rows, _ = table1.check_totals(df, TILES)
    assert len(rows) == 1 and "ASC" in rows[0]
