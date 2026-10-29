import pytest

from experiments.external_audits.sen4agrinet import splits


def test_parse_and_composition():
    df = splits.parse_images(["netcdf/2019_31UCP_patch_05_06.nc", "netcdf/2020_31TBF_patch_17_11.nc", "netcdf/2020_31TBF_patch_17_12.nc"])
    assert list(df["country"]) == ["France", "Catalonia", "Catalonia"]
    assert df.loc[0, "row"] == 5 and df.loc[0, "col"] == 6
    df.insert(0, "split", "test")
    df.insert(0, "scenario", 3)
    c = splits.composition(df)
    assert c.loc[(c["tile"] == "31TBF"), "n_patches"].item() == 2


def test_bad_names():
    with pytest.raises(ValueError):
        splits.parse_images(["netcdf/foo.nc"])
    with pytest.raises(ValueError):
        splits.country_of("30TXT")


def test_hf_path():
    assert splits.hf_path("2019_31TBF_patch_17_11") == "data/2019/31TBF/2019_31TBF_patch_17_11.nc"


def test_sample_per_tile_year_deterministic_and_capped():
    import pandas as pd

    rows = [{"scenario": 2, "split": "test", "country": "France", "year": 2019, "tile": "31TCJ", "row": i, "col": 0, "stem": f"2019_31TCJ_patch_{i}_0"} for i in range(40)]
    rows += [{"scenario": 2, "split": "test", "country": "France", "year": 2019, "tile": "31TCL", "row": i, "col": 0, "stem": f"2019_31TCL_patch_{i}_0"} for i in range(5)]
    df = pd.DataFrame(rows)
    a = splits.sample_per_tile_year(df, 25, 42)
    b = splits.sample_per_tile_year(df, 25, 42)
    assert a.equals(b)
    assert (a["tile"] == "31TCJ").sum() == 25 and (a["tile"] == "31TCL").sum() == 5
    assert a["stem"].is_unique


def test_decode_times():
    t = splits.decode_times("seconds since 2019-01-06 10:54:31", [0, 4319750])
    assert str(t[0]) == "2019-01-06 10:54:31" and str(t[1]) == "2019-02-25 10:50:21"
    with pytest.raises(ValueError):
        splits.decode_times("days since 2019-01-01", [0])


def test_stac_monthly_counts_dedup_and_threshold():
    feats = [
        {"properties": {"datetime": "2020-04-25T10:59:21.933000Z", "eo:cloud_cover": 18.0}},
        {"properties": {"datetime": "2020-04-25T10:59:21.933000Z", "eo:cloud_cover": 18.0}},  # reprocessed duplicate
        {"properties": {"datetime": "2020-05-02T10:59:21Z", "eo:cloud_cover": 9.99}},
        {"properties": {"datetime": "2020-05-12T10:59:21Z", "eo:cloud_cover": 10.0}},
    ]
    out = splits.stac_monthly_counts(feats, 2020)
    assert out[3] == {"month": 4, "n_granules": 1, "n_cloud_lt10": 0}
    assert out[4] == {"month": 5, "n_granules": 2, "n_cloud_lt10": 1}
    assert out[0]["n_granules"] == 0
