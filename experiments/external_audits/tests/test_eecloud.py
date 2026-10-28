import math

from experiments.external_audits.timematch.eecloud import row_from_info


def test_rows_for_dates_without_images():
    r = row_from_info("denmark", "32VNH", "2017-06-01", 41.0, None, "worldcover40")
    assert r["n_images"] == 0
    assert r["image_ids"] == ""
    assert math.isnan(r["frac_cloudy_csplus"])
    assert math.isnan(r["frac_cloudy_s2cp40"])
    assert math.isnan(r["frac_nodata"])
    assert r["tile"] == "32VNH" and r["date"] == "2017-06-01" and r["tile_cloudy_pct"] == 41.0


def test_row_from_info_maps_keys():
    info = {
        "n_images": 2,
        "n_pixels": 1000,
        "csplus_cloudy": 0.42,
        "s2cp40": 0.4,
        "s2cp16": 0.6,
        "nodata": 0.01,
        "image_ids": ["a", "b"],
    }
    r = row_from_info("denmark", "32VNH", "2017-06-01", 41.0, info, "worldcover40")
    assert r["n_images"] == 2 and r["n_pixels"] == 1000
    assert r["frac_cloudy_csplus"] == 0.42
    assert r["frac_cloudy_s2cp40"] == 0.4 and r["frac_cloudy_s2cp16"] == 0.6
    assert r["frac_nodata"] == 0.01
    assert r["image_ids"] == "a;b"
    assert r["support_kind"] == "worldcover40"
