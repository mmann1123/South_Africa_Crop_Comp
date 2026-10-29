import pandas as pd
import pytest
from shapely.geometry import box

from experiments.external_audits.stoian import l1c


def test_parse_footprint():
    g = l1c.parse_footprint("geography'SRID=4326;POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'")
    assert g.area == pytest.approx(1.0)


def test_footprint_share_full_and_half():
    # a UTM 30N tile square of 100 km at 500000 E, 4600000 N (about 41.5 N, 3 W): footprint covering all of it
    tile = (500000.0, 4600000.0, 600000.0, 4700000.0)
    from pyproj import Transformer

    t = Transformer.from_crs(32630, 4326, always_xy=True).transform
    big = box(*tile)
    lon0, lat0 = t(tile[0] - 5000, tile[1] - 5000)
    lon1, lat1 = t(tile[2] + 5000, tile[3] + 5000)
    assert l1c.footprint_share(box(lon0 - 0.1, lat0 - 0.1, lon1 + 0.1, lat1 + 0.1), tile, 32630) == pytest.approx(1.0, abs=1e-6)
    lonm, _ = t(550000, 4650000)
    half = box(lon0 - 0.1, lat0 - 0.1, lonm, lat1 + 0.1)
    assert 0.45 < l1c.footprint_share(half, tile, 32630) < 0.55


def test_effective_clear_windows():
    df = pd.DataFrame({"date": ["2016-02-01", "2016-04-01", "2016-06-01"], "cloud_cover": [0.0, 50.0, 100.0], "share": [1.0, 1.0, 0.5]})
    assert l1c.effective_clear(df, "2016-01-01", "2016-12-31") == pytest.approx(1.5)
    assert l1c.effective_clear(df, "2016-03-01", "2016-10-31") == pytest.approx(0.5)
