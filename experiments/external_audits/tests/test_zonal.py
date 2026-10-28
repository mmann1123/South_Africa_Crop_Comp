import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from experiments.external_audits.common import zonal

CRS = "EPSG:32633"


def _grid():
    # 10 x 10 pixels at 10 m, origin (0, 100): x from 0..100, y from 100 down to 0
    return Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0), (10, 10)


def test_field_index_map_two_squares():
    transform, shape = _grid()
    gdf = gpd.GeoDataFrame(
        {"crop": ["a", "b"]},
        geometry=[box(0, 80, 20, 100), box(50, 50, 70, 70)],
        crs=CRS,
        index=[101, 202],
    )
    idx, fields = zonal.field_index_map(gdf, transform, shape, CRS)
    assert idx.shape == shape and idx.dtype == np.int32
    assert (idx == 0).sum() == 4 and (idx == 1).sum() == 4 and (idx == -1).sum() == 92
    assert list(fields["field_id"]) == [101, 202]
    assert list(fields["n_pixels"]) == [4, 4]
    assert list(fields["field_pos"]) == [0, 1]


def test_field_index_map_drops_empty_fields():
    transform, shape = _grid()
    gdf = gpd.GeoDataFrame(
        geometry=[box(0, 80, 20, 100), box(500, 500, 520, 520)], crs=CRS, index=[1, 2]
    )
    idx, fields = zonal.field_index_map(gdf, transform, shape, CRS)
    assert list(fields["field_id"]) == [1]
    assert fields.attrs["n_dropped"] == 1
    assert idx.max() == 0


def test_field_means_matches_numpy():
    rng = np.random.default_rng(0)
    cube = rng.random((3, 10, 10))
    idx = -np.ones((10, 10), dtype=np.int32)
    idx[0:2, 0:2] = 0
    idx[5:7, 5:7] = 1
    out = zonal.field_means(cube, idx, 2)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], cube[:, idx == 0].mean(axis=1))
    np.testing.assert_allclose(out[:, 1], cube[:, idx == 1].mean(axis=1))


def test_field_fraction_bool():
    cube = np.zeros((1, 10, 10), dtype=bool)
    cube[0, 0, 0] = True  # one of four pixels in field 0
    cube[0, 5:7, 5:7] = True  # all four in field 1
    idx = -np.ones((10, 10), dtype=np.int32)
    idx[0:2, 0:2] = 0
    idx[5:7, 5:7] = 1
    out = zonal.field_fraction(cube, idx, 2)
    np.testing.assert_allclose(out[0], [0.25, 1.0])


def test_grid_from_bbox_north_up_10m():
    t = zonal.grid_from_bbox(432000, 5808000, 456000, 5832000, size=2400)
    assert t.a == 10.0 and t.e == -10.0 and t.c == 432000 and t.f == 5832000
