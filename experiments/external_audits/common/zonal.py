"""Field-polygon zonal statistics on regular raster grids (EOPatch tiles)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio import features


def grid_from_bbox(minx: float, miny: float, maxx: float, maxy: float, size: int = 2400) -> Affine:
    """North-up affine for a square grid of `size` pixels covering the bbox."""
    res_x = (maxx - minx) / size
    res_y = (maxy - miny) / size
    return Affine(res_x, 0.0, minx, 0.0, -res_y, maxy)


def field_index_map(
    gdf: gpd.GeoDataFrame, transform: Affine, shape: tuple[int, int], crs
) -> tuple[np.ndarray, pd.DataFrame]:
    """Rasterise fields to an int32 index raster (-1 = background).

    Returns the raster and a frame with field_pos (0..n-1), field_id (gdf index),
    n_pixels, keeping only fields that cover at least one pixel. The number of
    fields dropped for covering no pixel is in frame.attrs['n_dropped'].
    """
    g = gdf.to_crs(crs) if gdf.crs is not None and str(gdf.crs) != str(crs) else gdf
    shapes = ((geom, i) for i, geom in enumerate(g.geometry) if geom is not None and not geom.is_empty)
    raw = features.rasterize(
        shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32", all_touched=False
    )
    counts = np.bincount(raw[raw >= 0], minlength=len(g))
    keep = np.flatnonzero(counts > 0)
    remap = -np.ones(len(g) + 1, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    idx = np.where(raw >= 0, remap[np.clip(raw, 0, len(g))], -1).astype(np.int32)
    fields = pd.DataFrame(
        {
            "field_pos": np.arange(len(keep), dtype=np.int32),
            "field_id": g.index.values[keep],
            "n_pixels": counts[keep].astype(np.int64),
        }
    )
    fields.attrs["n_dropped"] = int(len(g) - len(keep))
    return idx, fields


def field_means(cube: np.ndarray, index_map: np.ndarray, n_fields: int) -> np.ndarray:
    """Per-field mean of a (T, H, W) cube; returns (T, n_fields)."""
    flat_idx = index_map.ravel()
    sel = flat_idx >= 0
    ids = flat_idx[sel]
    counts = np.bincount(ids, minlength=n_fields).astype(np.float64)
    out = np.empty((cube.shape[0], n_fields), dtype=np.float64)
    for t in range(cube.shape[0]):
        vals = cube[t].reshape(-1)[sel].astype(np.float64)
        out[t] = np.bincount(ids, weights=vals, minlength=n_fields) / counts
    return out


def field_fraction(cube_bool: np.ndarray, index_map: np.ndarray, n_fields: int) -> np.ndarray:
    """Per-field fraction of True in a boolean (T, H, W) cube; returns (T, n_fields)."""
    return field_means(cube_bool.astype(np.float64), index_map, n_fields)
