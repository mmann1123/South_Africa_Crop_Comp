"""Pure functions for the L1C granule effective clear count."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform


def parse_footprint(footprint: str):
    """OData Footprint string (geography'SRID=4326;POLYGON ((lon lat, ...))') to a shapely geometry."""
    return wkt.loads(footprint.split(";", 1)[1].rstrip("'"))


def footprint_share(footprint_wgs84, tile_bounds: tuple[float, float, float, float], epsg: int) -> float:
    """Share of the nominal tile square (bounds in the tile's UTM CRS) covered by a WGS84 footprint."""
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    tile = box(*tile_bounds)
    return float(transform(to_utm, footprint_wgs84).intersection(tile).area / tile.area)


def effective_clear(df: pd.DataFrame, start: str, end: str) -> float:
    """Sum over granules of (1 - cloudCover/100) x footprint share, for granules sensed in [start, end].

    `df` has columns date (datetime-like), cloud_cover (percent) and share (0 to 1).
    """
    d = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    w = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    sub = df[w]
    return float(((1.0 - sub["cloud_cover"] / 100.0) * sub["share"]).sum())
