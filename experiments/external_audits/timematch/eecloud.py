"""Earth Engine helpers: per-date cloud fractions over cropland in a TimeMatch tile."""
from __future__ import annotations

import math
from pathlib import Path

S2 = "COPERNICUS/S2_HARMONIZED"
CSPLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S2CP = "COPERNICUS/S2_CLOUD_PROBABILITY"
WORLDCOVER = "ESA/WorldCover/v100/2020"
CS_CLEAR = 0.6  # cs_cdf >= 0.6 is clear (Cloud Score+ documented default)
P40 = 40
P16 = 16
SCALE_M = 60  # reduction scale; cropland fractions over a 110 km tile


def support_geometry(tile_dir: Path):
    """Tile footprint from the shipped blocks shapefile as an ee.Geometry rectangle
    in the tile's native UTM CRS. (Blocks are a 10x10 grid over the whole tile.)"""
    import ee
    import geopandas as gpd

    blocks = gpd.read_file(Path(tile_dir) / "blocks.shp")
    minx, miny, maxx, maxy = [float(v) for v in blocks.total_bounds]
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj=str(blocks.crs), geodesic=False), str(blocks.crs)


def cropland_mask():
    import ee

    return ee.Image(WORLDCOVER).select("Map").eq(40)


def date_images(tile: str, date: str):
    """S2_HARMONIZED granules of one MGRS tile on one UTC date, with cs_cdf and
    probability bands linked by system:index."""
    import ee

    start = ee.Date(date)
    coll = (
        ee.ImageCollection(S2)
        .filterDate(start, start.advance(1, "day"))
        .filter(ee.Filter.eq("MGRS_TILE", tile))
    )
    coll = coll.linkCollection(ee.ImageCollection(CSPLUS), ["cs_cdf"])
    coll = coll.linkCollection(ee.ImageCollection(S2CP), ["probability"])
    return coll


def cloud_fractions(images, support, cropland) -> dict | None:
    """Mean cloud flags over support ∩ cropland. Returns None when no images."""
    import ee

    ids = images.aggregate_array("system:index").getInfo()
    if not ids:
        return None
    mosaic = images.mosaic()
    observed = mosaic.select("B2").mask()
    cs_cloudy = mosaic.select("cs_cdf").lt(CS_CLEAR).rename("csplus_cloudy")
    p40 = mosaic.select("probability").gte(P40).rename("s2cp40")
    p16 = mosaic.select("probability").gte(P16).rename("s2cp16")
    nodata = observed.Not().rename("nodata").unmask(1)  # 1 where no observation
    ones = ee.Image.constant(1).rename("n_pixels")
    stack = ee.Image.cat([cs_cloudy, p40, p16, nodata]).updateMask(cropland)
    means = stack.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=support, scale=SCALE_M, maxPixels=1e10, tileScale=4
    )
    count = ones.updateMask(cropland).reduceRegion(
        reducer=ee.Reducer.count(), geometry=support, scale=SCALE_M, maxPixels=1e10, tileScale=4
    )
    info = means.getInfo()
    info["n_pixels"] = count.getInfo()["n_pixels"]
    info["n_images"] = len(ids)
    info["image_ids"] = ids
    return info


def row_from_info(country: str, tile: str, date: str, tile_cloudy_pct: float, info: dict | None, support_kind: str) -> dict:
    nan = float("nan")
    row = {
        "country": country,
        "tile": tile,
        "date": date,
        "tile_cloudy_pct": tile_cloudy_pct,
        "n_images": 0,
        "image_ids": "",
        "n_pixels": 0,
        "frac_cloudy_csplus": nan,
        "frac_cloudy_s2cp40": nan,
        "frac_cloudy_s2cp16": nan,
        "frac_nodata": nan,
        "support_kind": support_kind,
    }
    if info is None:
        return row
    row.update(
        {
            "n_images": int(info["n_images"]),
            "image_ids": ";".join(info.get("image_ids", [])),
            "n_pixels": int(info.get("n_pixels") or 0),
            "frac_cloudy_csplus": _f(info.get("csplus_cloudy")),
            "frac_cloudy_s2cp40": _f(info.get("s2cp40")),
            "frac_cloudy_s2cp16": _f(info.get("s2cp16")),
            "frac_nodata": _f(info.get("nodata")),
        }
    )
    return row


def _f(v) -> float:
    return float("nan") if v is None else float(v)


def is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)
