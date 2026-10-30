"""Earth Engine helpers: per-date cloud fractions over the sampled Gallo patches (cropland and all pixels)."""
from __future__ import annotations

import math

S2 = "COPERNICUS/S2_HARMONIZED"
CSPLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
WORLDCOVER = "ESA/WorldCover/v100/2020"
CS_CLEAR = 0.6  # cs_cdf >= 0.6 is clear (Cloud Score+ documented default), same as the timematch audit
SCALE_M = 10


def support_geometry(footprints):
    """Union of patch rectangles; footprints is a list of (left, bottom, right, top, crs) tuples."""
    import ee

    feats = [ee.Feature(ee.Geometry.Rectangle([l, b, r, t], proj=crs, geodesic=False)) for l, b, r, t, crs in footprints]
    return ee.FeatureCollection(feats).geometry()


def cropland_mask():
    import ee

    return ee.Image(WORLDCOVER).select("Map").eq(40)


def sensing_dates(support, year: int) -> list[str]:
    """Distinct UTC sensing dates (YYYYMMDD) of S2_HARMONIZED granules touching the support in a year."""
    import ee

    coll = ee.ImageCollection(S2).filterBounds(support).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    stamps = coll.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYYMMdd")).getInfo()
    return sorted(set(stamps))


def date_images(support, date: str):
    """Granules touching the support on one UTC date (YYYYMMDD), cs_cdf linked by system:index."""
    import ee

    start = ee.Date.parse("YYYYMMdd", date)
    coll = ee.ImageCollection(S2).filterBounds(support).filterDate(start, start.advance(1, "day"))
    return coll.linkCollection(ee.ImageCollection(CSPLUS), ["cs_cdf"])


def cloud_fractions(images, support, cropland) -> dict | None:
    """Mean Cloud Score+ cloudy / nodata flags over support-and-cropland and over all support pixels."""
    import ee

    ids = images.aggregate_array("system:index").getInfo()
    if not ids:
        return None
    mosaic = images.mosaic()
    observed = mosaic.select("B2").mask()
    cloudy = mosaic.select("cs_cdf").lt(CS_CLEAR).And(observed).rename("cloudy")
    nodata = observed.Not().rename("nodata").unmask(1)
    csv = mosaic.select("cs_cdf").unmask(-1).rename("cs_cdf")
    stack = ee.Image.cat([cloudy.unmask(0), nodata, csv])
    kw = dict(geometry=support, scale=SCALE_M, maxPixels=1e10, tileScale=4)
    crop = stack.updateMask(cropland).reduceRegion(ee.Reducer.mean(), **kw).getInfo()
    allp = stack.reduceRegion(ee.Reducer.mean(), **kw).getInfo()
    n = ee.Image.constant(1).rename("n").updateMask(cropland).reduceRegion(ee.Reducer.count(), **kw).getInfo()["n"]
    return {"n_images": len(ids), "image_ids": ids, "n_crop_pixels": n, "crop": crop, "all": allp}


def match_date(date: str, ee_dates: set, max_offset: int = 1) -> tuple[str | None, int | None]:
    """Nearest EE sensing date to a deposit date within max_offset days (ties go to the earlier one)."""
    import datetime as dt

    d = dt.datetime.strptime(date, "%Y%m%d").date()
    for off in sorted(range(-max_offset, max_offset + 1), key=lambda o: (abs(o), o)):
        e = (d + dt.timedelta(days=off)).strftime("%Y%m%d")
        if e in ee_dates:
            return e, off
    return None, None


def row_from_info(area: str, year: int, date: str, ee_date: str | None, offset: int | None, info: dict | None) -> dict:
    nan = float("nan")
    row = {"area": area, "year": year, "date": date, "ee_date": ee_date, "offset_days": offset, "n_images": 0, "image_ids": "", "n_crop_pixels": 0,
           "crop_cloudy": nan, "crop_nodata": nan, "crop_clear": nan, "all_cloudy": nan, "all_nodata": nan, "all_clear": nan}
    if info is None:
        return row
    c, a = info["crop"], info["all"]
    row.update({"n_images": int(info["n_images"]), "image_ids": ";".join(info["image_ids"]), "n_crop_pixels": int(info["n_crop_pixels"] or 0),
                "crop_cloudy": _f(c.get("cloudy")), "crop_nodata": _f(c.get("nodata")), "all_cloudy": _f(a.get("cloudy")), "all_nodata": _f(a.get("nodata"))})
    row["crop_clear"] = 1 - row["crop_cloudy"] - row["crop_nodata"] if not is_nan(row["crop_cloudy"]) else nan
    row["all_clear"] = 1 - row["all_cloudy"] - row["all_nodata"] if not is_nan(row["all_cloudy"]) else nan
    return row


def _f(v) -> float:
    return float("nan") if v is None else float(v)


def is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)
