"""Earth Engine helpers for the Maleki audit: site polygons, the paper's mask recipe, per-date validity bands."""
from __future__ import annotations

import ee

from experiments.external_audits.maleki import catalogue as C


def site_geometry(name: str) -> ee.Geometry:
    fc = ee.FeatureCollection(C.WDPA).filter(
        ee.Filter.And(ee.Filter.stringContains("DESIG", C.RAMSAR_DESIG), ee.Filter.eq("NAME", C.SITES[name]))
    )
    return fc.geometry()


def collection(geom: ee.Geometry, product: str = C.COLLECTION) -> ee.ImageCollection:
    """Granules of 2021 over the site that pass the paper's CLOUDY_PIXEL_PERCENTAGE < 50 filter."""
    return (
        ee.ImageCollection(product)
        .filterBounds(geom)
        .filterDate(f"{C.YEAR}-01-01", f"{C.YEAR + 1}-01-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.CLOUDY_PCT_MAX))
    )


def _scl_clear(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    bad = ee.Image.constant(0)
    for c in C.SCL_MASKED:
        bad = bad.Or(scl.eq(c))
    return bad.Not()


def validity(img: ee.Image) -> ee.Image:
    """Bands cov (pixel has data), valid (all paper masks), valid_scl (SCL mask only), as 0/1."""
    cov = img.select("B2").mask().unmask(0).gt(0)
    scl_ok = _scl_clear(img).unmask(0)
    valid = scl_ok.And(img.select("MSK_CLDPRB").lt(C.CLDPRB_MAX)).And(img.select("MSK_SNWPRB").lt(C.SNWPRB_MAX)).And(cov)
    # unmask outside the granule footprint too (sameFootprint=False), else sampleRegions drops every point
    return ee.Image.cat([cov.rename("cov"), valid.rename("valid"), scl_ok.And(cov).rename("valid_scl")]).unmask(0, False).toInt8()


def per_date_stack(col: ee.ImageCollection, dates: list[str]) -> ee.Image:
    """One 3-band validity image per acquisition date (same-day granules merged by max), as 3 * len(dates) bands."""
    out = None
    for i, d in enumerate(dates):
        day = col.filterDate(d, ee.Date(d).advance(1, "day"))
        im = day.map(validity).max().rename([f"c{i}", f"v{i}", f"s{i}"])
        out = im if out is None else out.addBands(im)
    return out
