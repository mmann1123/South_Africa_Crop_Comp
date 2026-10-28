"""Faithful Earth Engine port of the compositing and interpolation stage of
You et al. 2021 `GEE_code_for_crop_classification.js` (the recipe DACCN states
it followed), extended with per-pixel validity and zero-fill counters.

Deviations from the deposited script, all documented in the DACCN README:
one carrier band ('red') instead of ten indices (zero fill applies to every
band identically, so one suffices to detect it); no Savitzky-Golay step (the
module is private and smoothing only spreads zeros); counters added.
"""
from __future__ import annotations

import ee

S2 = "COPERNICUS/S2_HARMONIZED"
CLOUD_SCORE_THRESHOLD = 0.2
N_COMPOSITES = 37  # DOY 0, 10, ..., 360
TARGETS = (9, 30)  # inclusive list indices interpolated by the script
SEARCH = 9
VARIANT_WINDOWS = {"A": ("01-01", None), "B": ("04-01", "11-01")}  # B: imagery April 1 .. October 31

_SRC = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "QA60"]
_DST = ["aerosol", "blue", "green", "red", "red1", "red2", "red3", "nir", "red4", "h2o", "cirrus", "swir1", "swir2", "QA60"]


def sentinel2toa(img: ee.Image) -> ee.Image:
    return img.select(_SRC, _DST).divide(10000).toDouble().set("system:time_start", img.get("system:time_start"))


def _rescale(img: ee.Image, lo: float, hi: float) -> ee.Image:
    return img.subtract(lo).divide(hi - lo)


def simple_cloud_score(toa: ee.Image) -> ee.Image:
    """Hancher/Hewig/Housman simple cloud score exactly as in the script."""
    score = ee.Image(1)
    score = score.min(_rescale(toa.select("blue"), 0.1, 0.5))
    score = score.min(_rescale(toa.select("aerosol"), 0.1, 0.3))
    score = score.min(_rescale(toa.select("aerosol").add(toa.select("cirrus")), 0.15, 0.2))
    score = score.min(_rescale(toa.select("red").add(toa.select("green")).add(toa.select("blue")), 0.2, 0.8))
    ndmi = toa.normalizedDifference(["red4", "swir1"])
    score = score.min(_rescale(ndmi, -0.1, 0.1))
    ndsi = toa.normalizedDifference(["green", "swir1"])
    score = score.min(_rescale(ndsi, 0.8, 0.6))
    return score.rename("cloud_score")


def cloud_mask(toa: ee.Image) -> ee.Image:
    cloud = simple_cloud_score(toa).gt(CLOUD_SCORE_THRESHOLD)
    return toa.updateMask(cloud.eq(0))


def masked_collection(region: ee.Geometry, year: int, variant: str) -> ee.ImageCollection:
    start_md, end_md = VARIANT_WINDOWS[variant]
    start = ee.Date(f"{year}-{start_md}")
    end = ee.Date(f"{year}-{end_md}") if end_md else ee.Date.fromYMD(year + 1, 1, 1)
    return (
        ee.ImageCollection(S2)
        .filterBounds(region)
        .filterDate(start, end)
        .map(sentinel2toa)
        .map(cloud_mask)
        .select(["red"])
    )


def composites(collection: ee.ImageCollection, year: int, region: ee.Geometry) -> ee.List:
    """37 ten-day median composites with band `valid` (1 where the median is
    unmasked) and properties doy, length, system:time_start, as in the script."""
    starts = ee.List.sequence(0, 360, 10)

    def make(start):
        start = ee.Number(start)
        filtered = collection.filter(ee.Filter.dayOfYear(start, start.add(10))).median()
        length = filtered.bandNames().length()
        valid = ee.Image(ee.Algorithms.If(length.gt(0), filtered.select(0).mask(), ee.Image(0))).rename("valid")
        valid = valid.updateMask(valid)  # masked where invalid, 1 where valid (mirrors updateMask(mask))
        return (
            valid.set("system:time_start", ee.Date.fromYMD(year, 1, 1).advance(start, "day").millis())
            .set("doy", start)
            .set("length", length)
        )

    return starts.map(make)


def _masked_dummy() -> ee.Image:
    return ee.Image(0).rename("valid").updateMask(ee.Image(0))


def counter_image(comps: ee.List, targets: tuple[int, int] = TARGETS, search: int = SEARCH) -> ee.Image:
    """Bands valid_XX, zero_XX for each target step plus n_valid and n_zero.

    zero = target invalid AND (before mosaic masked OR after mosaic masked),
    which is what target.unmask(Y).unmask(0) yields per pixel in the script.
    """
    lo, hi = targets
    bands = []
    for k, i in enumerate(range(lo, hi + 1)):
        target = ee.Image(comps.get(i))
        before = ee.ImageCollection.fromImages(comps.slice(i - search, i)).filter(ee.Filter.gt("length", 0))
        after = ee.ImageCollection.fromImages(comps.slice(i + 1, i + 1 + search).reverse()).filter(ee.Filter.gt("length", 0))
        before_ok = before.merge(ee.ImageCollection([_masked_dummy()])).mosaic().mask().unmask(0)
        after_ok = after.merge(ee.ImageCollection([_masked_dummy()])).mosaic().mask().unmask(0)
        valid = target.mask().unmask(0).rename(f"valid_{k:02d}")
        zero = valid.Not().And(before_ok.And(after_ok).Not()).rename(f"zero_{k:02d}")
        bands += [valid, zero]
    img = ee.Image.cat(bands)
    n_valid = img.select("valid_.*").reduce(ee.Reducer.sum()).rename("n_valid")
    n_zero = img.select("zero_.*").reduce(ee.Reducer.sum()).rename("n_zero")
    return img.addBands([n_valid, n_zero]).toInt16()


def counters(region: ee.Geometry, year: int, variant: str) -> ee.Image:
    return counter_image(composites(masked_collection(region, year, variant), year, region))


def smoke(region: ee.Geometry, year: int, variant: str) -> dict:
    img = counters(region, year, variant).select(["n_valid", "n_zero", "zero_00", "zero_21"])
    return img.reduceRegion(ee.Reducer.mean(), region, 10, maxPixels=1e8).getInfo()
