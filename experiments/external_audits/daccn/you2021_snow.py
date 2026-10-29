"""Earth Engine side of the DACCN snow check: per-step snow flag on the valid composites of the
You et al. 2021 recipe, plus Sentinel-2 L2A SCL and MODIS MOD10A1 cross-check bands for the same
10-day windows. Reuses the recipe's scaling, cloud mask and day-of-year windows unchanged.
"""
from __future__ import annotations

import ee

from . import snowrule, you2021_recipe as R

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
MODIS = "MODIS/061/MOD10A1"
BANDS = ["green", "swir1", "nir"]


def _window(start_md, end_md, year):
    start = ee.Date(f"{year}-{start_md}")
    end = ee.Date(f"{year}-{end_md}") if end_md else ee.Date.fromYMD(year + 1, 1, 1)
    return start, end


def _flag(med: ee.Image) -> ee.Image:
    ndsi = med.normalizedDifference(["green", "swir1"])
    return ndsi.gt(snowrule.NDSI_MIN).And(med.select("nir").gt(snowrule.NIR_MIN)).And(med.select("green").gt(snowrule.GREEN_MIN))


def snow_image(region: ee.Geometry, year: int, variant: str) -> ee.Image:
    """Bands valid_kk, snow_kk, scl_kk, modis_kk for the 22 interpolated steps (DOY 90..300)."""
    start_md, end_md = R.VARIANT_WINDOWS[variant]
    start, end = _window(start_md, end_md, year)
    l1c = ee.ImageCollection(R.S2).filterBounds(region).filterDate(start, end).map(R.sentinel2toa).map(R.cloud_mask).select(BANDS)
    # L2A and MODIS windows follow the variant's date range too, so every cross-check shares the composite's window
    scl = ee.ImageCollection(S2_SR).filterBounds(region).filterDate(start, end).select("SCL")
    mod = ee.ImageCollection(MODIS).filterBounds(region).filterDate(start, end).select("NDSI_Snow_Cover")
    bands = []
    for k, i in enumerate(range(R.TARGETS[0], R.TARGETS[1] + 1)):
        d0 = i * 10
        flt = ee.Filter.dayOfYear(d0, d0 + 10)
        sub = l1c.filter(flt)
        n = sub.size()
        med = ee.Image(ee.Algorithms.If(n.gt(0), sub.median(), ee.Image.constant([0, 0, 0]).rename(BANDS).updateMask(ee.Image(0))))
        valid = med.select("green").mask().unmask(0)
        snow = _flag(med).unmask(0).And(valid)
        s_sub = scl.filter(flt)
        s_n = s_sub.size()
        s_snow = ee.Image(ee.Algorithms.If(s_n.gt(0), s_sub.map(lambda im: im.eq(11)).max().unmask(0), ee.Image(0)))
        s_cnt = ee.Image(ee.Algorithms.If(s_n.gt(0), s_sub.count().unmask(0), ee.Image(0)))
        s_val = s_snow.where(s_cnt.eq(0), -1)  # -1 = no Sentinel-2 L2A observation at this pixel in the window
        m_sub = mod.filter(flt)
        m_n = m_sub.size()
        m_val = ee.Image(ee.Algorithms.If(m_n.gt(0), m_sub.max().unmask(-1), ee.Image(-1)))
        bands += [valid.rename(f"valid_{k:02d}"), snow.rename(f"snow_{k:02d}"), s_val.rename(f"scl_{k:02d}"), m_val.rename(f"modis_{k:02d}")]
    return ee.Image.cat(bands).toInt16()


def l2a_coverage(region: ee.Geometry, year: int) -> dict:
    """Scene counts over a region and year for L1C, L2A and MOD10A1 (Apr 1 to Oct 31)."""
    out = {}
    for name, cid in (("l1c", R.S2), ("l2a", S2_SR), ("mod10a1", MODIS)):
        c = ee.ImageCollection(cid).filterBounds(region).filterDate(f"{year}-04-01", f"{year}-11-01")
        out[name] = c.size().getInfo()
    return out
