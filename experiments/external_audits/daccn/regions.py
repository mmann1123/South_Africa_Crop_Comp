"""DACCN study regions, sampling frames and transfer pairs."""
from __future__ import annotations

import ee

GAUL1 = "FAO/GAUL/2015/level1"
CDL = "USDA/NASS/CDL"
WORLDCOVER = "ESA/WorldCover/v100/2020"
USA = "United States of America"
CHINA = "China"

REGIONS = [  # (display name, country); GAUL ADM1_NAME spelling via GAUL_NAMES
    ("Minnesota", USA),
    ("Nebraska", USA),
    ("Wisconsin", USA),
    ("Iowa", USA),
    ("Heilongjiang", CHINA),
    ("Jilin", CHINA),
]
GAUL_NAMES = {"Heilongjiang": "Heilongjiang Sheng", "Jilin": "Jilin Sheng"}
# DACCN Table 1/2 transfer cases: source -> target (spatial, same year); temporal = each region 2018 -> 2019.
SPATIAL_PAIRS = [
    ("Minnesota", "Wisconsin"),
    ("Nebraska", "Iowa"),
    ("Minnesota", "Heilongjiang"),
    ("Nebraska", "Jilin"),
]
# CDL codes sampled in the US states. DACCN labels corn and soybean plus an
# "other" class from CDL pixels with confidence > 90; the exact "other" set is
# not stated in accessible text, so the frame is corn (1), soybeans (5) and a
# third class pooling the other major Corn Belt crops: spring wheat (23),
# winter wheat (24), alfalfa (36), oats (28), dry beans (42), sugarbeets (41).
CDL_CLASSES = {1: "corn", 5: "soybeans"}
CDL_OTHER = [23, 24, 36, 28, 42, 41]
CONFIDENCE_MIN = 90


def frame_label(country: str) -> str:
    return "cdl_conf90" if country == USA else "worldcover40"


def lookup(name: str, country: str) -> ee.Feature:
    fc = ee.FeatureCollection(GAUL1).filter(ee.Filter.eq("ADM1_NAME", GAUL_NAMES.get(name, name))).filter(ee.Filter.eq("ADM0_NAME", country))
    n = fc.size().getInfo()
    if n != 1:
        raise ValueError(f"GAUL level-1 lookup for {name}, {country} returned {n} features, expected 1")
    return fc.first()


def frame_image(country: str, year: int) -> tuple[ee.Image, str, dict[int, str]]:
    """Class image (band 'class', 0 = outside frame) and the class legend."""
    if country == USA:
        cdl = ee.Image(f"{CDL}/{year}")
        crop = cdl.select("cropland")
        conf = cdl.select("confidence")
        cls = ee.Image(0)
        for code, _ in CDL_CLASSES.items():
            cls = cls.where(crop.eq(code), code)
        cls = cls.where(crop.remap(CDL_OTHER, [1] * len(CDL_OTHER), 0).eq(1), 99)
        cls = cls.updateMask(conf.gt(CONFIDENCE_MIN)).updateMask(cls.gt(0)).rename("class")
        legend = {**CDL_CLASSES, 99: "other_cdl_pooled"}
        return cls, frame_label(country), legend
    wc = ee.Image(WORLDCOVER).select("Map")
    cls = ee.Image(40).updateMask(wc.eq(40)).rename("class")
    return cls, frame_label(country), {40: "worldcover_cropland"}
