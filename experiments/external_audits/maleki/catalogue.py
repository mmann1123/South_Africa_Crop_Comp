"""Constants for the Maleki and Rahdari 2026 audit (Remote Sensing 18:1234, doi 10.3390/rs18081234)."""
from __future__ import annotations

PAPER = "maleki"
YEAR = 2021
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
COLLECTION_ALT = "COPERNICUS/S2_SR"  # the paper says only "Level-2A ... via GEE"; counts are checked on both
WDPA = "WCMC/WDPA/current/polygons"
WORLDCOVER = "ESA/WorldCover/v200"  # 2021 map, 10 m

# Sites as the Ramsar polygons that Figure 1 of the paper outlines (WDPA reports 850 and 2558 km2).
SITES = {
    "Camargue": "Camargue",
    "Champagne": "Etangs de la Champagne humide",
}
RAMSAR_DESIG = "Ramsar"

# Preprocessing recipe quoted from section 2.2.2 of the paper.
CLOUDY_PCT_MAX = 50  # CLOUDY_PIXEL_PERCENTAGE < 50 (granule level)
CLDPRB_MAX = 1  # MSK_CLDPRB < 1
SNWPRB_MAX = 10  # MSK_SNWPRB < 10
SCL_MASKED = (2, 3, 8, 9, 10)  # dark area, cloud shadow, cloud medium, cloud high, cirrus
GRID_STEP = 6  # days
GRID_N = 60  # 1 Jan to 31 Dec 2021: grid days 1, 7, ..., 355
HALF_WINDOW = 3  # a grid step counts as observed if a valid acquisition lies within +-3 days

# WorldCover classes used for the strata (the paper's OW / AV / TV proxies).
WC_CLASSES = {80: "water", 90: "herbaceous wetland", 10: "tree cover", 30: "grassland", 40: "cropland"}

SEED = 42
N_POINTS = 2000  # random points per site
SCALE = 10
CHUNK = 250  # points per Earth Engine call

# Table 4 and Table 5 of the paper (OA %), transcribed from the PDF text.
FEATURES = ["bands", "indices", "band_ind"]
TABLE4 = {  # (train, test) -> {algorithm: [OA bands, OA indices, OA band_ind]}
    ("Camargue", "Champagne"): {"IT": [84.3, 87.5, 90.0], "XGB": [89.7, 85.0, 86.3], "RF": [86.2, 85.4, 83.1], "MLP": [75.2, 82.0, 81.4]},
    ("Champagne", "Camargue"): {"IT": [76.9, 84.2, 84.0], "XGB": [83.8, 72.0, 78.6], "RF": [82.7, 73.4, 77.0], "MLP": [75.4, 80.5, 82.4]},
}
TABLE5 = {  # within-site baseline OA %
    "Camargue": {"IT": [95.2, 94.1, 95.7], "XGB": [93.5, 93.7, 93.8], "RF": [92.1, 92.5, 92.5], "MLP": [92.1, 91.4, 89.8]},
    "Champagne": {"IT": [96.8, 95.2, 95.5], "XGB": [95.6, 95.8, 95.9], "RF": [94.9, 95.3, 95.4], "MLP": [94.6, 95.3, 95.0]},
}
