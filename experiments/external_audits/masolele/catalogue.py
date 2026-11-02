"""Constants for the Masolele et al. 2021 audit (RSE 264:112600, open access https://edepot.wur.nl/551332)."""
from __future__ import annotations

PAPER = "masolele"
PDF_URL = "https://edepot.wur.nl/551332"

# Reference data: De Sy et al. 2019 follow-up land use on the FAO FRA-2010 Remote Sensing Survey,
# 10 x 10 km sample units at every 1 degree confluence point, restricted to Hansen et al. (2013)
# loss 2000-2005. The labelled units are not public, so the audit samples Hansen loss pixels
# (lossyear 1-5) inside the 10 x 10 km unit at every tropical confluence point.
HANSEN = "UMD/hansen/global_forest_change_2023_v1_11"
LOSS_YEARS = (1, 5)  # 2001-2005
LAT_MAX = 23  # confluence latitudes -23..23 (the paper maps tropical countries only)
CONTINENTS = {  # lon ranges of confluence points, inclusive
    "Latin America": (-118, -34),
    "Africa": (-18, 51),
    "Asia": (60, 155),
}
BOX_HALF_DEG = 0.045  # about 10 x 10 km at the equator
PIXELS_PER_UNIT = 3
SEED = 42

# Satellite data: Landsat 5 and 7 SR, scenes under 50 percent cloud, QA band cloud filter,
# January-June and July-December median composites, 2000-2005 (12 composites).
YEARS = list(range(2000, 2006))
HALVES = (1, 2)
MAX_CLOUD_COVER = 50
SR = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
SENSORS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",  # Collection 1 SR, which the paper used, is retired; C02 L2 is its successor
    "L7": "LANDSAT/LE07/C02/T1_L2",
}
CHUNK = 400  # points per sampleRegions request
