"""Constants for the Mai et al. 2025 audit (Remote Sensing 17:1494, doi 10.3390/rs17091494)."""
from __future__ import annotations

PAPER = "hexi"
DOI = "10.3390/rs17091494"
YEARS = [2021, 2022, 2023]  # 2022 is the paper's year; 2021 and 2023 bracket the QA60 check
PAPER_YEAR = 2022
SEED = 42

# Facts pinned from the paper (Sections 2.1 to 2.3, Table A1).
PAPER_EXTENT = (92.33, 36.48, 104.27, 42.8)  # 36 29'-42 48' N, 92 20'-104 16' E (lon_min, lat_min, lon_max, lat_max)
PAPER_RECIPE = "Sentinel-2 QA60 cloud removal, weekly maximum-value composite, linear interpolation of the removed values, calendar year 2022"
PAPER_PRODUCT = "not named; Table A1 lists B10 and B9, which only the L1C (TOA) collection carries"

# Earth Engine ids.
S2_L1C = "COPERNICUS/S2_HARMONIZED"
S2_SR = "COPERNICUS/S2_SR_HARMONIZED"  # QA60 check only
CSPLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CDL = "USDA/NASS/CDL"
WORLDCOVER = "ESA/WorldCover/v200"
GAUL1 = "FAO/GAUL/2015/level1"
GAUL2 = "FAO/GAUL/2015/level2"

# Source domain: the six states named in Section 2.1, the paper's five crops as CDL codes.
STATES = ["Iowa", "Missouri", "Kansas", "Minnesota", "North Dakota", "South Dakota"]
CDL_CLASSES = {1: "maize", 36: "alfalfa", 28: "oats", 31: "canola", 23: "spring wheat"}
CDL_CONF_MIN = 95  # "greater than 95%"
POINTS_PER_STATE_CLASS = 60
SAMPLE_SCALE_M = 300  # grid on which stratifiedSample draws points (30 m was too slow over whole states)

# Target domain: the five Gansu prefectures that make up the Hexi Corridor inside the paper's extent.
GANSU = "Gansu Sheng"
HEXI_PREFECTURES = ["Jiuquan", "Jiayuguan", "Zhangye", "Jinchang", "Wuwei"]
WC_CROPLAND = 40
N_TARGET_POINTS = 1800
TARGET = "Hexi"

# Weekly composites: 52 seven-day bins from 1 January (the paper does not give week edges; 31 December is left out).
N_WEEKS = 52
WEEK_BLOCK = 4  # weeks per Earth Engine call (13 hit the user memory limit)
CHUNK = 100  # points per call
SCALE_M = 10
CS_THRESHOLDS = {"cs50": 0.5, "cs65": 0.65}  # Cloud Score+ cs_cdf clear if >= threshold; 0.5 is the catalogue's general setting
CS_PRIMARY = "cs50"
QA60_CLOUD_BITS = (10, 11)  # opaque cloud, cirrus
SEASON_WEEKS = (13, 43)  # 0-based inclusive weeks 14 to 44, about 1 April to 31 October
