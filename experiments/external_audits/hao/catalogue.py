"""Constants for the Hao et al. 2020 audit (Sci. Total Environ. 733:138869, doi 10.1016/j.scitotenv.2020.138869)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from experiments.external_audits.common import paths

PAPER = "hao"
DOI = "10.1016/j.scitotenv.2020.138869"
SEED = 42
PDF = paths.REPO_ROOT / "writeup" / "literature" / "Transfer Learning for Crop classification with Cropland Data Layer data (CDL) as training samples.pdf"
FIG_PAGES = {"HS": 6, "AB": 10, "NE": 12}  # 1-based PDF page of Fig. 4 (Hengshui), Fig. 6 (Alberta), Fig. 8 (Nebraska)

# Facts pinned from the paper text (Sections 2 and 3).
DOY_START, WINDOW_DAYS, N_WINDOWS = 60, 15, 18  # windows DOY 60-75, ..., 315-330; "DOY 60 and 330"
WINDOW_ENDS = [DOY_START + WINDOW_DAYS * (k + 1) for k in range(N_WINDOWS)]  # x axis of Figs. 4, 6, 8: 75 ... 330
TRAIN_YEAR, TARGET_YEAR = 2017, 2018
N_TRAIN_SAMPLES = 8448
# Table 1: visually selected samples per crop
TABLE1 = {"corn": 2759, "soybean": 3135, "cotton": 854, "spring_wheat": 450, "canola": 120, "winter_wheat": 1130}
GRID_M, PER_CELL = 100_000, 30  # 100 km x 100 km grid, up to 30 pixels per crop and grid cell
CONF_MIN = 95  # CDL confidence threshold
FIELD_RADIUS_PX = 3  # 7 x 7 Landsat pixels (fields larger than 4 ha): all 49 pixels must share the crop class
GRID_CRS = "EPSG:5070"  # CONUS Albers, for the 100 km grid

# CDL 2017 codes per crop. Winter wheat includes the double-cropped classes that contain it.
CDL_CROPS = {
    "corn": [1], "soybean": [5], "cotton": [2], "spring_wheat": [23], "canola": [31],
    "winter_wheat": [24, 26, 225, 236, 238],
}
CROP_IDS = {c: i + 1 for i, c in enumerate(CDL_CROPS)}

# Study regions (Section 2) and the HLS tiles the paper lists for 2018.
REGIONS = {
    "AB": {"box": [-112.0, 52.5, -110.0, 54.0], "tiles": ["12UVD", "12UVE", "12UWD", "12UWE"], "crops": ["canola", "spring_wheat"]},
    "NE": {"box": [-101.0, 40.5, -99.0, 42.0], "tiles": ["14TLL", "14TLM", "14TML", "14TMM"], "crops": ["corn", "soybean"]},
    "HS": {"box": [115.0, 37.0, 117.0, 39.0], "tiles": [], "crops": ["cotton", "corn", "winter_wheat"]},
}
CDL = "USDA/NASS/CDL/{year}"
ACI = "AAFC/ACI"
ACI_CODES = {"canola": 153, "spring_wheat": 146}  # 'Canola and Rapeseed', 'Spring Wheat'
GFSAD = "projects/sat-io/open-datasets/GFSAD/GCEP30"  # the paper's cropland mask (30 m); 2 = cropland
GFSAD_CROP = 2
HLS = {"S": "NASA/HLS/HLSS30/v002", "L": "NASA/HLS/HLSL30/v002"}
FMASK_CLOUD_BITS = 0b1111  # cirrus, cloud, adjacent to cloud/shadow, cloud shadow
FMASK_SNOW_BIT = 0b10000
L8 = "LANDSAT/LC08/C02/T1_L2"  # the paper used C01 T1_SR, which Earth Engine has since removed
L8_CLOUD_BITS = 0b11010  # QA_PIXEL: dilated cloud (1), cloud (3), cloud shadow (4)
S2_L1C = "COPERNICUS/S2_HARMONIZED"
S2_CSP = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
QA60_BITS = 0b110000000000  # opaque cloud (10) and cirrus (11): the L1C cloud mask
CSP_CLEAR = 0.65  # Cloud Score+ cs_cdf: 0.65 is the value Google suggests for "clear" with moderate tolerance
CSP_STRICT = 0.80

N_POINTS = {"AB": 2000, "NE": 2000, "HS": 2000}
STREAM_CONTROL_N = 1000  # points per control domain for the L8 + S2 L1C stream (AB, NE, CONUS)


def window_dates(year: int) -> list[tuple[dt.date, dt.date]]:
    """[start, end) calendar dates of the 18 composite windows (day of year 60 is 1 March in 2017 and 2018)."""
    jan1 = dt.date(year, 1, 1)
    return [(jan1 + dt.timedelta(days=DOY_START - 1 + WINDOW_DAYS * k), jan1 + dt.timedelta(days=DOY_START - 1 + WINDOW_DAYS * (k + 1))) for k in range(N_WINDOWS)]
