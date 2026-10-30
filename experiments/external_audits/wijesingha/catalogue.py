"""Constants for the Wijesingha, Dzene and Wachendorf 2024 audit."""
from __future__ import annotations

PAPER = "wijesingha"
REPO_URL = "https://github.com/jmatics/landsat-croptype-spatial-temporal-transferability"
REPO_RAW = "https://raw.githubusercontent.com/jmatics/landsat-croptype-spatial-temporal-transferability/main/"
NOTEBOOKS = {
    "NH": "00_1_SR_VI_NH_Data_model_training_TF_BCNN_RF.ipynb",
    "WE": "00_2_SR_VI_WE_Data_model_training_TF_BCNN_RF.ipynb",
}
DOWNLOAD_SCRIPT = "L578_WE_Download.py"

# geoBoundaries gbOpen DEU ADM2 (pinned commit, as in the stoian audit): 'Kassel' is the
# Regierungsbezirk North Hesse, 'Weser-Ems' the Weser-Ems statistical region. The authors'
# VG25 district shapefile is not in their repository.
GEOB_URL = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/DEU/ADM2/geoBoundaries-DEU-ADM2.geojson"
REGION_NAME = {"NH": "Kassel", "WE": "Weser-Ems"}
WORLDCOVER = "ESA/WorldCover/v200"  # class 40 = cropland (2021 map, used as a static cropland proxy)
CROPLAND_CLASS = 40

TRAIN_YEARS = list(range(2013, 2019))  # the notebooks keep 2010-2018 and drop 2010-2012 from training
TEST_YEARS = [2010, 2011, 2012]
YEARS = TEST_YEARS + TRAIN_YEARS
BIMONTH_STARTS = [3, 5, 7, 9]  # month = [*range(3, 10, 2)] in the download script: Mar-Apr, May-Jun, Jul-Aug, Sep-Oct
MAX_CLOUD_COVER = 60  # filterMetadata('CLOUD_COVER', 'less_than', 60)
SENSORS = {
    "L5": {"id": "LANDSAT/LT05/C02/T1_L2", "sr": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"], "cirrus": False},
    "L7": {"id": "LANDSAT/LE07/C02/T1_L2", "sr": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"], "cirrus": False},
    "L8": {"id": "LANDSAT/LC08/C02/T1_L2", "sr": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], "cirrus": True},
}
N_POINTS = 2000  # random cropland points per region
SEED = 42
PAPER_NH_RF_LOSS = 0.120  # mF1 loss of the NH random forest under temporal, spatial and spatio-temporal transfer (report card)
