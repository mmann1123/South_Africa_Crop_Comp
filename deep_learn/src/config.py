import os

# Working directory for scripts and outputs
WORKING_DIR = "/home/mmann1123/Documents/github/South_Africa_Crop_Comp/deep_learn/src"

# Repository root and centralized data output directory
REPO_ROOT = os.path.abspath(os.path.join(WORKING_DIR, "..", ".."))
DATA_OUTPUT_DIR = os.path.join(REPO_ROOT, "data")

# Raw per-band parquet data (external input)
DATA_DIR = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/S1c_data"

# Pre-computed xr_fresh time-series features (external input)
FEATURES_DIR = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/features/time series features extract"
# Test features directory (parent of FEATURES_DIR, contains testing_*.parquet)
TEST_FEATURES_DIR = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/features"

# Training label GeoJSON files (external input)
LABELS_DIR = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/ref_fusion_competition_south_africa_train_labels"

# Test label GeoJSON files (external input)
TEST_LABELS_DIR = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/ref_fusion_competition_south_africa_test_labels"

# Generated data paths (centralized in data/)
MERGED_DL_PATH = os.path.join(DATA_OUTPUT_DIR, "merged_dl_train.parquet")           # training only
MERGED_DL_TEST_PATH = os.path.join(DATA_OUTPUT_DIR, "merged_dl_test.parquet")       # test/inference only
FINAL_DATA_PATH = os.path.join(DATA_OUTPUT_DIR, "final_data.parquet")               # training only (xr_fresh features)

# Patch-level data
PATCH_GEOJSON_PATH = os.path.join(DATA_OUTPUT_DIR, "patch_level.geojson")
PATCH_DATA_PATH = os.path.join(DATA_OUTPUT_DIR, "patch_level_data.parquet")
COMBINED_FIELDS_PATH = os.path.join(LABELS_DIR, "combined_fields.geojson")

# Out-of-sample test data
COMBINED_TEST_FEATURES_PATH = os.path.join(DATA_OUTPUT_DIR, "combined_test_features.parquet")
TEST_PATCH_DATA_PATH = os.path.join(DATA_OUTPUT_DIR, "test_patch_data.parquet")
TEST_PATCHES_GEOJSON_PATH = os.path.join(DATA_OUTPUT_DIR, "test_patches.geojson")

# Model output directories
MODEL_DIR = os.path.join(WORKING_DIR, "models")
XGB_TUNER_DIR = os.path.join(WORKING_DIR, "xgb_tuner")
TABNET_DIR = os.path.join(WORKING_DIR, "saved_models_tabnet")
REPORTS_DIR = os.path.join(WORKING_DIR, "reports")

# Band names and regions
BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
TRAIN_REGIONS = ["34S_19E_258N", "34S_19E_259N"]
TEST_REGION = "34S_20E_259N"

# Month number to name mapping (no June due to cloud cover)
MONTH_MAP = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "07": "July", "08": "August", "09": "September",
    "10": "October", "11": "November", "12": "December",
}
