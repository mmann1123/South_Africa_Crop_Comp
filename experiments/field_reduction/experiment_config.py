"""Centralized configuration for the field reduction experiment."""

import os
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
DEEP_LEARN_SRC = os.path.join(REPO_ROOT, "deep_learn", "src")
sys.path.insert(0, DEEP_LEARN_SRC)

from config import (
    MERGED_DL_PATH,
    MERGED_DL_TEST_PATH,
    FINAL_DATA_PATH,
    COMBINED_TEST_FEATURES_PATH,
    XGB_TUNER_DIR,
    MODEL_DIR,
    TABNET_DIR,
    TEST_LABELS_DIR,
    TEST_REGION,
    DEEP_FIELD_PYTHON,
    ML_FIELD_PYTHON,
)

# Experiment output directories
MODELS_DIR = os.path.join(EXPERIMENT_DIR, "models")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# Experiment parameters
FRACTIONS = [0.25, 0.50, 0.75]
SUBSAMPLE_SEED = 42
SEEDS_ENSEMBLE = [42, 101, 202, 303, 404]

# Ground truth path for OOS scoring
TEST_LABELS_GEOJSON = os.path.join(
    TEST_LABELS_DIR,
    f"ref_fusion_competition_south_africa_test_labels_{TEST_REGION}",
    "labels.geojson",
)

# Map experiment model names -> model_comparison.csv names (for 1.00 baseline)
BASELINE_NAME_MAP = {
    "tabnet_pixel": "TabNet (pixel)",
    "ltae_field": "L-TAE Field (field)",
    "ltae_pixel": "L-TAE (pixel)",
    "xgboost_field": "XGBoost (field)",
    "base_lgbm_pixel": "Base LightGBM (pixel)",
    "base_lr_pixel": "Base LR (pixel)",
}

# Map experiment model names -> display names for results
DISPLAY_NAME_MAP = {
    "tabnet_pixel": "TabNet (pixel)",
    "ltae_field": "L-TAE Field (field)",
    "ltae_pixel": "L-TAE (pixel)",
    "xgboost_field": "XGBoost (field)",
    "base_lgbm_pixel": "Base LightGBM (pixel)",
    "base_lr_pixel": "Base LR (pixel)",
}

# Model -> conda environment
MODEL_ENV_MAP = {
    "tabnet_pixel": DEEP_FIELD_PYTHON,
    "ltae_field": DEEP_FIELD_PYTHON,
    "ltae_pixel": DEEP_FIELD_PYTHON,
    "xgboost_field": ML_FIELD_PYTHON,
    "base_lgbm_pixel": ML_FIELD_PYTHON,
    "base_lr_pixel": ML_FIELD_PYTHON,
}

# Model -> training script
MODEL_SCRIPT_MAP = {
    "tabnet_pixel": os.path.join(EXPERIMENT_DIR, "train_tabnet_pixel.py"),
    "ltae_field": os.path.join(EXPERIMENT_DIR, "train_ltae_field.py"),
    "ltae_pixel": os.path.join(EXPERIMENT_DIR, "train_ltae_pixel.py"),
    "xgboost_field": os.path.join(EXPERIMENT_DIR, "train_xgboost_field.py"),
    "base_lgbm_pixel": os.path.join(EXPERIMENT_DIR, "train_base_ml.py"),
    "base_lr_pixel": os.path.join(EXPERIMENT_DIR, "train_base_ml.py"),
}

# OOS comparison CSV (for baseline metrics)
OOS_COMPARISON_CSV = os.path.join(
    REPO_ROOT, "out_of_sample", "scoring_results", "model_comparison.csv"
)
