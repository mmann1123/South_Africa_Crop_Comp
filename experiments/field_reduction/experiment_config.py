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
    "ensemble_ml": "Ensemble (ML)",
}

# Classical ML models used in the ensemble majority vote
ML_ENSEMBLE_MODELS = ["xgboost_field", "base_lgbm_pixel", "base_lr_pixel"]

# Map experiment model names -> display names for results
DISPLAY_NAME_MAP = {
    "tabnet_pixel": "TabNet (pixel)",
    "ltae_field": "L-TAE Field (field)",
    "ltae_pixel": "L-TAE (pixel)",
    "xgboost_field": "XGBoost (field)",
    "base_lgbm_pixel": "Base LightGBM (pixel)",
    "base_lr_pixel": "Base LR (pixel)",
    "xgboost_field_l2": "XGBoost L2 (field)",
    "base_lgbm_pixel_l2": "LightGBM L2 (pixel)",
    "ltae_sparse_pixel": "L-TAE-S (pixel)",
    "ltae_linear_pixel": "L-TAE-Lin (pixel)",
}

# Training Level and Feature Type (matches model_comparison.csv)
MODEL_TRAINING_LEVEL = {
    "tabnet_pixel": "pixel",
    "ltae_field": "field",
    "ltae_pixel": "pixel",
    "xgboost_field": "field",
    "base_lgbm_pixel": "pixel",
    "base_lr_pixel": "pixel",
    "xgboost_field_l2": "field",
    "base_lgbm_pixel_l2": "pixel",
    "ltae_sparse_pixel": "pixel",
    "ltae_linear_pixel": "pixel",
}

MODEL_FEATURE_TYPE = {
    "tabnet_pixel": "raw pixel (band x month)",
    "ltae_field": "raw temporal (field-averaged)",
    "ltae_pixel": "raw pixel (temporal sequence)",
    "xgboost_field": "xr_fresh time-series",
    "base_lgbm_pixel": "raw pixel (band x month)",
    "base_lr_pixel": "raw pixel (band x month)",
    "xgboost_field_l2": "xr_fresh time-series",
    "base_lgbm_pixel_l2": "raw pixel (band x month)",
    "ltae_sparse_pixel": "raw pixel (temporal sequence)",
    "ltae_linear_pixel": "raw pixel (temporal sequence)",
}

# L2 regularization strength for _l2 model variants
L2_REG_LAMBDA = 1.0

# Model -> conda environment
MODEL_ENV_MAP = {
    "tabnet_pixel": DEEP_FIELD_PYTHON,
    "ltae_field": DEEP_FIELD_PYTHON,
    "ltae_pixel": DEEP_FIELD_PYTHON,
    "xgboost_field": ML_FIELD_PYTHON,
    "base_lgbm_pixel": ML_FIELD_PYTHON,
    "base_lr_pixel": ML_FIELD_PYTHON,
    "xgboost_field_l2": ML_FIELD_PYTHON,
    "base_lgbm_pixel_l2": ML_FIELD_PYTHON,
    "ltae_sparse_pixel": DEEP_FIELD_PYTHON,
    "ltae_linear_pixel": DEEP_FIELD_PYTHON,
}

# Model -> training script
MODEL_SCRIPT_MAP = {
    "tabnet_pixel": os.path.join(EXPERIMENT_DIR, "train_tabnet_pixel.py"),
    "ltae_field": os.path.join(EXPERIMENT_DIR, "train_ltae_field.py"),
    "ltae_pixel": os.path.join(EXPERIMENT_DIR, "train_ltae_pixel.py"),
    "xgboost_field": os.path.join(EXPERIMENT_DIR, "train_xgboost_field.py"),
    "base_lgbm_pixel": os.path.join(EXPERIMENT_DIR, "train_base_ml.py"),
    "base_lr_pixel": os.path.join(EXPERIMENT_DIR, "train_base_ml.py"),
    "xgboost_field_l2": os.path.join(EXPERIMENT_DIR, "train_xgboost_field.py"),
    "base_lgbm_pixel_l2": os.path.join(EXPERIMENT_DIR, "train_base_ml.py"),
    "ltae_sparse_pixel": os.path.join(EXPERIMENT_DIR, "train_ltae_sparse_pixel.py"),
    "ltae_linear_pixel": os.path.join(EXPERIMENT_DIR, "train_ltae_linear_pixel.py"),
}

# OOS comparison CSV (for baseline metrics)
OOS_COMPARISON_CSV = os.path.join(
    REPO_ROOT, "out_of_sample", "scoring_results", "model_comparison.csv"
)
