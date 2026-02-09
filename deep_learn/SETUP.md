# Setup and Run Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended for deep learning models (CPU works but is slow)
- ~15 GB disk space for data files and model outputs

## Install Dependencies

The install script auto-detects your CUDA version and installs PyTorch and TensorFlow with matching GPU support:

```bash
cd deep_learn
bash install.sh
```

It detects CUDA via `nvidia-smi` / `nvcc`, maps it to the correct PyTorch index URL (cu118, cu121, cu124, cu126), and falls back to CPU if no GPU is found.

To force a specific variant:

```bash
bash install.sh cpu       # CPU-only
bash install.sh cu124     # force CUDA 12.4
bash install.sh cu121     # force CUDA 12.1
bash install.sh cu118     # force CUDA 11.8
```

## Configure Data Paths

Edit `src/config.py` to match your data locations:

```python
DATA_DIR = "/path/to/S1c_data"              # per-band parquet files + GeoTIFFs
FEATURES_DIR = "/path/to/features/time series features extract"  # xr_fresh features
LABELS_DIR = "/path/to/train_labels"        # training GeoJSON labels
```

All scripts import paths from this single file.

## Data Preparation (Stage 0)

Run from `deep_learn/src/`:

```bash
cd deep_learn/src

# Step 1: Merge per-band parquets into single training + test files
python create_merged_dl_data.py
# Creates: merged_dl_258_259.parquet (~7.5M rows, training)
#          merged_dl_test_259N.parquet (~3.4M rows, inference only)

# Step 2: Merge xr_fresh features for classical ML
python create_final_data.py
# Creates: Data/final_data.parquet (~7.5M rows, 189 features)
```

Both scripts take a few minutes depending on disk speed.

## Running Models

All commands run from `deep_learn/src/`. Scripts are independent within each stage unless noted.

### Stage 1: Classical ML

Uses `Data/final_data.parquet`. Expected results: Kappa ~0.68-0.69, F1 ~0.76-0.77.

```bash
# XGBoost with Optuna hyperparameter tuning (75 trials)
python "Classical Machine Learning/Field Level/xg_boost_hyper.py"

# Voting and Stacking ensembles
python "Classical Machine Learning/Field Level/Ensemble - Voting and Stacking.py"

# SMOTE + meta-learner (run after Ensemble script for shared artifacts)
python "Classical Machine Learning/Field Level/SMOTE_meta.py"

# Pixel-level models with field aggregation
python "Classical Machine Learning/pixel_level/base_ml_models.py"
python "Classical Machine Learning/pixel_level/pixel_voting.py"

# Inference (requires trained ensemble models from above)
python "Classical Machine Learning/Field Level/inference_classical_ensemble.py"
```

### Stage 2: Pixel/Field-Level Deep Learning

Uses `merged_dl_258_259.parquet`. Best performer: CNN+BiLSTM (Kappa ~0.77, F1 ~0.84).

```bash
# CNN+BiLSTM ensemble (BEST MODEL) - trains 5 models, 25 epochs each
python "Deep Learning/Pixel_Field_Level/cnn_bilstm.py"

# Field-level evaluation of trained CNN+BiLSTM (run after cnn_bilstm.py)
python "Deep Learning/Pixel_Field_Level/field_acc_cnnlstm.py"

# 1D CNN hyperparameter search with Optuna (25 trials)
python "Deep Learning/Pixel_Field_Level/cnn_dl_hyper.py"

# Train 1D CNN with best Optuna params (run after cnn_dl_hyper.py)
python "Deep Learning/Pixel_Field_Level/best_ccn_params.py"

# TabNet ensemble (pixel-level evaluation)
python "Deep Learning/Pixel_Field_Level/TabTransformer.py"

# TabNet ensemble (field-level evaluation, saves models)
python "Deep Learning/Pixel_Field_Level/TabTransformer_Final_Field.py"
```

### Stage 3: Patch-Level Deep Learning

Requires patch data preparation first. Expected results: Kappa ~0.65-0.66, F1 ~0.74-0.75.

```bash
# Step 1: Create patches from field boundaries (requires geopandas)
python "Deep Learning/Patch Level/Create_Patches.py"

# Step 2: Extract pixel values per patch from GeoTIFFs (requires rasterio)
python "Deep Learning/Patch Level/Create Master Data.py"

# Step 3: Train models
python "Deep Learning/Patch Level/3D_CNN.py"
python "Deep Learning/Patch Level/Multi_Channel_CNN.py"
python "Deep Learning/Patch Level/Ensemble - 3D CNN.py"

# Step 4: Evaluate (run after corresponding training script)
python "Deep Learning/Patch Level/results_3d_cnn.py"
python "Deep Learning/Patch Level/results_multi_channel_cnn.py"
python "Deep Learning/Patch Level/results_ensemble_patching.py"

# Ensemble inference (run after Ensemble - 3D CNN.py)
python "Deep Learning/Patch Level/Inference_Ensemble.py"
```

## Output Locations

All model artifacts are saved under `src/`:

| Directory              | Contents                                             |
| ---------------------- | ---------------------------------------------------- |
| `models/`              | PyTorch `.pt` weights, Keras `.h5`, `.pkl` artifacts |
| `xgb_tuner/`           | XGBoost tuning results and saved models              |
| `saved_models_tabnet/` | TabNet model checkpoints (`.zip`)                    |
| `reports/`             | Auto-generated model reports (see below)             |

## Model Reports

Every training script auto-generates a standardized report on completion via the `ModelReport` class in `report.py`. Each run creates a timestamped folder under `reports/`:

```text
reports/
    XGBoost_Field-Level_20260204_153012/
        report.pdf              # Multi-page PDF (summary, per-class, confusion matrix, etc.)
        metadata.json           # Machine-readable metrics + hyperparameters
        metrics.csv             # Single row: accuracy, kappa, F1 weighted, F1 macro
        per_class_metrics.csv   # Per-class precision, recall, F1, support
        confusion_matrix.csv    # NxN labeled confusion matrix
        confusion_matrix.png    # Heatmap (300 dpi)
        feature_importance.png  # Top-20 features (tree models only)
        training_curves.png     # Loss/accuracy over epochs (DL models only)
        predictions.csv         # Field-level true vs predicted labels
```

### Which scripts generate reports

| Script                                   | Report name                          | Extras                          |
| ---------------------------------------- | ------------------------------------ | ------------------------------- |
| `xg_boost_hyper.py`                      | XGBoost Field-Level                  | feature importance, predictions |
| `Ensemble - Voting and Stacking.py`      | Voting Ensemble, Stacking Ensemble   | predictions (2 reports)         |
| `SMOTE_meta.py`                          | SMOTE Meta-Learner                   | predictions                     |
| `base_ml_models.py`                      | {Model} (Pixel-Level) x4             | feature importance (trees)      |
| `cnn_bilstm.py`                          | CNN-BiLSTM Ensemble (5-seed)         | --                              |
| `field_acc_cnnlstm.py`                   | CNN-BiLSTM Field-Level               | predictions                     |
| `TabTransformer_Final_Field.py`          | TabTransformer Ensemble (Field)      | predictions                     |
| `3D_CNN.py`                              | 3D CNN Patch-Level                   | training curves                 |

### Cross-model comparison

After running multiple models, compare them all:

```bash
python compare_models.py
```

This scans `reports/*/metadata.json` and generates:

- `reports/model_comparison.csv` — comparison table sorted by F1 weighted
- `reports/model_comparison.pdf` — table + grouped bar chart (accuracy, kappa, F1)

You can also point it at a custom directory: `python compare_models.py /path/to/reports/`

## Data Split Rules

- **Regional split:** Training uses regions 34S_19E_258N and 34S_19E_259N. Test region 34S_20E_259N is for inference only.
- **Field-level split:** Within training data, train/val/test splits are done on unique field IDs (`fid`) with `random_state=42`. All pixels from a field stay in the same partition to prevent data leakage.
