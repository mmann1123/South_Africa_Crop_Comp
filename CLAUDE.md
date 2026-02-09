# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satellite-based crop classification competition for the Western Cape of South Africa, using time-series Sentinel-2 multispectral imagery. Originally from the [Radiant Earth Spot The Crop Challenge](https://github.com/radiantearth/spot-the-crop-challenge). Contains winning models from the competition's regular track.

**Crop classes**: Lucerne/Medics, Small grain grazing, Barley, Canola, Wheat, and others (9 total).

## Repository Layout

Two independent pipelines exist side by side:

- **`light_learn/`** - Lightweight end-to-end pipeline using classical ML (numbered scripts `0_` through `7_`)
- **`deep_learn/`** - Advanced models: XGBoost ensembles, TabTransformer, CNN-LSTM, 3D CNN, Vision Transformers

Both pipelines produce predictions that go into `submissions/prediction.csv` for automated evaluation.

## Pipeline (light_learn)

The numbered scripts run sequentially:

1. `0_SA_ee_download_data_prep.py` - Downloads Sentinel-2 data via Google Earth Engine API (requires EE auth + GDAL)
2. `2_SA_xr_fresh_extraction.py` - Time-series feature extraction using `xr_fresh` and `geowombat` (30+ statistical features per band)
3. `3_SA_extract_values_to_polys.py` - Extracts raster values to field polygons using Ray parallel processing
4. `4_SA_model.py` - Trains ensemble models (LightGBM, XGBoost, RF, SVM) with Optuna hyperparameter tuning
5. `5_SA_image_validation.py` - Visual QA of training field imagery
6. `6_SA_report_charts.py` - Summary statistics and distribution charts
7. `7_test_setup.py` - Prepares test submission files, aggregates predictions to field level

## Deep Learning Models (deep_learn)

- **Classical ML / Field Level**: XGBoost with Optuna tuning, SMOTE for class imbalance, ensemble voting/stacking
- **Pixel/Field Level**: TabTransformer (PyTorch TabNet, 5-model ensemble), CNN-BiLSTM
- **Patch Level**: 3D CNN on spatial-temporal patches, Vision Transformers

All deep learning models aggregate pixel/patch predictions to field level (group by FID, majority vote or mean pooling).

## Key Architecture Patterns

- **Field-level aggregation**: All predictions must be at the field (FID) level, not pixel level
- **Fid-wise data splitting**: Train/val/test splits are done on unique field IDs to prevent data leakage
- **Ensemble approach**: Multiple models combined via voting or averaging for final predictions
- **Preprocessing pipeline**: sklearn Pipeline with SimpleImputer -> StandardScaler -> VarianceThreshold

## Submission & Scoring

Place predictions in `submissions/prediction.csv` with either:
- One column: `crop_name`
- Two columns: `crop_name,probability`

Predictions must be ordered by field ID (see `scoring/field_id.csv` and `scoring/field_fid.csv`).

**Automated evaluation**: PRs that modify `submissions/` trigger a GitHub Action (`score.yml`) that computes Cohen's Kappa, weighted F1, and Cross Entropy against hidden ground truth stored in a GitHub secret.

Run scoring locally (requires `GROUND_TRUTH` env var):
```bash
python scoring/score.py
```

## Environment Setup

### deep_learn
```bash
pip install -r deep_learn/requirements.txt
```
Key deps: torch, tensorflow, xgboost, lightgbm, optuna, rasterio, geopandas, scikit-learn

### light_learn (Docker)
```bash
cd light_learn && docker-compose up
# Inside container:
source activate spfeas
```
The Docker image includes GDAL, geowombat, xr_fresh, spfeas, and all ML libraries.

### Google Earth Engine
Step 0 requires Earth Engine authentication:
```python
import ee
ee.Authenticate()
ee.Initialize()
```

## Key Dependencies

- **Geospatial**: Google Earth Engine, rasterio, geopandas, geowombat, shapely
- **Feature engineering**: xr_fresh (time-series statistics)
- **Classical ML**: scikit-learn, LightGBM, XGBoost
- **Deep learning**: PyTorch, TensorFlow, TabNet
- **Optimization**: Optuna
- **Parallelization**: Ray, Dask
