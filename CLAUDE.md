# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satellite-based crop classification for the Western Cape of South Africa using time-series Sentinel-2 imagery, posed as the AI4FoodSecurity (South Africa) Track 1 challenge, with field labels from Radiant Earth (the repo began from the [Spot The Crop Challenge](https://github.com/radiantearth/spot-the-crop-challenge)). Models train on two regions (`34S_19E_258N`, `34S_19E_259N`) and are evaluated out-of-sample on a holdout region (`34S_20E_259N`).

**Crop classes**: Lucerne/Medics, Small grain grazing, Barley, Canola, Wheat (5 total).
**Primary metric**: F1 macro (Cohen's Kappa and weighted F1 also reported). The challenge's official metric is field-level cross-entropy (the "Xent" column).

## Running the Pipeline

Numbered orchestrator scripts in the repo root run the full workflow:

```bash
python 00_run_all_steps.py                 # steps 0-5 sequentially (--start/--stop/--dry-run)
python 0_create_training_data.py           # builds data/*.parquet from external inputs
python 1_train_all_models.py               # classical + DL (--classical-only, --dl-only, --force)
python 2_compare_training_results.py       # training metrics comparison
python 3_create_test_data.py               # out-of-sample feature datasets
python 4_run_inference.py                  # OOS predictions (--models <name>, --list, --force)
python 5_compare_and_submit.py             # ensemble vote -> submissions/prediction.csv
python 6_field_reduction_experiment.py     # training-set-size ablation (slow; --skip-tabnet)
```

Most orchestrators support `--dry-run`. Training scripts skip work if saved model artifacts exist; use `--force` to retrain.

## Two Conda Environments

Orchestrators dispatch subprocesses to the correct env via `DEEP_FIELD_PYTHON` / `ML_FIELD_PYTHON` from `deep_learn/src/config.py` (falls back to `sys.executable` if envs are missing):

- **`deep_field`** — PyTorch, TensorFlow, TabNet. Setup: `conda activate deep_field && bash deep_learn/install.sh`
- **`ml_field`** — XGBoost (GPU), LightGBM (CUDA, built from source), scikit-learn. Setup: `bash deep_learn/install_ml.sh`

GPU notes: XGBoost uses `device="cuda", tree_method="hist"` with the standard pip package; LightGBM CUDA requires the source build done by `install_ml.sh`. TensorFlow needs `LD_LIBRARY_PATH` pointing at pip-installed nvidia libs — handled by conda activation hooks set up in `install.sh`.

## Configuration & Data Layout

`deep_learn/src/config.py` is the single source of truth for all paths, bands, and regions. Scripts outside `deep_learn/src` (root orchestrators, `out_of_sample/`, `experiments/`) import it via `sys.path.insert(0, DEEP_LEARN_SRC)`.

- **External inputs** (not in repo) live under `/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/`: raw per-band parquets (`DATA_DIR`), xr_fresh features (`FEATURES_DIR` = training, `TEST_FEATURES_DIR` = its parent, containing `testing_*.parquet`), and label GeoJSONs.
- **Generated data** is centralized in `data/`: `final_data.parquet` (field-level xr_fresh features), `merged_dl_train/test.parquet` (pixel-level time series), `patch_level_data.parquet`, `combined_test_features.parquet`, etc.
- **Model artifacts** stay in `deep_learn/src/`: `models/`, `xgb_tuner/`, `lgbm_tuner/`, `saved_models_tabnet*/`.
- **Bands**: B2, B6, B11, B12, EVI, hue. Months 05 and 06 are excluded (missing data / cloud cover).

## Repository Layout

- **`light_learn/`** — legacy standalone classical-ML pipeline (numbered scripts, Docker-based: `cd light_learn && docker-compose up`, then `source activate spfeas`). Mostly superseded by the root pipeline. Its step 0 uses Google Earth Engine and needs `ee.Authenticate()`.
- **`writeup/`** — LaTeX manuscripts; conventions live in `writeup/CLAUDE.md`.

## Key Architecture Patterns

- **Field-level aggregation**: final predictions are per field (FID); pixel/patch models aggregate by FID via majority vote or mean pooling
- **FID-wise splitting**: train/val/test splits on unique field IDs with `random_state=42` to prevent leakage — all scripts follow this
- **Ensembling**: multiple models combined via voting/stacking; final submission is an ensemble vote over `out_of_sample/predictions_*.csv`
- Label encoder variable naming is inconsistent across scripts: `le` (xgboost, SMOTE, Ensemble, 3D_CNN) vs `label_encoder` (cnn_bilstm, TabTransformer, base_ml)

## Submission & Scoring

`submissions/prediction.csv` with one column (`crop_name`) or two (`crop_name,probability`), ordered by field ID (see `light_learn/scoring/field_id.csv` and `field_fid.csv`).

PRs touching `submissions/` trigger the `score.yml` GitHub Action, which scores against hidden ground truth (a GitHub secret) and comments results on the PR. Local scoring (requires `GROUND_TRUTH` env var):

```bash
python light_learn/scoring/score.py
```

OOS scoring locally uses the test labels GeoJSON directly (see `experiments/field_reduction/experiment_config.py`); confusion matrices land in `out_of_sample/scoring_results/`.
