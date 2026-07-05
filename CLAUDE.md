# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satellite-based crop classification for the Western Cape of South Africa using time-series Sentinel-2 imagery, originally from the [Radiant Earth Spot The Crop Challenge](https://github.com/radiantearth/spot-the-crop-challenge). Models train on two regions (`34S_19E_258N`, `34S_19E_259N`) and are evaluated out-of-sample on a holdout region (`34S_20E_259N`).

**Crop classes**: Lucerne/Medics, Small grain grazing, Barley, Canola, Wheat, and others (9 total).
**Primary metric**: F1 macro (Cohen's Kappa and weighted F1 also reported).

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

- **Root `0_`–`6_` scripts** — orchestrators (see above)
- **`deep_learn/src/`** — all model code:
  - `Classical Machine Learning/` — field-level (XGBoost+Optuna, SMOTE meta-learner, voting/stacking ensembles) and pixel-level baselines (LightGBM, XGBoost, RF, LR)
  - `Deep Learning/` — pixel/field-level (TabNet, CNN-BiLSTM, L-TAE, TempCNN) and patch-level (3D CNN)
  - `run_all_classical_models.py`, `run_all_dl_models.py` — stage runners invoked by step 1
  - `report.py` — `ModelReport` class; generates timestamped folders under `deep_learn/src/reports/` with PDF, 300dpi PNGs, CSVs, and `metadata.json`
  - `compare_models.py` — scans `reports/` metadata into a comparison table + chart
  - `model_registry.py` / `model_registry.json` — tracks trained models, metrics, artifact paths
- **`out_of_sample/`** — per-model inference scripts (`inference_*.py`) producing `predictions_*.csv`; `compare_predictions.py` computes pairwise agreement and the ensemble vote
- **`experiments/field_reduction/`** — ablation measuring OOS F1 vs. training fraction (0.25/0.50/0.75); `run_experiment.py` orchestrates, `experiment_config.py` holds fractions/model-env maps, results in `results/`
- **`light_learn/`** — legacy standalone classical-ML pipeline (numbered scripts, Docker-based: `cd light_learn && docker-compose up`, then `source activate spfeas`). Mostly superseded by the root pipeline.
- **`writeup/`** — LaTeX manuscripts and figures. Two variants: the active IEEE TGRS submission (`tgrs-article.tex` + online supplement `tgrs-supplement.tex`, S-numbered floats) and the original Springer Nature version (`sn-article.tex`).
  - **Table notes go *below* the table, never in the `\caption{}`.** The caption holds a short title only; put all explanatory notes (column/symbol definitions, †/‡ footnotes, "n=…", method caveats, "bold = best") in a `\footnotesize` block beneath the tabular: `\end{tabular}\par\vspace{3pt}\begin{minipage}{\columnwidth}\footnotesize <note>\end{minipage}\end{table}` (use `\textwidth` for `table*`). Applies to both manuscripts.
  - **Group closely related figures into one multi-panel float** (panels labelled (a), (b), … with one shared caption) instead of many small separate floats, to keep the supplement readable.
  - Main↔supplement cross-references are hardcoded (`\ref` does not cross files), so re-number S-figures/S-tables by hand and re-grep the main text after any supplement float change.

## Key Architecture Patterns

- **Field-level aggregation**: final predictions are per field (FID); pixel/patch models aggregate by FID via majority vote or mean pooling
- **FID-wise splitting**: train/val/test splits on unique field IDs with `random_state=42` to prevent leakage — all scripts follow this
- **Ensembling**: multiple models combined via voting/stacking; final submission is an ensemble vote over `out_of_sample/predictions_*.csv`
- **Preprocessing**: sklearn Pipeline of SimpleImputer → StandardScaler → VarianceThreshold
- Label encoder variable naming is inconsistent across scripts: `le` (xgboost, SMOTE, Ensemble, 3D_CNN) vs `label_encoder` (cnn_bilstm, TabTransformer, base_ml)

## Submission & Scoring

`submissions/prediction.csv` with one column (`crop_name`) or two (`crop_name,probability`), ordered by field ID (see `light_learn/scoring/field_id.csv` and `field_fid.csv`).

PRs touching `submissions/` trigger the `score.yml` GitHub Action, which scores against hidden ground truth (a GitHub secret) and comments results on the PR. Local scoring (requires `GROUND_TRUTH` env var):

```bash
python light_learn/scoring/score.py
```

OOS scoring locally uses the test labels GeoJSON directly (see `experiments/field_reduction/experiment_config.py`); confusion matrices land in `out_of_sample/scoring_results/`.

## Key Dependencies

- **Geospatial**: Google Earth Engine (light_learn step 0 only; needs `ee.Authenticate()`), rasterio, geopandas, geowombat
- **Feature engineering**: xr_fresh (time-series statistics)
- **ML/DL**: scikit-learn, LightGBM, XGBoost, PyTorch, TensorFlow, pytorch-tabnet, Optuna
- **Parallelization**: Ray, Dask
