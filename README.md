# South Africa Crop Classification

**Satellite-based crop type classification for the Western Cape of South Africa using multi-temporal Sentinel-2 imagery.** This repository benchmarks classical machine learning, deep learning, and hybrid approaches against one another on a single region, with a deliberate emphasis on *out-of-sample* spatial transfer rather than the in-region cross-validation that most studies report.

The data originate from the [Radiant Earth Spot The Crop Challenge](https://github.com/radiantearth/spot-the-crop-challenge). Models train on two tiles (`34S_19E_258N`, `34S_19E_259N`) and are evaluated on a spatially disjoint holdout tile (`34S_20E_259N`) that no training pixel touches.

**Crop classes (9):** Lucerne/Medics, Small grain grazing, Barley, Canola, Wheat, and others.
**Primary metric:** macro F1 (Cohen's Kappa and weighted F1 also reported).

## Why this repository

Most crop-classification papers validate within a single region and report that dense temporal deep nets (CNN-BiLSTM, TempCNN, L-TAE) win. We evaluate every model *twice* — once with conventional field-wise cross-validation inside the training region, and once on a holdout tile — and find that this dual evaluation reorders the models. The dense temporal networks that look strongest in-region suffer the largest generalization gaps under spatial transfer, while parsimonious, sparse-feature models (Random Forest, gradient-boosted trees, TabNet) transfer with much smaller losses. Inputs are restricted to optical Sentinel-2 time series alone — no SAR — to isolate the signal available from spectral time series.

The full analysis is written up in [`writeup/sn-article.tex`](writeup/sn-article.tex).

## Models

- **Classical (field-level):** XGBoost + Optuna, LightGBM, Random Forest, logistic regression, SMOTE meta-learner, voting/stacking ensembles
- **Deep learning (pixel/field-level):** TabNet, CNN-BiLSTM, L-TAE, TempCNN
- **Patch-level:** 3D CNN
- **Features:** `xr_fresh` automated time-series statistics over bands B2, B6, B11, B12, EVI, and hue (months 05 and 06 excluded for cloud cover / missing data)

## Installation

Two conda environments separate the deep-learning and classical-ML stacks. Orchestrators dispatch each step to the correct interpreter automatically.

```bash
# Deep learning: PyTorch, TensorFlow, TabNet
conda activate deep_field
bash deep_learn/install.sh

# Classical ML: XGBoost (GPU), LightGBM (CUDA), scikit-learn
bash deep_learn/install_ml.sh
```

GPU notes: XGBoost runs on the standard pip package (`device="cuda", tree_method="hist"`); LightGBM CUDA is built from source by `install_ml.sh`. TensorFlow's `LD_LIBRARY_PATH` is wired up by conda activation hooks set during install.

## Usage

The numbered scripts in the repo root run the full workflow end to end. Most support `--dry-run`, and training scripts skip work when artifacts already exist (`--force` to retrain).

```bash
python 00_run_all_steps.py            # steps 0-5 sequentially (--start/--stop/--dry-run)

python 0_create_training_data.py      # build data/*.parquet from external inputs
python 1_train_all_models.py          # classical + DL (--classical-only, --dl-only, --force)
python 2_compare_training_results.py  # in-region training metrics
python 3_create_test_data.py          # out-of-sample feature datasets
python 4_run_inference.py             # holdout-tile predictions (--models <name>, --list)
python 5_compare_and_submit.py        # ensemble vote -> submissions/prediction.csv

python 6_field_reduction_experiment.py  # training-set-size ablation (slow; --skip-tabnet)
```

## Data layout

`deep_learn/src/config.py` is the single source of truth for all paths, bands, and regions.

- **External inputs** (not in repo) live under `/mnt/bigdrive/.../Agriculture_Comp/`: raw per-band parquets, `xr_fresh` features, and label GeoJSONs.
- **Generated data** is centralized in `data/`: `final_data.parquet` (field-level features), `merged_dl_train/test.parquet` (pixel-level time series), `patch_level_data.parquet`, `combined_test_features.parquet`.
- **Model artifacts** stay in `deep_learn/src/`: `models/`, `xgb_tuner/`, `lgbm_tuner/`, `saved_models_tabnet*/`.

All train/val/test splits are made on unique field IDs (`random_state=42`) to prevent pixel-level leakage. Final predictions are per field (FID); pixel and patch models aggregate to field level by majority vote or mean pooling.

## Repository structure

```
0_ – 6_ *.py            orchestrator scripts (the pipeline above)
deep_learn/src/         all model code
  Classical Machine Learning/   XGBoost, LightGBM, RF, SMOTE, ensembles
  Deep Learning/                TabNet, CNN-BiLSTM, L-TAE, TempCNN, 3D CNN
  report.py                     ModelReport -> timestamped reports/ (PDF, PNG, CSV, JSON)
  compare_models.py             scans reports/ into a comparison table
  model_registry.py             tracks trained models and metrics
out_of_sample/          per-model inference + ensemble vote
experiments/            training-set-size ablation
light_learn/            legacy Docker-based classical pipeline (superseded)
writeup/                LaTeX paper and figures
```

## Scoring

`submissions/prediction.csv` holds one prediction per field, ordered by field ID. Pull requests touching `submissions/` trigger the `score.yml` GitHub Action, which scores against hidden ground truth and comments on the PR. To score locally (requires the `GROUND_TRUTH` env var):

```bash
python light_learn/scoring/score.py
```

Out-of-sample scoring uses the test-label GeoJSON directly; confusion matrices land in `out_of_sample/scoring_results/`.

## Acknowledgements

Built on the [Radiant Earth Spot The Crop Challenge](https://github.com/radiantearth/spot-the-crop-challenge) dataset and Radiant Earth's MLHub labels. Feature extraction uses [`xr_fresh`](https://github.com/mmann1123/xr_fresh). The initial analysis of this dataset was carried out as a capstone project ([Capstone Group 3](https://github.com/DishaKacha7/Capstone_Group_3)).
