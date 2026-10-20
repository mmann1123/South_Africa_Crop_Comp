# Holdout September fill check

## Why

The holdout tile (`34S_20E_259N`) has no September observation for 40.6% of its pixels
(39.3% of fields); the training tiles have none missing. The published pipelines filled
these gaps differently:

| Pipeline | Fill of missing holdout values |
|---|---|
| Raw-input deep models (L-TAE, L-TAE-S, TempCNN, Transformer, CNN-BiLSTM, TimesNet, MiniRocket) | 0 reflectance before scaling, a value never seen in training |
| Pixel TabNet | the holdout's own column medians |
| xr_fresh classical models | features computed on the shortened series (NaN-aware xr_fresh functions) |

This experiment re-runs holdout inference with the missing September values filled from
**training-tile** medians (no holdout statistic), so every model faces the same, plausible
input. No model is retrained.

## Steps and scripts

| Script | What it does | Env |
|---|---|---|
| `make_filled_holdout.py` | copy of `data/merged_dl_test.parquet` with September filled from training medians (`data/`, git-ignored) | deep_field |
| `tabnet_fill.py` | pixel TabNet under median vs. zero fill; loader for the February checkpoints under pytorch_tabnet 4.6.0 | deep_field |
| `run_deep_field.py` | the published `out_of_sample/inference_*.py` and `field_reduction/predict_oos.py` code, input and output paths redirected only | deep_field |
| `run_ts_baselines.py` | TimesNet and MiniRocket with the saved models and `recent_baselines/common.load_holdout` | ts_baselines |
| `rebuild_xrfresh.py` | the 174 holdout xr_fresh features recomputed with xr_fresh 0.2.2 from the pixel series (original and filled) | xr_fresh |
| `combine_xrfresh.py` | field-mean feature files for the classical models, and per-band pixel files for Voting | deep_field |
| `run_classical.py` | the published classical inference (Voting and Stacking retrain inside their scripts, fixed seeds) | ml_field |
| `score.py` | per-model scores, reproduction check, paired bootstrap (B = 10,000, seed 42) of filled vs. original | deep_field |
| `score_sept_observed.py` | published predictions scored on the fields with September observed vs. missing (no imputation) | deep_field |
| `run_sweep.py`, `run_sweeps.sh` | the published data-efficiency sweep scripts (`field_reduction/seed_sweep*.py`), holdout input redirected | deep_field, ml_field |

Every published file is left unchanged; outputs go to `predictions/` and `results/`.

## Reproduction checks

- **Deep models:** the original-file re-run reproduces the published predictions field for
  field for every model except pixel TabNet. TabNet's published file (2026-02-10) predates its
  saved checkpoints (2026-02-17/18); the current checkpoints differ on 221 of 2,415 fields and
  score 0.609 vs. 0.603.
- **xr_fresh features:** the rebuild from the original series matches the published holdout
  feature files for every feature; at most 2.9 pixels per million differ, all float32 rounding
  ties at the series mean (`results/xrfresh_rebuild_check.csv`). The original extraction looks
  up `"autocorr"` in `function_mapping`, which returns None, so the three `autocorr_lag_k`
  features are copies of `absolute_sum_of_changes`; the rebuild reproduces this.
- **Classical field features:** the published `data/combined_test_features.parquet` was built
  from an older pixel grid and differs from the rebuild on 85 fields (mostly B12). The
  classical comparison is therefore rebuilt-original vs. rebuilt-filled, run in the same
  environment. Stacking, Voting and field LightGBM still reproduce their published
  predictions exactly; field XGBoost, the pixel LR/RF/XGBoost models and xr_fresh TabNet
  differ on 118 to 878 fields (library versions).
- **Sweep:** the classical sweep on the original features reproduces the published
  `field_reduction/results/seed_sweep_var.csv` exactly.

## Results

`results/fill_comparison.csv` (all 27 models; patch CNNs not re-run): with September filled
from training medians, the dense pixel networks rise most (CNN-BiLSTM 0.51 → 0.66, L-TAE
0.54 → 0.65, TempCNN 0.57 → 0.66, MiniRocket 0.50 → 0.61), pixel TabNet barely changes
(0.61 → 0.62), and the xr_fresh classical models gain 0.02 to 0.05. The top six are pixel
deep temporal models (0.63 to 0.66).

`results/sept_observed_subset.csv`: on the 1,352 holdout fields with September observed, the
published predictions already rank the dense pixel networks first (CNN-BiLSTM 0.66, TempCNN
0.65) and pixel TabNet 14th (0.62); on the 1,063 fields with September missing, the dense
networks fall to 0.21 to 0.36 while TabNet (0.58) and L-TAE-S (0.54) hold up.

`results/sweep_*_trainmed.csv`: data-efficiency curves on the filled holdout (see also
`../sparse_gate_backbones/`). The published near-flat L-TAE curve was an artifact of the
zero fill; every model gains with more training fields.
