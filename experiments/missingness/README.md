# Missingness shift experiments

Experiments for the manuscript in `writeup/missingness_paper/`. The holdout tile is missing
September for 40.6% of pixels (39.3% of fields) and the training tiles miss no month the models
use; how each model filled that gap at inference decided the published ranking.

## Already done (separate folders, not moved because their scripts use relative paths)

- `../holdout_fill_check/`: all 27 models re-scored with September filled from training medians;
  xr_fresh holdout features rebuilt; published predictions scored on September-observed vs
  September-missing fields; data-efficiency sweep on the filled holdout. See its README.
- `../sparse_gate_backbones/`: sparsemax gate (with softmax control) on TempCNN and CNN-BiLSTM at
  full data and across 10 to 100% of training fields. See its README.

## Fill switch in the main pipeline

`deep_learn/src/missing.py` provides `fill_missing`, used by every raw band x month holdout
loader (`out_of_sample/inference_*.py`, `experiments/field_reduction/predict_oos.py`,
`experiments/recent_baselines/common.py`). Choose the method per run with the `HOLDOUT_FILL`
environment variable:

| `HOLDOUT_FILL` | Fill for missing holdout values |
|---|---|
| `published` (default) | each script's original rule: 0 for raw-sequence models, holdout medians then 0 for pixel TabNet |
| `zero` | 0 |
| `self_median` | the holdout's own column medians, then 0 (uses target-data statistics) |
| `train_median` | training-tile column medians (`data/train_medians.csv`) |
| `train_mean` | training-tile column means (`data/train_means.csv`) |
| `interp` | per pixel, linear interpolation between neighbouring months of the same band; edges fall back to `train_median` |

Example: `HOLDOUT_FILL=train_median python out_of_sample/inference_ltae.py`.
Columns never observed in training (May) are filled with 0 under every method, the value the
models saw in training.

Tests: `python -m unittest discover -s tests -v` (unit tests plus a regression test on the real
holdout showing `published` is bit-identical to the old code and `train_median` reproduces the
re-analysis input); `python tests/e2e_fill_pipeline.py` (GPU; L-TAE, pixel TabNet and field
L-TAE-S inference under both settings must equal the saved re-analysis predictions).

## Planned (see `reports/Missing data robustness novelty check.md`, "Ways to push this forward")

1. Fill-value sweep: re-score existing checkpoints under each `HOLDOUT_FILL` method, plus step
   deletion for L-TAE and the Transformer.
2. Month-dropout control for the sparsemax gate (retraining; 2 x 2 dropout by gate).
3. Per-model fill-operator table for the supplement.
4. Per-month missing-data table (training tiles vs holdout) and the September gap overlaid on the
   holdout error map.
5. Gradient attribution at the September slot.
