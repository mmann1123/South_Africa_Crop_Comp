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

1. Fill-value sweep: done, see `fill_sweep/README.md`. Step deletion for L-TAE and the
   Transformer still to do.
2. Month-dropout control for the sparsemax gate (done 2026-10-23; see `month_dropout/README.md`).
3. Per-model fill-operator table for the supplement.
4. Per-month missing-data table (training tiles vs holdout) and the September gap overlaid on the
   holdout error map.
5. Gradient attribution at the September slot.

## Tree + deep ensembles (2026-10-23)

`ensemble.py`: hard majority vote over saved holdout predictions of five deep raw-sequence pixel
models (CNN-BiLSTM, TempCNN, L-TAE, Transformer, TimesNet) and five xr_fresh tree models (XGBoost
and LightGBM field; RF, LightGBM, XGBoost pixel). Members are whole families or every 3-member
combination, and vote ties go to the higher published in-region F1m, so no holdout information
enters the ensembles. "Zero" means the published inputs: zero-filled raw sequences for the deep
models, NaN-aware xr_fresh features for the trees. Unit tests: `tests/test_ensemble_vote.py`.

| Fill | deep-5 | tree-5 | mixed-10 | best single member |
|---|---|---|---|---|
| training median | 0.660 | 0.611 | 0.661 | 0.656 (CNN-BiLSTM) |
| zero (published) | 0.572 | 0.570 | 0.603 | 0.582 (XGBoost field) |

With a sensible fill, adding trees to the deep vote changes nothing (mixed minus deep +0.001,
CI [−0.012, 0.014]), and among 3-member votes the deep-only triples are best (mean 0.659) while
1 deep + 2 tree triples never beat their best member. Under zero fill the families fail on
different fields (deep x tree disagreement 0.32 vs 0.25 within deep), and mixing helps: mixed-10
beats deep-5 by +0.032 [0.015, 0.049] and tree-5 by +0.033 [0.013, 0.053]. Mixing families is a
hedge against missingness shift, not a gain in transfer, and it is a weaker hedge than month
dropout (single dense models 0.653 to 0.657 under zero fill; `month_dropout/README.md`).

## The September gap is the "south-central error zone" (2026-10-23)

`sept_gap_map.py` overlays each holdout field's share of September-missing pixels on correct and
incorrect fields (`figures/sept_gap_error_map.pdf`). The gap is a sharp-edged north-south strip
(likely a scene or cloud-mask boundary) covering 76% of the fields in the 8 to 16 km zone that
the CEA supplement described as a "south-central error zone", against 26% of fields elsewhere.

Percent of fields correct, by zone and September status (`results/sept_gap_zone.csv`):

| Model, inputs | outside zone, Sept observed | outside, Sept missing | inside, observed | inside, missing |
|---|---|---|---|---|
| L-TAE, published (zero) | 71 | 50 | 71 | 45 |
| L-TAE, training median | 71 | 82 | 71 | 76 |
| CNN-BiLSTM, published (zero) | 72 | 51 | 71 | 36 |
| CNN-BiLSTM, training median | 72 | 80 | 71 | 75 |
| TabNet (pixel), published | 70 | 84 | 68 | 72 |

Inside the zone, fields with September observed are classified as well as anywhere else; the
zone's errors are the September-missing fields. In a class-adjusted logit of field correctness
(`results/sept_gap_logit.csv`), the distance effect under the published inputs (L-TAE −0.32,
CNN-BiLSTM −0.31, TempCNN −0.31 log-odds per 10 km, all p < 0.001) vanishes once September-missing
share enters the model (−0.01, +0.10, −0.11, all n.s.), and with a training-median fill there is
no distance effect to explain. The supplement's reading ("local conditions within the tile",
dense networks most affected) was this artifact. The raw September-missing fields look easier
under a sensible fill because they differ in crop mix; the logit adjusts for class.

Two residual effects under the training-median fill: a class-adjusted September penalty for
field XGBoost (−0.54 [−0.76, −0.31]) and pixel TabNet (−0.36 [−0.60, −0.13]), but not for the
dense networks (CIs span zero). A median fill removes information, so some penalty is expected;
why it falls on the tabular models only is unexplained.

## Gradient attribution at the September slot (2026-10-23)

`attribution.py`: integrated gradients (baseline = training mean; step count raised per model
until the completeness error is below 2% on a calibration subset and below 3% on every input set;
128 to 2,048 steps) and plain gradients, per month, for the predicted class, on 5,000
September-missing and 5,000 September-observed holdout pixels. Models: the 2 x 2 arms of
TempCNN and CNN-BiLSTM, 5 seeds each (`figures/attribution_september.pdf`,
`results/attribution_{september,by_month}.csv`).

| Backbone, arm | IG share on Sept: zero-filled | training-median fill | Sept observed | pixel predictions changed by the fill |
|---|---|---|---|---|
| TempCNN, dense | 0.32 | 0.04 | 0.13 | 37% |
| TempCNN, dense + dropout | 0.18 | 0.05 | 0.15 | 10% |
| TempCNN, gated | 0.35 | 0.06 | 0.15 | 40% |
| TempCNN, gated + dropout | 0.27 | 0.05 | 0.13 | 9% |
| CNN-BiLSTM, dense | 0.37 | 0.06 | 0.17 | 64% |
| CNN-BiLSTM, dense + dropout | 0.39 | 0.07 | 0.18 | 15% |
| CNN-BiLSTM, gated | 0.24 | 0.06 | 0.16 | 26% |
| CNN-BiLSTM, gated + dropout | 0.26 | 0.06 | 0.17 | 9% |

A zero-filled September draws 1.5 to 2.6 times the attribution of a real September in every arm,
and the training-median fill moves the weight to October, the next observed month. The clearest
mechanism measure is the flip rate: the zero fill changes 37% (TempCNN) and 64% (CNN-BiLSTM) of
dense pixel predictions, the gate brings CNN-BiLSTM to 26% but not TempCNN (40%), and month
dropout brings every arm to 9 to 15%. That ordering matches the holdout F1 results of the 2 x 2.

Caveat: attribution share is not a clean mechanism measure here. Integrated gradients scale with
the distance from the baseline, which is large for a zero-filled month, so the September share
stays high under dropout for CNN-BiLSTM (0.39) although its predictions barely move; plain
gradient shares show no consistent pattern across arms. Dropout makes the decision insensitive
to the zero without necessarily lowering the attribution assigned to it. Attention maps for
L-TAE and the Transformer (also in the to-do item) were not computed.
