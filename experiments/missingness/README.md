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

## Learning curves down to 5% (2026-10-23)

`learning_curves/run.sh` added 5% for every swept model and 5/10/15% for pixel TabNet (10 draws
each), appending to the existing sweep files; `learning_curves/summarize.py` combines them
(`results/learning_curves.csv`, `figures/learning_curves.pdf`). Training-median-filled holdout;
one training seed per draw. At 100% the draws are the 5 training seeds of each full-data model
scored singly (`learning_curves/single_seed_100.py`; XGBoost and logistic regression are
deterministic single runs); the 5-seed ensembles, 0.01 to 0.04 higher, are separate markers in the
figure and the last column below. 5% keeps 132 of 2,653 training fields, at least
15 per class.

| Model | 5% | 10% | 15% | 25% | 50% | 75% | 100% | 100%, 5-seed ensemble |
|---|---|---|---|---|---|---|---|---|
| TempCNN, gated | 0.512 | 0.564 | 0.590 | 0.608 | 0.634 | 0.639 | 0.645 | 0.654 |
| TempCNN | 0.501 | 0.547 | 0.581 | 0.607 | 0.621 | 0.633 | 0.637 | 0.653 |
| L-TAE-S | 0.508 | 0.557 | 0.565 | 0.612 | 0.636 | 0.644 | 0.643 | 0.643 |
| L-TAE | 0.503 | 0.554 | 0.568 | 0.592 | 0.636 | 0.638 | 0.643 | 0.652 |
| CNN-BiLSTM | 0.522 | 0.554 | 0.561 | 0.580 | 0.592 | 0.616 | 0.615 | 0.642 |
| CNN-BiLSTM, gated | 0.516 | 0.543 | 0.557 | 0.575 | 0.592 | 0.596 | 0.619 | 0.638 |
| Logistic regression (pixel) | 0.528 | 0.547 | 0.563 | 0.574 | 0.599 | 0.606 | 0.605 | 0.605 |
| XGBoost (field) | 0.463 | 0.513 | 0.524 | 0.549 | 0.557 | 0.592 | 0.612 | 0.612 |
| TabNet (pixel) | 0.431 | 0.482 | 0.501 | 0.539 | 0.561 | 0.582 | 0.579 | 0.616 |

At 5 to 15% the raw-sequence models and logistic regression are statistically indistinguishable
(every Welch test against logistic regression p > 0.05, except gated TempCNN at 15%, +0.027,
p = 0.02). The two feature-selecting tabular models are the least data-efficient: XGBoost trails
logistic regression by 0.035 to 0.065 and TabNet by 0.062 to 0.097 (all p ≤ 0.002) across 5 to
15%. The deep sequence models separate from the linear baseline only from about 25%. There is no
low-data advantage for sparsity: the gated variants are within noise of their dense versions at
every fraction. Draws are not paired across models (different splits), so these are unpaired
comparisons.

## Paired low-data comparison, 5 to 15% (2026-11-18)

The 5 to 15% comparison above was unpaired (each sweep script rebuilt its own split, so draws did
not share fields). `learning_curves/paired/` reruns it paired: `make_splits.py` wrote `splits.json`
(one 64/16/20 field split, 2,653/664/830 fields, and 10 stratified draws per fraction of
132/265/397 training fields), every sweep script reads it through `PAIRED_FID_FILE`
(`field_reduction/subsample.py`; `tests/test_paired_splits.py` checks that an unset variable
reproduces the published splits). Nine models, 3 fractions, 10 draws, one training seed per
draw, training-median-filled holdout; `run.sh` then `resume.sh` (the first launch stopped on
2026-11-09), about 23 GPU hours; `summarize.py` writes `results/paired_summary.csv` and
`results/paired_contrasts.csv` (paired t-interval, Wilcoxon signed-rank, draws favouring).

| Model | 5% | 10% | 15% |
|---|---|---|---|
| TempCNN, gated | 0.512 | 0.556 | 0.585 |
| TempCNN | 0.503 | 0.556 | 0.584 |
| L-TAE-S | 0.508 | 0.557 | 0.565 |
| L-TAE | 0.503 | 0.554 | 0.568 |
| CNN-BiLSTM, gated | 0.482 | 0.522 | 0.559 |
| CNN-BiLSTM | 0.491 | 0.530 | 0.556 |
| Logistic regression (pixel) | 0.489 | 0.531 | 0.554 |
| XGBoost (field) | 0.467 | 0.510 | 0.528 |
| TabNet (pixel) | 0.431 | 0.482 | 0.501 |

Paired contrasts against pixel logistic regression (mean difference over the 10 draws, 95% paired
t-interval, draws favouring the model):

| Model | 5% | 10% | 15% |
|---|---|---|---|
| TempCNN | +0.014 [0.000, 0.028], 8/10 | +0.025 [0.004, 0.046], 9/10 | +0.029 [0.011, 0.047], 9/10 |
| TempCNN, gated | +0.023 [0.006, 0.040], 8/10 | +0.026 [−0.000, 0.051], 7/10 | +0.030 [0.011, 0.050], 8/10 |
| L-TAE | +0.015 [−0.005, 0.035], 6/10 | +0.023 [−0.006, 0.052], 7/10 | +0.013 [−0.006, 0.033], 6/10 |
| L-TAE-S | +0.019 [0.005, 0.033], 9/10 | +0.026 [0.005, 0.047], 7/10 | +0.011 [−0.009, 0.031], 7/10 |
| CNN-BiLSTM | +0.002 [−0.011, 0.016], 5/10 | −0.001 [−0.024, 0.022], 6/10 | +0.002 [−0.016, 0.018], 6/10 |
| CNN-BiLSTM, gated | −0.007 [−0.030, 0.016], 5/10 | −0.009 [−0.046, 0.027], 4/10 | +0.005 [−0.022, 0.031], 6/10 |
| XGBoost (field) | −0.022 [−0.047, 0.003], 3/10 | −0.020 [−0.051, 0.010], 3/10 | −0.027 [−0.047, −0.007], 2/10 |
| TabNet (pixel) | −0.057 [−0.079, −0.036], 1/10 | −0.048 [−0.071, −0.026], 1/10 | −0.054 [−0.068, −0.039], 0/10 |

Pairing sharpens the picture and softens the earlier null. TempCNN is ahead of logistic regression
at every fraction (paired intervals exclude zero, Wilcoxon p = 0.02 to 0.03, 8 or 9 draws of 10),
but by 0.014 to 0.029, against 0.05 at 100%. L-TAE and L-TAE-S lead by 0.011 to 0.026 with
intervals that cross zero at most fractions. CNN-BiLSTM, dense or gated, equals logistic
regression. The two feature-selecting tabular models still trail (XGBoost by 0.02 to 0.03,
TabNet by 0.05 to 0.06; TabNet significant at every fraction, XGBoost at 15% and by Wilcoxon at 5%).
The gate adds nothing at low data on any backbone (gated minus dense −0.009 to +0.009, every
interval crossing zero). The defensible wording is: at 5 to 15% of training fields the best deep
temporal model leads a pixel logistic regression by 0.01 to 0.03 macro-F1, a quarter to a half
of its full-data lead, the recurrent model does not lead at all, and sparsity gives no low-data
advantage.

## Re-scoring the filled predictions: paired tests, per-class, cereal merge, seed SD (2026-11-19)

`filled_rescoring.py` re-scores the training-median-filled holdout predictions saved by
`holdout_fill_check` (no retraining) for the four items the paper plan still needed. Outputs in
`results/`: `paired_bootstrap_filled.csv`, `per_class_filled.csv`, `cereal_merge_filled.csv`,
`per_seed_sd_filled.csv`; figure `figures/per_class_f1_filled.pdf`.

**Paired field bootstrap (B = 10,000, seed 42), macro-F1, filled predictions.** The three dense
pixel models lead every reference model by a significant margin: over pixel TabNet by 0.036 to
0.040 (CNN-BiLSTM +0.040 [0.021, 0.060], TempCNN +0.039, L-TAE +0.036), over field XGBoost by
0.040 to 0.044, over pixel logistic regression by 0.047 to 0.051; the Transformer leads logistic
regression by 0.035 [0.010, 0.060]. TabNet (+0.011), XGBoost (+0.007) and MiniRocket (0.000) are
not separable from logistic regression. The two published paired claims reverse or vanish:
TabNet over CNN-BiLSTM is now −0.040 [−0.060, −0.020]; L-TAE-S over L-TAE is −0.010
[−0.025, 0.006] at pixel level and −0.001 at field level; Transformer vs L-TAE −0.012, n.s.
Aggregation to field level lowers macro-F1 for all seven raw-sequence architectures, significantly
for six (TempCNN −0.040, L-TAE −0.052, L-TAE-S −0.044, CNN-BiLSTM −0.065, TimesNet −0.047, TabNet
−0.127; MiniRocket −0.020 [−0.040, 0.000]), while it raises XGBoost by 0.054 [0.032, 0.077] and
leaves LightGBM unchanged (+0.013, n.s.).

**Per-class F1 (field-bootstrap CIs).** Lucerne/medics 0.85 to 0.88 and canola 0.82 to 0.87 for
all four figure models; the cereals sit at barley 0.50 to 0.59, wheat 0.47 to 0.50, small-grain
grazing 0.39 to 0.46. The structural finding stands; the model ordering within each class now
follows the overall ranking (CNN-BiLSTM and L-TAE above XGBoost and TabNet on every cereal).

**Cereal merge.** With barley, wheat and small-grain grazing merged, macro-F1 rises by 0.17 to
0.25 for every model (3-class 0.63 to 0.85). The top of the ranking is unchanged (CNN-BiLSTM
0.850, TempCNN 0.849, L-TAE 0.848, Transformer 0.842, L-TAE-S 0.840), so the S-F sentence
"sparse models remain at the top" is withdrawn.

**Per-seed SD.** On the filled holdout the five-member SD of macro-F1 is 0.007 (L-TAE pixel),
0.010 (TempCNN) and 0.011 (CNN-BiLSTM), against 0.037, 0.015 and 0.023 under the published fill;
L-TAE field 0.018 (was 0.027), TabNet 0.007 (was 0.005). The published range "0.015 to 0.037" was
itself inflated by the zero fill, which amplified seed-to-seed disagreement on the September-gap
fields.

**Raw XGBoost (Table raw vs xr_fresh).** `experiments/xrfresh_vs_raw_xgb.py` reads
`merged_dl_test.parquet` without any fill, and that parquet holds NaN for the missing September
(it is the source of the 44.0 percent field-missing count above), so raw XGBoost routed September
through XGBoost's native missing-value branch rather than a zero. The table stands as a
raw-vs-xr_fresh comparison; note that the default direction for a feature never missing in
training is arbitrary, so the fairest version would use the training-median fill for both arms.

## Ensemble-size and loss/sampler ablations under the fill (2026-11-19)

The two supplement ablations (S-D iv and v) were re-inference on the zero-filled holdout.
`experiments/ablations/rescore_filled.py` re-scores the same checkpoints with
`HOLDOUT_FILL=train_median` (the xr_fresh TabNet reads the field features rebuilt from the
filled series); outputs `experiments/ablations/results/{ensemble_size_sweep,loss_sampler_ablation}_filled.csv`
with the published value alongside. The 5-seed CNN-BiLSTM (field) and L-TAE-S (field) rows
reproduce the fill check (0.591 and 0.599).

Ensemble size, holdout macro-F1 (published zero-fill value in brackets):

| Model | 1 seed, mean ± SD | 3 seeds | 5 seeds |
|---|---|---|---|
| CNN-BiLSTM (field) | 0.578 ± 0.014 (0.481 ± 0.018) | 0.586 (0.520) | 0.591 (0.506) |
| L-TAE-S (field) | 0.558 ± 0.020 (0.562 ± 0.022) | 0.575 (0.581) | 0.599 (0.601) |
| TabNet (xr_fresh, field) | 0.313 ± 0.020 (0.313 ± 0.024) | 0.391 (0.395) | 0.446 (0.433) |

Loss and sampler, holdout macro-F1 (published in brackets):

| Setting | CNN-BiLSTM (field) | L-TAE-S (field) | TabNet (xr_fresh, field) |
|---|---|---|---|
| focal, no resampling (published config) | 0.591 (0.509) | 0.599 (0.601) | 0.446 (0.433) |
| weighted CE | 0.589 (0.487) | 0.605 (0.598) | 0.427 (0.419) |
| plain CE | 0.572 (0.409) | 0.594 (0.588) | 0.388 (0.350) |
| focal + resampling | 0.547 (0.558) | 0.598 (0.600) | 0.392 (0.395) |
| plain CE + resampling | 0.594 (0.503) | 0.600 (0.601) | 0.458 (0.463) |

Two published sentences do not survive. "A single-seed L-TAE-S (0.56) already exceeds the
five-seed CNN-BiLSTM field ensemble": on filled inputs a single CNN-BiLSTM seed (0.578) exceeds a
single L-TAE-S seed (0.558), and the CNN-BiLSTM ensemble gains only 0.013 from five seeds against
L-TAE-S's 0.041. "The fragile CNN-BiLSTM swings widely (0.41 to 0.56) across objectives and even
at its best stays below L-TAE-S's worst": filled, CNN-BiLSTM spans 0.547 to 0.594 and L-TAE-S
0.594 to 0.605; the two ranges touch, and the published spread was the zero fill interacting with
the loss (plain CE lost 0.18 under zero fill and 0.02 under the training-median fill). What
survives: ensembling and the imbalance objective move scores by at most a few hundredths for the
raw-sequence models and never reorder them; the xr_fresh TabNet is unaffected by the fill
(its features are NaN-aware) and remains the model most sensitive to the objective.

## MiniRocket gate table (S-I) under the fill (2026-11-19)

Supplement Table S-I compared five classifier heads on the same frozen MiniRocket features and
reported the sparsemax gate 0.06 above the softmax gate at pixel level. It was scored under the
zero fill. `experiments/recent_baselines/minirocket_sparsity/rescore_filled.py` (ts_baselines env)
applies the saved heads to the training-median-filled holdout, transforming each seed's features
once and feeding every head the same batch; results in `minirocket_sparsity/results/filled_rescoring.csv`,
predictions `results/predictions/<name>_<level>_trainmed.csv`. The L2 reference reproduces the
fill check (0.605 pixel, 0.585 field).

| Head on MiniRocket features | Pixel, filled (published) | Field, filled (published) |
|---|---|---|
| Linear, L2 logistic (Table 1) | 0.605 (0.50) | 0.585 (0.48) |
| Linear, ridge | 0.620 (0.49) | 0.579 (0.45) |
| Linear, L1 logistic | 0.608 (0.51) | 0.577 (0.49) |
| Dense gate (softmax) | 0.623 (0.49) | 0.590 (0.50) |
| Sparse gate (sparsemax) | 0.620 (0.55) | 0.566 (0.50) |

Under a consistent in-range fill every head lands between 0.58 and 0.62 and the sparse gate's
advantage disappears: at pixel level it is 0.003 below the softmax gate (was 0.06 above), at field
level 0.024 below. The published gap was robustness to the out-of-range fill (the sparse gate lost
0.075 to the zero fill, the softmax gate 0.136), the same pattern as L-TAE-S against L-TAE. The S-I
sentence "adding the L-TAE-S style gate to MiniRocket's features raises its holdout macro-F1 by
0.06 over an identical network with a dense gate" is withdrawn; the table can stay as a second
backbone showing the fill effect.

## Step deletion: no imputation at all (2026-10-24)

`step_deletion.py` masks each pixel's missing months out of temporal attention (weight exactly 0,
remaining weights renormalized; for the Transformer also a key-padding mask and a mean over observed
steps), using the existing checkpoints with no retraining. A month is deleted if any band is
missing (41.5% of holdout pixels lose at least one month; none loses all). Checks: masked forward
with all steps observed equals `model(x)`; the output does not depend on a masked step's value;
the zero-fill rerun reproduces the existing zero-fill predictions (0 to 2 fields differ, AMP).

| Model | Zero fill | Training-median fill | Step deletion | Deletion − median [95% CI] |
|---|---|---|---|---|
| L-TAE | 0.540 | 0.652 | 0.653 | +0.001 [−0.007, 0.009] |
| L-TAE-S | 0.599 | 0.643 | 0.640 | −0.003 [−0.009, 0.002] |
| Transformer | 0.577 | 0.641 | 0.648 | +0.008 [−0.002, 0.018] |
| L-TAE + month dropout | 0.656 | 0.664 | 0.659 | −0.005 [−0.012, 0.002] |
| L-TAE-S + month dropout | 0.655 | 0.657 | 0.654 | −0.003 [−0.008, 0.002] |

Deleting the missing month is as good as filling it with the training median for every model
(no interval excludes zero), and far better than zero fill for models trained without dropout
(+0.04 to +0.11). Models trained with month dropout are indifferent among all three. So the
attention models need no imputation: what hurt them was a fill value outside the training
distribution, not the missing information. Paired field bootstrap, B = 10,000, seed 42.

## Nodata handling in general-purpose SITS codebases (2026-10-28)

Background audit of 17 widely used codebases, read at pinned commits without running them
(`research_notes/SITS nodata handling audit/audit.md`, with file and line links). The question was
whether our failure mode appears in public code: a constant written into a month that training
almost always observes, at inference only, before a training-fitted scaler.

| How gaps are handled | Count | Codebases |
|---|---|---|
| Masked consistently | 4 | SITS-BERT, U-TAE, WorldCereal/prometheo, EuroCropsML |
| Interpolated identically in training and prediction | 3 | R `sits`, eo-learn, Sen4AgriNet |
| Composited away upstream | 3 | Presto, CropHarvest, OpenMapFlow |
| No gap handling (irregular dates; cloudy values left raw) | 5 | BreizhCrops, PSE-TAE, L-TAE, TimeMatch, AI4Food starter pack |
| Zero padding without a mask, but identical in training | 1 | TSViT |
| Loader only | 1 | torchgeo PASTIS |

None reproduces our failure mode. The closest case is an edge path in OpenMapFlow's `_fillna`,
reused by Presto: a band that is NaN across a whole tile is filled with raw 0 before normalisation.
A month missing over a whole tile falls back to the band's mean over the other months, which is in
range. Both behaviours were checked against the pinned commit. WorldCereal is the strongest
counter-example: it masks missing composites and also simulates missing timesteps in training.

The paper should therefore not claim this failure is common in public code. The supported claim is
narrower: reference pipelines are protected by masking, consistent interpolation or compositing,
and a hand-built fixed monthly grid that reuses their architectures without that data handling
(like ours) loses the protection. Six codebases come from two groups, so the sample is not
independent; MCTNet and Che et al. were not checked. Whether published transfer failures had a
missing-data contribution is a separate paper-by-paper audit.

## Month-by-month missing-month injection (2026-10-28)

`month_injection.py` blanks each of the ten months in turn, for every pixel, on the holdout and on
the in-region test fields. The injected month is then zero-filled, filled with the training
median, or deleted from attention; natural gaps get the training median in every condition, so
each row isolates the injected month. 13 models, 5-seed ensembles, at most 50 pixels per field
(the same pixels everywhere). Results: `results/month_injection.csv`,
`figures/month_injection.pdf`.

Mean change in macro-F1 over the ten months relative to no injection (worst month in brackets):

| Model | Holdout, zero | Holdout, median | Holdout, deletion | In-region, zero | In-region, median |
|---|---|---|---|---|---|
| TempCNN, dense | −0.24 (−0.47) | −0.03 (−0.11) | | −0.30 (−0.56) | −0.04 (−0.10) |
| TempCNN, gated | −0.28 (−0.41) | −0.02 (−0.09) | | −0.35 (−0.52) | −0.03 (−0.09) |
| CNN-BiLSTM, dense | −0.23 (−0.47) | −0.03 (−0.13) | | −0.32 (−0.60) | −0.07 (−0.16) |
| CNN-BiLSTM, gated | −0.11 (−0.27) | −0.02 (−0.11) | | −0.17 (−0.26) | −0.04 (−0.11) |
| L-TAE | −0.18 (−0.42) | −0.03 (−0.10) | −0.02 (−0.08) | −0.22 (−0.46) | −0.04 (−0.10) |
| L-TAE-S | −0.08 (−0.19) | −0.01 (−0.06) | −0.01 (−0.04) | −0.09 (−0.19) | −0.03 (−0.09) |
| Transformer | −0.11 (−0.23) | −0.03 (−0.12) | −0.02 (−0.10) | −0.14 (−0.28) | −0.03 (−0.08) |
| any model + month dropout (6) | −0.01 to 0.00 (≤ −0.06) | −0.03 to −0.00 | −0.01 to 0.00 | −0.02 to 0.00 | −0.04 to −0.02 |

Zero fill damages every month for every model trained without dropout, in-region as much as on the
holdout. The failure is therefore not specific to September or to the transfer setting: zero-filling
any month that training always observed breaks the dense networks wherever they are applied. It is
worst in August to November, the months that separate the crops. The sparse gate softens it for
CNN-BiLSTM and L-TAE but not for TempCNN, as in the 2 x 2. Month dropout, trained on random months
and never targeted, removes the damage in every month. The injected losses are larger than the real
September effect (about 0.1) because every pixel loses the month, not 41%.

The median-fill and deletion rows isolate the real information loss, and they agree: most months
cost under 0.03, while October costs 0.04 to 0.16 for every model, with or without dropout. That is
genuine information no fill can restore, separate from the artifact. Deletion is slightly better
than the median fill for the attention models (mean −0.00 to −0.02 against −0.01 to −0.03). With
dropout, zero fill is as good as the median, because the dropout models were trained on that value.
These median-fill rows also serve as the leave-one-month-out occlusion test.
