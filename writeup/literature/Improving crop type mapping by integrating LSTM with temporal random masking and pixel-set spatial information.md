# Improving crop type mapping by integrating LSTM with temporal random masking and pixel-set spatial information

**Citation:** Xinyu Zhang, Zhiwen Cai, Qiong Hu, Jingya Yang, Haodong Wei, Liangzhi You, and
Baodong Xu (2024). *ISPRS Journal of Photogrammetry and Remote Sensing*, 218, 87-101.
DOI: `10.1016/j.isprsjprs.2024.10.013` (verified against Crossref on 2026-10-21: title, journal,
volume, pages, and author list match).

## Status: partial extraction

The PDF in this folder is a ScienceDirect article-preview page (abstract, keywords, full
introduction, section snippets of two to four sentences each, CRediT statement, and reference list),
not the full article; the university subscription did not cover the full text at download time.
Everything below is drawn from that preview. Methods, study-area, data, and evaluation-protocol
details that are not in the preview are flagged as unknown rather than inferred. The briefer should
be regenerated if the full text is obtained.

## Objectives

The paper proposes `Mask-PSTIN` (Mask Pixel-set SpatioTemporal Integration Network), combining a
training-time *temporal random masking* augmentation with a *pixel-set aggregation encoder* (PSAE)
feeding a bidirectional LSTM with temporal attention (AtBiLSTM). The motivation has three parts:

1. Satellite time series have "gaps and noise ... caused by clouds and rain," and limited, noisy
   training samples can cause a deep model "to overfit a specific temporal pattern or noise present
   in the training data." Temporal random masking is meant to increase training variability so the
   model learns "more generalizable and robust features."
2. Interpolation-based gap reconstruction, the authors note citing Che et al. (2024) and Zhao et al.
   (2021), "may not reliably reconstruct time series, offering limited improvement in crop type
   mapping and increasing computational costs." Augmentation is chosen as the alternative.
3. Pixel-level classifiers neglect within-field spatial context, and CNNs over-rely on texture that
   10 to 30 m imagery does not resolve; pixel-set encoders (Garnot et al. 2020) work but require
   field-parcel boundaries, which are unavailable in smallholder regions. PSAE aggregates spectral
   information from spatially neighbouring pixels in an image patch without parcel data.

Stated contributions: (1) the masking augmentation; (2) the PSAE plus AtBiLSTM architecture for
post-harvest and early-season mapping; (3) gradient-backpropagation feature-importance analysis;
and (4) evaluation "across three regions characterized by diverse agricultural landscapes and
cropping patterns."

## Methods

### Architecture (from the preview snippet and abstract)

`Mask-PSTIN` = temporal random masking (data augmentation, training only) + `PSTIN`, where `PSTIN`
= PSAE (adaptive aggregation of spectral vectors from the neighbourhood of the target pixel within
an image patch) followed by a BiLSTM with a temporal attention layer. Inputs are patches of
time-series satellite imagery; outputs are per-pixel crop labels. Temporal random masking
"selectively masks observations at certain time steps in the multi-temporal satellite images during
the training." The preview does not state the masking rate, whether masked steps are removed or
replaced by a value (zero, mean, or otherwise), whether a mask indicator is passed to the LSTM, or
whether masking is applied at inference.

### Data

Region A is Auvergne, central France (Allier, Puy-de-Dome, Cantal, Haute-Loire; 44.6 to 46.8 N,
2 to 4.5 E), temperate. Regions B and C are not identified in the preview; the introduction's
emphasis on "smallholder farming regions like southern China" suggests at least one Chinese site,
but this is an inference. Sensor, band set, temporal resolution, compositing, cloud masking, and the
number and source of labels are not given in the preview.

### Baselines

RF, SVM, Transformer, and CNN-LSTM, plus the ablation ladder LSTM, Mask-LSTM, PSTIN, Mask-PSTIN.

**Evaluation protocol:** Unknown in detail. The abstract reports "average overall accuracy (OA)"
and "mean F1-Score (mF1)" averaged across the three regions, plus early-season accuracy as a
function of observation cut-off date. The preview does not say whether train/test splits are
pixel-random, field-wise, or spatially blocked within each region, whether any model was trained
in one region and tested in another, or whether test data were drawn from the same year as
training. Nothing indicates a cross-region or cross-year transfer test; the most likely reading of
"evaluated across three regions" is three independent within-region experiments. Whether the test
series contain cloud gaps, and whether those gaps match the training series, is not stated, so the
train/test missingness regime cannot be classified from the preview. No comparison of imputation
strategies at inference is mentioned anywhere in the abstract, introduction, or snippets.

## Key Findings

All numbers are three-region averages from the abstract and the ablation snippet.

| Model | Avg OA (percent) | Avg mF1 |
|---|---|---|
| LSTM (baseline) | 80.9 | 0.781 |
| PSTIN (adds PSAE, no masking) | 83.9 | 0.818 |
| Mask-PSTIN (adds temporal random masking) | 87.4 | 0.865 |

- Spatial aggregation (PSAE) alone adds 3.0 OA points and 0.037 mF1 over a plain LSTM.
- Temporal random masking on top of PSAE adds a further 3.5 OA points and 0.047 mF1, so the
  augmentation is worth slightly more than the architecture change in this study.
- The `Mask-LSTM` row of Table 2 (masking without PSAE) exists but its values are cut off in the
  preview; the snippet says only that it improved on the basic LSTM.
- `Mask-PSTIN` "significantly outperformed" RF, SVM, Transformer, and CNN-LSTM in all three regions
  (values not in preview).
- Early-season: `Mask-PSTIN` identified crop types "earlier and more accurately ... before or during
  their developing stages" than the machine-learning baselines (dates and curves not in preview).
- Gradient-based feature importance shows `Mask-PSTIN` "exhibiting broader attention across various
  time steps" than the unmasked model, which the authors read as the masked model relying on more
  of the season rather than a few peak dates.
- Cited by 30 as of the download date (October 2026), so the technique has had uptake.

## Relevance to Our Crop-Classification Study

Our revised findings are: (1) the holdout tile lacks September for 41 percent of pixels, that month
is never missing in training, and zero-filling it before standardization in the old pipeline cut
every raw-sequence deep model's holdout macro-F1 by 0.08 to 0.14 and produced an apparent
in-region / out-of-region ranking inversion; with the month filled from training-tile medians the
dense temporal pixel models lead (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64,
TimesNet 0.63), TabNet is 0.62, and tree, linear, and MiniRocket models sit at 0.59 to 0.61 with no
inversion; (2) a sparsemax channel gate gives no transfer gain on complete inputs but +0.06
macro-F1 robustness to the zero-filled month on CNN-BiLSTM and L-TAE-S, while TabNet itself falls
from 0.61 to 0.51 under zero fill; (3) learning curves on the corrected holdout put every model
near 0.55 macro-F1 at 10 percent of fields (XGBoost 0.51), with deep models rising to about 0.65 at
100 percent and logistic regression plateauing at 0.61.

**What the paper shows about missing-timestep handling.** It is a training-side intervention:
randomly hide time steps during training so that the model cannot depend on any one of them. The
reported effect size, +3.5 OA and +0.047 mF1 averaged over three regions on top of an already
spatially aware model, is substantial and, notably, is obtained without any change to the test
data. That is the same qualitative lever as the DA-TD arm in Wang et al. (2025), where random
temporal dropout lifted real-world cropland F1 from 47.14 to 62.23. The paper also contains, in its
own introduction, the explicit judgement that interpolation-based reconstruction offers "limited
improvement ... and increasing computational costs," citing Che et al. (2024), so it aligns with
the position that gap *handling* rather than gap *filling* is the productive direction.

**How it bears on revised finding (1).** The paper neither tests nor discusses a model trained on
complete series and evaluated on gapped ones, nor does it compare fill values. It therefore does not
directly support or refute the claim that the inversion was a zero-fill artifact. What it does
support, indirectly, is the mechanism: a BiLSTM that has never been forced to cope with an absent
step learns to lean on specific dates (their Fig. on gradient importance shows the unmasked model's
attention concentrated on fewer steps), which is exactly the failure mode that a never-seen zero at
a never-missing position would exploit. Their masked model distributes attention across the season,
which is what makes it robust. Our corrected ranking, with CNN-BiLSTM at the top once the month is
plausibly filled, is consistent with their finding that a BiLSTM-with-attention is the strongest
family they tested, but their comparison is within-region and ours is cross-tile, so the agreement
is in direction only.

**How it bears on revised finding (2).** Temporal random masking during training and a sparsemax
channel gate at inference are complementary routes to the same target: making the network's
decision insensitive to one corrupted or absent input. Their route changes the *training
distribution* so the network has seen absence; ours changes the *architecture* so the network can
zero out a channel. Their gain (+0.047 mF1 over PSTIN) is of the same order as our gate's robustness
gain (+0.06 macro-F1 under zero fill), but theirs appears on ordinary test data while ours appears
only under corruption and is nil on complete inputs. That contrast suggests a worthwhile ablation
for the revision: train CNN-BiLSTM and L-TAE with random month dropout (their recipe) and test on
both the median-filled and zero-filled holdout. If masking-trained models match the gated models
under zero fill *and* gain on complete inputs, the training-side remedy dominates. The paper's
result that masking helps even when the test data are presumably not heavily gapped also qualifies
our interpretation of the gate: part of what random masking buys is regularization against
overfitting a small, noisy training set, not only robustness to gaps, and the same may be true of
the gate's +0.06.

**How it bears on revised finding (3).** The introduction frames masking explicitly as a remedy for
"insufficient and noisy training data," which predicts that its benefit should be largest at small
training fractions. Our learning curves show all models converging to about 0.55 macro-F1 at 10
percent of fields; a masking-augmented deep model is the natural candidate for breaking that floor,
and the paper gives a prior that it might. No learning-curve data are in the preview, so this is a
hypothesis the paper motivates rather than a result it reports.

**Granularity and features.** Their PSAE aggregates neighbouring pixels within a patch without
parcel boundaries, a middle ground between our pixel-level models and our field-level (FID)
majority vote. Since our labels are field polygons, we already have the parcel constraint that
Garnot-style pixel-set encoders need; PSAE's advantage is specific to settings without boundaries
and is not something our pipeline lacks.

**What to cite it for.** (a) A three-region crop-type result that training-time temporal random
masking adds +3.5 OA and +0.047 mF1 to a spatially aware BiLSTM (abstract; Table 2). (b) A published
remote-sensing statement, with citations, that interpolation of gaps gives limited improvement and
costs compute. (c) Gradient-importance evidence that masking spreads a BiLSTM's attention across
the season. Do not cite it for transfer, for fill-value comparisons, or for any specific
train/test-split claim until the full text is checked.

**Matched or mismatched regime.** Cannot be determined from the preview. Training series are
artificially masked; whether test series have natural gaps, and at what rate, is not stated.

**Imputation strategies compared at inference.** None reported. The paper compares an augmentation
strategy, not fill values.

## Evaluation Caveats

1. **Partial text.** All quantitative claims above rest on the abstract and one ablation snippet.
   Train/test design, region identities for B and C, sensor, band set, temporal sampling, label
   source and count, masking rate, and the treatment of masked steps (removed vs value-replaced)
   are unknown. Treat this briefer as provisional.
2. **Probably in-region evaluation.** "Evaluated across three regions" reads as three separate
   within-region experiments. Unless the full text shows otherwise, the 87.4 OA / 0.865 mF1 are
   in-region numbers and not comparable to our spatially disjoint holdout macro-F1.
3. **Spatial leakage risk.** The model classifies pixels from patches; if train and test pixels
   are sampled from the same fields or adjacent patches, the PSAE's spatial aggregation would
   directly leak neighbouring-label information into test predictions. Whether splits are
   field-wise or patch-disjoint is not stated in the preview.
4. **Class balance.** mF1 is reported, which is the right metric, but per-class F1 and class
   frequencies are not in the preview, so minority-crop behaviour is unknown.
5. **Confounded ablation.** Masking is only evaluated jointly with the authors' architecture in the
   ladder shown; the `Mask-LSTM` value, which isolates the augmentation on a plain LSTM, is cut off.
6. **No gap-shift test.** The paper claims robustness to gaps and noise but, from the preview, does
   not evaluate on series with more or different gaps than the training series, so the robustness
   claim is supported by regularization gains on ordinary test data rather than by a stress test.
7. **No cross-year transfer** and no reported compute cost.
8. **Regularization vs robustness.** The masking gain may be partly or wholly a small-sample
   regularization effect; the paper's own framing (overfitting to "insufficient and noisy training
   data") invites that reading, and it cannot be separated from gap robustness without a learning
   curve or a stress test, neither of which appears in the preview.
