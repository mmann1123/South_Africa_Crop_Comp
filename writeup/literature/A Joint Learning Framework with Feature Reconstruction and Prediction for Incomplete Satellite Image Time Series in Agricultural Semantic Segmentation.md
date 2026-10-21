# A Joint Learning Framework with Feature Reconstruction and Prediction for Incomplete Satellite Image Time Series in Agricultural Semantic Segmentation

**Citation:** Yuze Wang, Mariana Belgiu, Haiyang Wu, Dandan Zhong, Yangyang Cao, and Chao Tao
(2025). arXiv:2505.19159v1 [cs.CV], 25 May 2025 (preprint; the PDF in this folder is the arXiv
version, which carries no DOI). Published version: Wang, Belgiu, Wu, Zhong, Cao, Li, and Tao (2026),
"Joint Learning for Feature Reconstruction and Prediction in Agricultural Semantic Segmentation From
Incomplete Satellite Image Time Series," *IEEE Transactions on Geoscience and Remote Sensing*, vol.
64, pp. 1-23. DOI: `10.1109/TGRS.2025.3649017` (verified against Crossref on 2026-10-21: journal,
volume, and six of seven authors match; the print title was revised from the preprint title and one
author, Li, was added, so the Crossref title is not a substring match of the file name; treat the
DOI as verified for the published version and cite the preprint by arXiv id if quoting preprint
tables). Code: `https://github.com/wangyuze-csu/Joint_FRP`.

## Objectives

The paper asks how a multi-temporal semantic-segmentation model trained on complete satellite image
time series (SITS) should be made robust to time gaps caused by persistent cloud cover, which the
authors describe as both disrupting temporal dependencies and inducing "a shift in the overall
distribution of temporal information." Their stated motivation is that models trained on complete
SITS "often struggle to generalize to incomplete sequences," which is precisely a train/test
missingness mismatch.

Two prior families are set up as the comparison: Data Reconstruction (DR), which fills the gaps
before prediction (linear, closest, and last-value interpolation, plus the deep reconstructors
U-TILISE and GANFilling), and Data Augmentation (DA), which trains under simulated gaps (temporal
dropout, DA-TD, and window slicing, DA-WS). The paper's claims are that DR injects noise and is
redundant for prediction, that DA models overfit particular missing patterns, forget how to handle
complete SITS, and learn "shortcut" temporal reasoning, and that a joint feature-reconstruction plus
prediction objective distilled from a complete-SITS teacher resolves these problems.

Concrete objectives:

1. Propose a teacher-student framework in which the student sees randomly masked SITS and is trained
   to (a) reconstruct the teacher's standardized deepest temporal feature, (b) match the teacher's
   logits via KL distillation, and (c) predict ground-truth labels.
2. Show gains over DR and DA baselines on cropland extraction (binary) and crop classification
   (8-class) under both simulated and real-world missingness.
3. Show that the framework is backbone-agnostic (seven architectures) and sensor-agnostic
   (Sentinel-2 and PlanetScope).
4. Characterize performance as a function of temporal missing rate (0 to 75 percent) and ablate the
   four modules (mask, distillation, feature reconstruction, feature standardization).

## Methods

### Data

Three datasets, all with monthly composites for calendar year 2019 and 12 nominal time steps:

- `Hunan SEN`: Sentinel-2 L2A, 10 m (20 m bands resampled), 256 x 256 pixel samples, cropland vs
  background. Complete subset: 5,000 samples from seven city-level areas of north-eastern Hunan with
  all 12 months present. Incomplete subset: 3,048 samples from four city-level areas of north-western
  and south-eastern Hunan with only 4 to 9 months available (monthly composites with more than 40
  percent cloud were treated as missing; residual cloudy pixels linearly interpolated).
- `Hunan PLA`: PlanetScope-2, 3.7 m, four bands, cropland vs background. Complete subset 5,184
  samples (10 to 12 months, 1 to 2 missing months gap-filled by linear interpolation); incomplete
  subset 4,584 samples with 4 to 9 months.
- `Fr&Cat S4A`: Sentinel-2 from the Sen4AgriNet dataset, Western France and Catalonia, 366 x 366
  pixel samples, 10 bands, labels merged to 8 classes (background, cereals, vegetables and melons,
  root and tuber, leguminous, oilseed, sugar, other). Complete subset 4,771 samples; incomplete
  subset 2,925 samples with 4 to 9 months.

Labels for the Hunan datasets are visual interpretations. Inputs are normalized channel-wise by the
dataset-wide mean and variance.

### Representation of missing time steps

This detail matters for us. Missingness is represented by *dropping* the time step: the temporal
mask produces a shorter sequence `X' in R^{T' x C x H x W}` with `T' < T` (Eq. 1), and the backbones
(U-TAE, TSViT, ConvLSTM and variants, 3D U-Net) consume variable-length sequences. There is no
zero-filling, no sentinel value, and no explicit missingness indicator channel. Where gaps are
filled (the DR baselines) they are filled with interpolated values before standardization.

### Framework

A teacher `f_T` is trained on complete SITS and frozen. For each training sample the student `f_S`
receives a randomly masked copy (mask ratio drawn uniformly from 25 to 75 percent). Three losses:

- `L_FR`: squared L2 distance between channel-standardized versions of the teacher's and student's
  deepest temporally fused feature (U-TAE: last encoder layer after temporal attention; RNNs: final
  hidden state; TSViT: class-token embeddings). Standardization is motivated by Fig. 4, which shows
  that temporal missingness causes "a global shift of the feature's distribution," so raw feature
  matching rewards matching magnitudes rather than patterns.
- `L_KD`: KL divergence between temperature-softened teacher and student logits.
- `L_CE`: cross-entropy against ground truth.

`L = sigma L_FR + gamma L_KD + lambda L_CE`. U-TAE is the default backbone; AdamW, batch 4, 100
epochs, one A6000 GPU.

### Baselines

- Baseline: train on 60 percent of the complete subset, test directly on incomplete inputs.
- DR: Last-IN, Closest-IN, Linear-IN, U-TILISE, GANFilling (the deep reconstructors are retrained on
  the same 60 percent split with 25 to 75 percent simulated dropout).
- DA: DA-TD (random 25 to 75 percent temporal dropout) and DA-WS (contiguous windows covering 25 to
  75 percent of the sequence); 50 percent of samples augmented per epoch.

**Evaluation protocol:** Two regimes. (i) *Simulated*: the complete subset is split 60/40 into
train and test by sample (256 x 256 patches randomly sampled across seven administrative areas; no
statement that the split is spatially blocked, so train and test patches can be near-neighbours),
and 25 to 75 percent of time steps are dropped from each test sample. (ii) *Real-world*: the same
models are applied to the entire incomplete subset, which for `Hunan SEN` comes from different
administrative areas than the complete subset. The real-world regime is therefore simultaneously a
spatial holdout and a missingness holdout, and the two effects are not separated. In both regimes
the *training* data are complete for the Baseline and DR models and simulated-incomplete for the DA
models and the proposed method, while the *test* data are incomplete. This is a mismatched
train/test missingness regime for the Baseline and DR arms and a matched (simulated) regime for the
DA and proposed arms. The paper does compare imputation strategies at inference (three interpolators
and two deep reconstructors) against masking-based approaches. Metrics: OA, average accuracy, Kappa,
mIoU, mean F1 over classes, and per-class F1.

## Key Findings

### Cropland extraction, Hunan SEN (Tables 1 and 2, cropland F1 in percent)

| Method | Simulated | Real-world |
|---|---|---|
| Baseline (train complete, test incomplete) | 77.34 | 47.14 |
| Last-IN | 73.03 | 34.71 |
| Closest-IN | 75.11 | 36.57 |
| Linear-IN | 74.74 | 43.61 |
| U-TILISE | 74.39 | 55.05 |
| GANFilling | 76.62 | 43.90 |
| DA-TD | 77.47 | 62.23 |
| DA-WS | 78.60 | 67.32 |
| Proposed | 81.53 | 70.37 |

Every interpolation-based DR method is *worse* than feeding the shortened sequence directly to the
Baseline (by 0.93 to 5.57 points simulated, 3.5 to 12.4 points real-world). Training-time exposure
to gaps (DA-TD) raises real-world cropland F1 from 47.14 to 62.23, a 15-point gain. The Baseline's
collapse from 77.34 (simulated, random gaps) to 47.14 (real-world, contiguous gaps) is attributed to
U-TAE's attention having no observations to attend to inside long contiguous gaps.

### Crop classification, Fr&Cat S4A (Tables 5 to 8, mean F1 in percent)

| Method | Simulated | Real-world |
|---|---|---|
| Baseline | 72.46 | 67.59 |
| Linear-IN | 69.45 | 59.66 |
| Closest-IN | 71.47 | 63.50 |
| U-TILISE | 60.93 | 44.14 |
| GANFilling | 57.90 | 60.61 |
| DA-TD | 72.85 | 62.56 |
| DA-WS | 55.32 | 38.90 |
| Proposed | 79.96 | 70.18 |

Here even the DA methods fail to beat the Baseline in the real-world regime, and DA-WS scores 0.00
F1 on vegetables/melons and oilseed crops because contiguous windows can drop an entire short growth
cycle. Linear interpolation costs 8 mean-F1 points in the real-world regime, driven by root/tuber
(68.82 to 40.14) and leguminous (44.75 to 23.01) classes.

### Missing-rate sweep (Table 11, S4A mean F1, simulated)

| Mask ratio | Baseline | Proposed |
|---|---|---|
| 0 percent | 84.77 | 84.04 |
| 25 percent | 82.05 | 83.19 |
| 50 percent | 72.12 | 80.47 |
| 75 percent | 48.50 | 70.70 |

A model trained on complete sequences loses 2.7 mean-F1 points at 25 percent missing, 12.7 at 50
percent, and 36.3 at 75 percent. For cropland extraction on Hunan SEN (Table 9) the corresponding
Baseline cropland-F1 drops are 1.1, 3.5, and 13.2 points. The proposed method costs 0.7 to 0.9 points
on complete inputs, whereas the DA methods (Appendix A) lose accuracy on complete and 25 percent
inputs because they overfit the heavy-missing regime.

### Backbone dependence (Tables 13 and 14, cropland F1)

Baseline cropland F1 on simulated gaps: ConvLSTM 64.77, ConvGRU 68.28, U-ConvLSTM 79.89,
FPN-ConvLSTM 60.81, 3D U-Net 54.09, U-TAE 77.34, TSViT 76.77. On real-world gaps: 63.06, 66.69,
72.65, 50.79, 57.68, 47.14, 55.64. Attention backbones are the most robust to random gaps and the
*least* robust to contiguous gaps; 3D convolution is the least robust to random gaps because
spatio-temporal mixing is "embedded in the most fundamental units of the network" so the temporal
axis cannot be re-aligned when steps are removed. The proposed method lifts every backbone to 79.95
to 81.53 (simulated) and 69.05 to 74.10 (real-world).

### Module ablation (Tables 15 and 16, Hunan SEN, cropland F1)

Simulated: none 77.34, +mask 77.47, +KD 78.76, +FR 79.67, +FS 81.53. Real-world: none 47.14, +mask
62.23, +KD 64.11, +FR (without standardization) 62.16, +FS 70.37. Adding feature reconstruction
without feature standardization *hurts* in the real-world regime, which the authors attribute to
gap-induced "substantial distortions in feature magnitudes."

## Relevance to Our Crop-Classification Study

Our revised holdout analysis found that the test tile is missing September for 41 percent of
pixels, that September is never missing in training, and that zero-filling it before standardization
cost every raw-sequence deep model 0.08 to 0.14 macro-F1 and manufactured an apparent in-region /
out-of-region ranking inversion. Filling the month with training-tile medians restores the dense
temporal models to the top of the holdout ranking (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65,
Transformer 0.64, TimesNet 0.63; TabNet 0.62; tree, linear, and MiniRocket 0.59 to 0.61). This paper
is the closest agricultural analogue to that experiment and it bears on the revised findings in
several specific ways.

**It documents the same train-complete / test-incomplete shift, with numbers.** The Baseline arm is
exactly our old pipeline's situation: a dense temporal model trained on complete sequences and
evaluated on sequences with gaps it never saw. The authors call the consequence a "shift in the
overall distribution of temporal information" and a "global shift of the feature's distribution"
(Fig. 4). Their magnitudes are larger than ours because their gaps are larger: 25 percent missing
(three of twelve months) costs the U-TAE baseline 2.7 mean-F1 points on S4A crop classification,
and 50 percent costs 12.7 points. Our holdout removes one of roughly ten months for 41 percent of
pixels, a far milder gap, yet the zero-fill representation cost 8 to 14 macro-F1 points. That
contrast is itself informative: Wang et al. *drop* the missing step so the network sees a shorter but
in-distribution sequence, whereas our old pipeline substituted an out-of-distribution value
(post-standardization zero) at an in-distribution position. The paper therefore supports the revised
finding that the inversion was a representation artifact rather than evidence of dense-model
overfitting, and it suggests that the penalty we observed is dominated by the *value* used to fill
the gap rather than by the gap itself.

**It shows that naive imputation is worse than principled gap handling, which qualifies but does
not contradict our median fill.** Every interpolator (last, closest, linear) and both deep
reconstructors under-perform the shortened-sequence Baseline in five of six dataset-by-regime
comparisons, by as much as 12 cropland-F1 points (Last-IN, Hunan SEN real-world) and 8 mean-F1
points (Linear-IN, S4A real-world). Our training-tile median fill is a global-statistic imputation
closer in spirit to mean imputation than to any of their per-pixel temporal interpolators. It worked
for us because the missing month is the same month for every affected pixel, the training tiles
have that month fully observed, and a median is by construction near the standardized zero point of
the training distribution rather than far from it. Their results warn that this is not a general
recipe: with multi-month or contiguous gaps, their interpolators injected enough noise to hurt, and
we should state the median fill as a single-known-month fix rather than a missing-data strategy.

**It anticipates the training-time masking remedy and quantifies its ceiling.** Temporal dropout
during training (DA-TD, which simply randomly removes steps) recovers 15 cropland-F1 points in the
hardest regime (47.14 to 62.23) and closes most of the gap to the proposed method, but on S4A crop
classification DA-TD gives only +0.4 mean-F1 simulated and *loses* 5 points real-world relative to
the Baseline. This is relevant to the revised finding (2) on the sparsemax channel gate. The gate
gives +0.06 macro-F1 robustness to the zero-filled month for CNN-BiLSTM and L-TAE-S but no gain on
complete inputs. Wang et al.'s DA ablation (Appendix A) shows the same shape: robustness
interventions buy nothing on complete inputs and can cost accuracy there if they overfit the
degraded regime. A sparse gate that learns to down-weight a channel whose value is implausible is a
test-time analogue of their training-time mask, and both are bounded by what the remaining months
can support. Our observation that TabNet (sparse attention over tabular features) is *not* robust to
zero fill (0.61 to 0.51) is consistent with their finding that the mask-only module is the weakest
of the four and that standardization of the matched features (FS, +8 points real-world) is what
makes the sparse reconstruction work; sparsity alone is not a missingness fix.

**It provides architecture-specific robustness evidence that bears on our model ranking.** Their
3D U-Net baseline is the least robust backbone to random gaps (54.09 cropland F1 against 77.34 for
U-TAE), and their attention backbones are the least robust to contiguous gaps. Our pixel models are
1D temporal rather than 3D spatio-temporal, and our gap is a single month, so the mapping is loose,
but the direction supports treating any apparent "dense model fails on holdout" result as
suspicious until the missingness regime is matched. Under matched regimes their U-TAE sits at 84.77
mean F1 on complete S4A inputs, and under our median fill the dense models sit at 0.63 to 0.66
macro-F1: both are the top of their respective tables once the shift is removed.

**Learning curves.** The paper does not vary training-set size, so it neither supports nor
contradicts revised finding (3) (all models near 0.55 macro-F1 at 10 percent of fields, deep models
rising to about 0.65 at 100 percent, logistic regression plateauing at 0.61).

**What to cite it for.** (a) Independent evidence that a dense temporal model trained on complete
SITS degrades under test-time gaps even when the gap is handled by dropping rather than filling
(Table 11, 2.7 to 36.3 mean-F1 points over 25 to 75 percent missing). (b) Evidence that interpolation
before prediction can be worse than no interpolation (Tables 1, 2, 5, 7). (c) Evidence that
training-time temporal dropout is a cheap and effective, if imperfect, remedy. (d) The explicit
framing of missingness as a feature-distribution shift that must be standardized away, which is the
same framing our revision adopts for the zero-fill artifact.

**Matched or mismatched regime.** Mismatched for the Baseline and DR arms (train complete, test
incomplete); matched-by-simulation for the DA and proposed arms (train with random 25 to 75 percent
gaps, test with gaps). Real-world test gaps are drawn from nature and differ in structure
(contiguous) from the simulated training gaps, so even the DA arms face a residual pattern shift.

**Imputation strategies compared at inference.** Yes: last-value, closest, and linear interpolation,
U-TILISE, and GANFilling, all compared against sequence shortening. Zero-filling and
training-statistic (mean or median) filling are not tested.

## Evaluation Caveats

1. **Spatial leakage in the simulated regime.** The 60/40 split is by randomly sampled 256 x 256
   patches within the same seven administrative areas; nothing indicates spatial blocking, so the
   simulated numbers are in-region accuracies and the simulated-regime gains (for example 77.34 to
   81.53) are not transfer results.
2. **Spatial holdout confounded with missingness holdout.** For `Hunan SEN` the real-world subset is
   from different administrative areas than the complete subset, so the Baseline's drop from 77.34 to
   47.14 mixes region transfer, gap structure (contiguous vs random), and gap amount. The paper reads
   it as a gap-structure effect; a region effect cannot be excluded. For `Fr&Cat S4A` the geographic
   relationship of the two subsets is not stated.
3. **Labels for Hunan are visual interpretations** of Sentinel-2 RGB and PlanetScope imagery, not
   field survey; label noise is uncharacterized.
4. **Class imbalance is only partly addressed.** Mean F1 and per-class F1 are reported for S4A,
   which is good practice, but the cropland tasks are binary with background F1 above 91 percent in
   every row, so OA and Kappa differences there are driven almost entirely by the cropland class.
5. **No variance reporting.** Every number is a single run; no seeds, confidence intervals, or
   significance tests. Several reported gains (for example DA-TD vs Baseline at +0.13 cropland F1
   simulated) are well within what a single-seed difference could produce.
6. **No cross-year transfer** and no test on genuinely unseen regions with complete data, so the
   framework's effect on ordinary spatial transfer (our main axis) is unmeasured.
7. **Gap representation is not varied.** The paper only compares sequence shortening against value
   imputation. It never tests zero-filling, mean-filling, or an indicator channel, so it cannot say
   whether its interpolation losses come from interpolation error or from any fixed-length filled
   representation; that is the comparison our revision needed and had to run ourselves.
8. **Hyperparameters for the loss weights** (sigma, gamma, lambda) and the distillation temperature
   are not given in the preprint text, and the mask-ratio range "25 to 75 percent" is truncated mid
   sentence in the PDF; reproducibility relies on the released code.
9. **Computational cost** of the teacher-student setup (two forward passes per step, a pretrained
   teacher per dataset) is not reported.
10. **Preprint vs print.** Tables quoted here are from arXiv v1; the TGRS version is retitled and may
    differ in numbers. Re-check before citing specific values from the published article.
