# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

**Citation:** Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long (2023).
*International Conference on Learning Representations (ICLR 2023)*. arXiv:2210.02186 (v3, 12 Apr 2023);
OpenReview id `ju_Uqw384Oq`. DOI: not found in PDF. A Crossref title search
(`query.title=TimesNet Temporal 2D-Variation Modeling for General Time Series Analysis`) returned no
matching record (top hits were a 2024 IEEE CYBER paper that cites TimesNet and a later book chapter by
the same group with a different title), consistent with ICLR proceedings not being DOI-registered. Cite
via arXiv/OpenReview. Code: `https://github.com/thuml/TimesNet` (now maintained inside THUML's
Time-Series-Library, `TSLib`).

## Objectives

TimesNet proposes a single, task-general deep backbone for time-series analysis, evaluated on five task
families: long- and short-term forecasting, imputation, sequence classification, and unsupervised
anomaly detection. It is a general machine-learning paper, not a remote-sensing or agricultural one;
there is no crop, land-cover, or satellite data anywhere in it.

The motivating claim is that real-world series are *multi-periodic* (daily plus weekly, weekly plus
quarterly, and so on), and that each time point varies along two axes at once: relative to its
neighbours within the same period (*intraperiod-variation*) and relative to the same phase in adjacent
periods (*interperiod-variation*). A 1D sequence makes only the first axis explicit. The authors argue
that reshaping the 1D series into a 2D tensor, with one axis indexing position within a period and the
other indexing which period, exposes both kinds of variation as spatial locality, so that ordinary 2D
convolution kernels (borrowed from computer vision) can model them jointly.

The secondary objective is to "bridge" time series to 2D vision backbones: the inception block used in
the main experiments can be swapped for ResNet, ResNeXt, ConvNeXt, or a Swin Transformer block.

For our purposes only the classification task matters. There, the stated objective is to show that the
2D representation supports the *hierarchical* features that sequence-level classification needs, where
forecasting-oriented models such as DLinear do poorly.

## Methods

**Period discovery (FFT).** For an input `X_1D` of length `T` with `C` variates, the model computes the
real FFT along time, averages amplitudes over variates (and, in the reference code, over the
mini-batch), zeroes the DC term, and picks the top-`k` frequencies `f_1..f_k` by amplitude (Eq. 1).
Each frequency gives a period `p_i = ceil(T / f_i)` (the code uses integer division `T // f_i`).
Only frequencies in `{1, ..., floor(T/2)}` are considered. `k` is a hyperparameter (`k = 3` for
classification).

**1D to 2D reshape.** For each selected period the sequence is zero-padded to a multiple of `p_i` and
reshaped to a `(T/p_i) x p_i` grid per channel (Eq. 3), so columns run along phase within a period and
rows run across successive periods. This produces `k` different 2D "views" of the same series.

**TimesBlock.** Inside each residual `TimesBlock` (Eq. 4), the period search is repeated on the current
deep features (not only the raw input; an ablation shows transforming only the raw input is worse).
Each 2D view passes through a shared, "parameter-efficient" `Inception_Block_V1` (Eq. 5): a bank of
2D convolutions with kernel sizes `1, 3, 5, ..., 2*num_kernels-1`, averaged, then GELU, then a second
inception bank back to `d_model`. Weights are shared across the `k` views so parameter count does not
grow with `k`. Outputs are reshaped back to 1D, truncated to length `T`, and fused by a softmax over
the selected frequencies' FFT amplitudes (Eq. 6, "adaptive aggregation", borrowed from Autoformer's
auto-correlation weighting).

**Embedding and heads.** Inputs are projected to `d_model` by a token embedding (a 1D convolution with
kernel 3 and circular padding in `TSLib`) plus a fixed sinusoidal positional embedding over step index.
For classification the final `T x d_model` representation passes through GELU and dropout, is masked
by the padding mask, flattened, and fed to a single linear layer to `num_class` logits. `d_model` is
set by a rule `min(max(2^ceil(log C), d_min), d_max)` with task-specific bounds (Table 7). Forecasting
also uses the Non-stationary Transformer's series normalisation; the classification path does not.

**Classification setup.** Ten multivariate datasets from the UEA archive (Bagnall et al., 2018),
preprocessed following Zerveas et al. (2021): `EthanolConcentration`, `FaceDetection`, `Handwriting`,
`Heartbeat`, `JapaneseVowels`, `PEMS-SF`, `SelfRegulationSCP1`, `SelfRegulationSCP2`,
`SpokenArabicDigits`, and `UWaveGestureLibrary`. Sequence lengths range from 29 (`JapaneseVowels`) to
1751 (`EthanolConcentration`); channel counts from 3 to 963 (`PEMS-SF`); training-set sizes from 120
(`UWaveGestureLibrary`) to 6599 (`SpokenArabicDigits`) sequences. Configuration (Table 7): `k = 3`,
2 layers, Adam with learning rate `1e-3`, cross-entropy, batch size 16, 30 epochs. More than 15
baselines are compared, including classical DTW, XGBoost, and Rocket; RNNs (LSTM, LSTNet, LSSL); TCN;
nine Transformer variants (vanilla, Reformer, Informer, Pyraformer, Autoformer, Non-stationary,
FEDformer, ETSformer, Flowformer); and MLPs (DLinear, LightTS). Baselines share the same input
embedding and final projection, so only the backbone differs. Each experiment is repeated three times
on a single TITAN RTX GPU. The sole metric is accuracy.

**Evaluation protocol:** For classification, each UEA dataset is evaluated on its fixed, archive-provided
train/test split, and the paper's own dataset table lists zero validation samples for every
classification dataset (format `(train, 0, test)`). Reported accuracy is therefore a single test-split
score per dataset, averaged over three runs (standard deviation stated as within 0.1%) and then
averaged across the ten datasets. There is no cross-validation, no held-out validation set described
for epoch or hyperparameter selection, no significance testing, and no notion of domain, region, or
distribution shift: train and test are drawn from the same collection campaign for each dataset. On
our spectrum this is in-distribution, i.i.d.-split evaluation; it is not comparable to a spatially
disjoint holdout. Because no validation split is reported, any early stopping or checkpoint choice
would have to use the test split itself (we did not audit the `TSLib` training runner, only the
vendored model files); the paper does not discuss this, so the 73.6% figure
should be read as potentially optimistic model selection on test, not as held-out generalisation.

**Leakage and imbalance.** The spatial-leakage question does not arise in the remote-sensing sense
(there are no fields or pixels), but the archive splits of several UEA datasets are by recording rather
than by subject or session, so subject-level leakage is plausible for some of them; the paper does not
address it. Class imbalance is not discussed and only accuracy is reported, with no macro-F1, per-class
recall, or Kappa. Several UEA datasets are balanced by construction, so accuracy is less misleading
there than it would be on our lucerne-dominated data, but the paper offers no balanced metric.

## Key Findings

**Classification (Fig. 5, Table 17).** TimesNet reports the best ten-dataset average accuracy, 73.6%,
versus Flowformer 73.0%, Rocket 72.5%, Non-stationary Transformer 72.7%, Informer 72.1%, LSTNet 71.8%,
DTW 67.0%, DLinear 67.5%, XGBoost 66.0%, and LSTM 48.6%. Per-dataset TimesNet accuracies are
`EthanolConcentration` 35.7, `FaceDetection` 68.6, `Handwriting` 32.1, `Heartbeat` 78.0,
`JapaneseVowels` 98.4, `PEMS-SF` 89.6, `SelfRegulationSCP1` 91.8, `SelfRegulationSCP2` 57.2,
`SpokenArabicDigits` 99.0, `UWaveGestureLibrary` 85.3.

**The average hides that classical methods win several datasets outright.** Rocket beats TimesNet on
`EthanolConcentration` (45.2 vs 35.7), `Handwriting` (58.8 vs 32.1), and `UWaveGestureLibrary`
(94.4 vs 85.3); XGBoost beats it on `PEMS-SF` (98.3 vs 89.6), the highest-dimensional dataset; and
plain DTW beats it on `UWaveGestureLibrary` (90.3). TimesNet's 1.1-point average margin over Rocket
comes mainly from two datasets where Rocket is weak, `SpokenArabicDigits` (99.0 vs 71.2) and `PEMS-SF`
(89.6 vs 75.1). With
ten datasets, no variance across splits, and no paired test, the ranking among the top five or six
methods is not statistically established.

**Representation analysis (Fig. 6).** Using CKA similarity between first- and last-layer
representations, the authors argue that classification and imputation reward *hierarchical*
(dissimilar-across-layers) representations, while forecasting and anomaly detection reward low-level
ones. TimesNet sits at the favourable end for each task; FEDformer does well in forecasting but learns
less hierarchical features and does worse in classification. They attribute DLinear's poor
classification (67.5%) to its single temporal linear layer.

**Other tasks.** TimesNet is reported best on long-term forecasting in over 80% of settings, best on
M4 short-term forecasting (SMAPE 11.829), best on imputation across ETT, Electricity, and Weather at
12.5-50% masking, and best average F1 (about 85.5%) across five anomaly-detection benchmarks. These are
outside our scope.

**Ablations (Appendix C).** Swapping the shared inception block for ResNeXt, ConvNeXt, or Swin blocks
improves anomaly-detection F1 slightly (up to about 86.9% vs 85.5%), at higher cost; independent
per-period inception weights also help slightly but make size depend on `k`. Adding Autoformer-style
trend decomposition does not help; the authors say that for series without clear periodicity the
model falls back to frequency 1 (period equals the full length), so the "2D" view collapses to the
original sequence. Performing the 2D transform only on raw input rather than in every block degrades
F1 (85.49% to 84.85%). Removing the softmax or using uniform weights in aggregation is worse. All
ablations are on anomaly detection, none on classification.

**Sensitivity and efficiency.** Classification accuracy is reported as robust to the choice of `k`
(tested on `SpokenArabicDigits` and `FaceDetection`). On ETTh1 imputation TimesNet has about 0.067 MB
of parameters and is the most efficient of the top three models; its parameter count is invariant to
series length because the 2D kernels and shared inception block do not depend on `T`.

## Relevance to Our Crop-Classification Study

TimesNet enters our study as one of the "recent baselines" requested in revision (reviewer R2-Q3a): a
widely cited, post-2022, general-purpose temporal deep net that a reader could reasonably expect to
beat our L-TAE/Transformer/TempCNN set. It is not a crop or remote-sensing model, has never been
evaluated under any spatial-transfer protocol, and its claims rest entirely on in-distribution UEA
accuracy. Its value to the manuscript is therefore as a stress test of our thesis, not as an
external comparator for transfer numbers.

**How we apply it.** `experiments/recent_baselines/train_timesnet.py` vendors TimesNet unmodified from
`TSLib` (commit `4e938a1`, `tslib_timesnet/`) and runs it inside the exact L-TAE-S harness. Via
`common.py`, inputs are the 60 chronological `merged_dl_train` columns (6 bands `B2`, `B6`, `B11`,
`B12`, `EVI`, `hue` x 10 months) reshaped to `(N, T=10, C=6)`; NaNs filled with 0; FID-wise 64/16/20
split with `random_state=42`; `StandardScaler` fitted on training rows; and `check_against_published`
asserts that feature columns, label classes, split sizes, and scaler statistics equal those of the
published L-TAE-S model. Both a field level (pixels averaged per FID before the split, 2653/664/830
train/val/test fields) and a pixel level (per-pixel, field majority vote at prediction) variant are
trained. The training protocol is copied from L-TAE-S: weighted focal loss (`gamma = 2`, inverse-count
`alpha`), AdamW (`lr = 1e-3`, weight decay `1e-4`), cosine schedule over 100 epochs, early stopping
with patience 15 on validation macro-F1, AMP, five seeds with mean logits. Only the backbone changes.
The TimesNet config is `d_model = 32`, `d_ff = 64`, `e_layers = 2`, `top_k = 2`, `num_kernels = 6`,
dropout 0.1, fixed before any holdout result was seen. Holdout predictions are written per field for
the spatially disjoint `34S_20E_259N` tile. Unlike the paper, model selection uses a proper validation
split and the headline metric is holdout macro-F1, so whatever TimesNet scores here is measured under a
stricter protocol than its own paper used.

**Why the FFT period design is a poor fit for our series.** TimesNet's central inductive bias is
multi-periodicity within a long sequence. Our sequences have `T = 10` and describe a single growing
season, which contains at most one phenological cycle and no repeated period at all. With `T = 10`, the
real FFT has only five non-DC bins, so the candidate periods are `10 // f` for `f = 1..5`, that is
`{10, 5, 3, 2, 2}`. The resulting 2D grids are `1 x 10` (a single row, so there is no interperiod axis
and the 2D kernel reduces to a 1D one), `2 x 5`, `4 x 3` (after padding 10 to 12), and `5 x 2`. These
"periods" are spectral artefacts of a short window, not agronomic cycles; the "interperiod" rows simply
juxtapose, for example, January-April with July-November months that happen to share a phase index.
The inception bank's larger kernels (up to `11 x 11` with `num_kernels = 6`) are far larger than these
grids, so most of each kernel lands on zero padding. The paper itself concedes that without clear
periodicity the model falls back to frequency 1, where the 2D view is the original sequence; on our
data TimesNet therefore behaves largely as a heavily parameterised temporal CNN with an
amplitude-weighted ensemble of a few reshapes.

**Irregular sampling and the dropped months.** We drop months `05` and `06` (missing data and cloud),
so the sequence is January-April followed directly by July-December. TimesNet, like the FFT, assumes
uniform sampling: its positional embedding indexes steps `0..9`, the classification path passes no
calendar marks (`x_mark = None`), and the FFT treats April-to-July as one step. Our L-TAE and L-TAE-S
instead encode true month positions (`MONTH_POSITIONS = [1, 2, 3, 4, 7, ..., 12]`). The token
embedding's circular padding also makes December and January neighbours, which is calendar-plausible
but joins the ends of one season's record. None of this is fatal, but it means TimesNet's period
estimates are computed on a time axis that is distorted exactly where the winter-crop season begins.
A further quirk is that `FFT_for_Period` averages amplitudes over the whole mini-batch, so a sample's
selected periods, and hence its prediction, depend on which other samples share its batch;
`train_timesnet.py` fixes evaluation batch size and order for reproducibility, but the dependence means
holdout predictions are conditioned on holdout batch composition, a subtle source of
train/holdout mismatch under distribution shift.

**Capacity relative to our overfitting finding.** In our configuration TimesNet has about 2.35 M
trainable parameters (counted from the vendored model with 5 classes), almost all in the two pairs of
dense inception banks (each `32 x 64` channels across six kernel sizes). That is roughly 880
parameters per training field at the field level, for a 60-number input. Its features are dense: every
inception filter mixes all embedding channels and all bands, and nothing resembles the sparse,
axis-aligned feature selection that our results tie to transfer robustness (TabNet's sparsemax masks,
trees, and the L-TAE-S channel gate). Our manuscript's prediction is therefore clear: TimesNet should
behave like the dense temporal cluster (L-TAE, Transformer, TempCNN, CNN-BiLSTM), scoring well
in-region and losing substantially more on the holdout tile than the sparse learners, with field-level
aggregation partly closing the gap. The 2-epoch field-level smoke run already reaches in-region
macro-F1 0.62, which says nothing yet about transfer. If the full run confirms the prediction, TimesNet
is a strong addition: a recent, high-profile architecture whose in-distribution SOTA claims do not buy
spatial transfer, reinforcing that model rankings are an artefact of the validation protocol. If it
transfers well, that would challenge the "sparsity, not capacity" thesis and need explaining (for
instance, by the large-kernel convolutions acting as heavy temporal smoothing).

**Indirect support for our framing.** Within the paper's own Table 17, Rocket and even XGBoost beat
TimesNet on four of ten UEA datasets, and its average lead is about one point with no significance
test. That pattern, a deep model winning on average while classical, feature-based methods win
individual datasets, echoes the long-standing "bake-off" finding that convolutional random-feature and
tree methods remain hard to beat on small time-series classification problems. It supports including
a Rocket-family baseline alongside TimesNet.

## Evaluation Caveats

All classification evidence is i.i.d. within-dataset accuracy on fixed UEA splits, with no reported
validation set, no cross-validation, three repeats, and no significance testing; the 73.6% average is
not a generalisation-under-shift result and should never be cited as evidence that TimesNet transfers.
No domain, spatial, temporal (cross-year), or cross-sensor shift is tested for any task.

Only accuracy is reported for classification. There is no macro-F1, per-class recall, Kappa, or
calibration, so behaviour on minority classes is unknown. Our lucerne-dominated, five-class winter-crop
data needs exactly those metrics.

The UEA sequences (29 to 1751 steps, regularly sampled sensor streams such as EEG, audio, gesture, and
traffic) differ greatly from ours: 10 irregular monthly composites of six optical bands and indices.
The architectural rationale (multiple periods within one sample) does not hold for a single-season
crop profile. The paper offers no result on series as short as ours.

Ablations that justify design choices (vision backbone swap, decomposition, raw-vs-deep transform,
aggregation) were run only on anomaly detection, not classification, so it is unknown which components
matter for classification.

Computational cost is reported only for ETTh1 imputation (very small `d_model`); in our classification
configuration the model is about 2.35 M parameters, orders of magnitude larger than the reported
0.067 MB, because `d_model`/`d_ff` and the inception kernel bank dominate. Cite the paper's efficiency
figures with that caveat.

The period-selection step is batch-dependent in the reference implementation (amplitudes averaged over
the mini-batch), which the paper does not mention; results can shift with batch size or composition.

No DOI exists in the PDF and none was found in Crossref; cite as ICLR 2023 with arXiv:2210.02186.
Cite TimesNet in our manuscript as a recent, general-purpose dense temporal baseline evaluated under our
spatial-holdout protocol, and attribute its published numbers explicitly to in-distribution UEA
accuracy.
