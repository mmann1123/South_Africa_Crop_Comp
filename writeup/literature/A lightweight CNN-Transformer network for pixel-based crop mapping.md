# A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery

**Citation:** Yumiao Wang, Luwei Feng, Weiwei Sun, Lihua Wang, Gang Yang, and Binjie Chen (2024).
*Computers and Electronics in Agriculture* 226, 109370. DOI: `10.1016/j.compag.2024.109370`
(verified: extracted from the PDF's first page; Crossref returns the identical title, authors,
volume 226, article 109370, 2024). Received 31 Mar 2024, revised 26 Jul 2024, accepted 21 Aug 2024.
Data "available on request"; no code link. Same journal as our current submission.

## Objectives

The paper proposes MCTNet, a lightweight pixel-level crop classifier that stacks several
CNN-Transformer fusion (`CTFusion`) stages, and an attention-based learnable positional encoding
(`ALPE`) meant to make the Transformer branch robust to missing observations from cloud cover. The
authors motivate it with two gaps: most CNN-Transformer hybrids in crop mapping are patch-based and
encoder-decoder (expensive labels, large models), and prior hybrids "gave limited consideration to
missing data." They explicitly reject interpolation and Savitzky-Golay smoothing as a remedy,
arguing that reconstruction is costly at scale and of uncertain fidelity, and instead want a model
that consumes gappy series directly. Contributions claimed: (1) MCTNet with the fewest parameters of
nine compared models; (2) `ALPE` to reduce the impact of missing values; (3) superior accuracy,
parameter count, and map quality relative to eight baselines.

## Methods

**Study areas and data (Sec. 2.1-2.2).** Two US states, Arkansas (large fields; soybeans, rice,
corn, cotton, others) and California (small fragmented plots; grapes, rice, alfalfa, almonds,
pistachios, others), year 2021. Sentinel-2 Level-2A, ten bands (`B2`-`B8`, `B8A`, `B11`, `B12`; 60 m
bands dropped). Cloud-affected pixels removed, then median composites at ten-day intervals give 36
time steps, so each sample is 10 bands x 36 steps = 360 features. Labels come from the USDA Cropland
Data Layer (CDL), filtered at 95 % CDL confidence, with ESA WorldCover 2021 masking non-cropland.
Classes under 5 % of samples merged into "others". All preprocessing in Google Earth Engine.

**Missing-value handling (Sec. 2.2.4, 2.3).** Composites that remain empty after cloud removal are
set to `0` ("we used 0 to mark the missing data"). The model receives two inputs: the zero-filled
series (Input 1) and a binary missingness indicator per time step (Input 2). Only the first
Transformer sub-module uses the indicator: `ALPE(t) = ECA(Conv1D(PE(t) x mask))`, i.e. the
sinusoidal positional encoding is zeroed at missing steps, passed through a 1D convolution and an
Efficient Channel Attention block, and added to the input. The CNN branch and later stages see the
zero-filled values with no mask. Training and test data are zero-filled identically, so there is no
train/test difference in fill rule.

**Architecture.** Three `CTFusion` stages (selected from 1-5), each a parallel 1D CNN sub-module
(two conv + batch-norm layers with a residual skip) and a Transformer-encoder sub-module (5 heads),
with pooling between stages; global max pooling and a linear softmax classifier. 55,059 parameters
in Arkansas, against 95,301 for L-TAE and 1.3 M for the Transformer baseline.

**Baselines.** SVM and RF (scikit-learn, on the 360 raw features), LSTM, Transformer encoder
(Russwurm and Korner 2020 variant), ResNet-18, TAE, L-TAE, and GL-TAE. Each tuned on the validation
set over a small grid (Table 3); batch 32, 200 epochs.

**Sampling and split (Sec. 2.4).** 10,000 random points per state drawn from the filtered CDL.
Per class, 300 points are drawn at random for training and validation (8:2), giving a
class-balanced 1,200 (AR) / 1,440 (CA) training set; all remaining points (8,500 AR / 8,200 CA) form
the test set, which keeps the natural, imbalanced class distribution.

**Evaluation protocol.** Random pixel-level hold-out within the same state and year. Train,
validation, and test points are drawn from one pool of random points over the same area, with no
field identifiers and no spatial blocking, so the headline "testing" accuracies are in-region,
in-distribution scores; they sit at the pooled-random end of our spectrum, below even field-wise
k-fold. Three secondary evaluations reach further. (i) **Mapping areas** (Sec. 3.3, Table 7): two
sub-regions per state "selected for their considerable geographical separation," scored as the
fraction of pixels agreeing with CDL. The paper does not say whether these areas were excluded from
the sampling pool, so this is at best a weak within-state spatial check against a model-derived
reference. (ii) **Cross-year** (Sec. 3.4, Fig. 11): models trained on 2021 tested on 5,000 new
points per state from 2020 and 2022, same areas. This is the only genuine distribution-shift test.
(iii) **Missingness experiments** (Sec. 4.2): stratifying test points by missing rate, and deleting
observations from fully observed corn pixels at test time. No cross-region (train one state, test the
other) experiment is run. Single training run per model; no seeds, confidence intervals, or
significance tests.

**Metrics.** Overall accuracy, Cohen's kappa, and macro-F1 (they note macro-F1 matters for
imbalanced data), plus normalized confusion matrices, parameter count, training and inference
time, and memory.

## Key Findings

**Headline accuracy (Table 5, in-region random hold-out).**

| Model | AR OA | AR macro-F1 | CA OA | CA macro-F1 |
|---|---|---|---|---|
| MCTNet | 0.968 | 0.933 | 0.852 | 0.829 |
| GL-TAE | 0.952 | 0.905 | 0.843 | 0.820 |
| Transformer | 0.956 | 0.909 | 0.813 | 0.789 |
| ResNet-18 | 0.945 | 0.894 | 0.837 | 0.821 |
| L-TAE | 0.941 | 0.884 | 0.838 | 0.805 |
| RF | 0.936 | 0.887 | 0.823 | 0.793 |
| TAE | 0.911 | 0.854 | 0.787 | 0.767 |
| SVM | 0.910 | 0.846 | 0.786 | 0.762 |
| LSTM | 0.907 | 0.843 | 0.799 | 0.766 |

MCTNet leads both states, but the margins over the next model are 0.009 to 0.028 macro-F1 from a
single run. In Arkansas RF (0.887) ties L-TAE (0.884) and beats TAE, SVM, and LSTM; in California RF
(0.793) sits between L-TAE and the Transformer. The authors attribute the LSTM, SVM, and TAE deficits
partly to sensitivity to missing data and noise.

**Ablation (Table 4).** Removing the Transformer branch costs most (AR macro-F1 0.933 to 0.860),
then the CNN branch (0.876), then `ALPE` (0.888, replaced by sinusoidal encoding), then the mask
alone (0.895). In California the same order holds with smaller gaps (0.829 vs 0.784-0.822).

**Efficiency (Table 6).** Fewest parameters (55 k) and lowest memory (0.74 MB), but training is not
faster than L-TAE or GL-TAE (75 s vs 51-54 s in Arkansas) because the stages are sequential.

**Mapping areas (Table 7).** Agreement with CDL is 0.85-0.90 for every model, and the ranking
changes from the test-set ranking: L-TAE beats MCTNet in California area 2 (0.892 vs 0.873), and RF
beats GL-TAE and L-TAE in California area 1 (0.895 vs 0.887 and 0.886), within 0.002 of MCTNet. The
authors note the discrepancy and attribute it to the random CDL samples not representing the map
areas; "a model's excellent performance on the testing data only suggests a higher likelihood of
superior performance in actual mapping."

**Cross-year (Fig. 11).** MCTNet is best in most year-state combinations, but RF is best in
California 2022, and RF is stable across years in California while weak in Arkansas. Exact values
are in the figure only.

**Missing-rate stratification (Fig. 15, Arkansas).** Fewer than 20 % of test samples have under
10 % missing steps, so almost every series is gappy. On cumulative subsets with missing rate below
10 % through below 40 %, MCTNet OA falls only from 0.973 to 0.968, while MCTNet_noALPE falls from
0.963 to 0.950.

**Interpolation and smoothing (Fig. 15).** Retraining on linearly interpolated, Savitzky-Golay
smoothed series lowers MCTNet's accuracy at every missing rate (MCTNet1 below MCTNet) but raises the
no-`ALPE` variant's (MCTNet_noALPE1 above MCTNet_noALPE), especially at high missingness. So the
best fill strategy depends on whether the model has an explicit missingness signal.

**Synthetic test-time deletion (Fig. 16).** On 1,361 Arkansas corn pixels fully observed in the
growing season (DOY 120-270), observations are deleted at increasing rates, either at random or as
one contiguous block, and both models re-scored without retraining. OA falls with missing rate for
both, contiguous gaps hurt more than scattered ones ("prolonged gaps cause the model to miss key
phenological information"), and MCTNet stays above MCTNet_noALPE at every rate. Values are given
only graphically.

**Attention (Fig. 17).** For one corn pixel, MCTNet_noALPE's first-head attention is drawn toward
missing (zero) dates, while MCTNet's concentrates on the DOY 180-300 phenological window.

## Relevance to Our Crop-Classification Study

**Direct bearing on the missingness-shift finding.** This is the most closely matched paper in the
corpus on the mechanism behind our corrected results. Like our raw-input deep pipelines, MCTNet's
inputs are zero-filled at missing composites. The difference is that their zeros appear in the
training data at the same rate as in the test data (random split, same scene, same year), and the
model is given an explicit mask. Under that consistency, zero fill barely hurts: OA moves 0.005
across missing-rate strata. Our holdout's September gap is the opposite case: 0 % missing in
training, 40.6 % of holdout pixels missing, zeros the network had never seen. Their result therefore
supports our reading that zero fill itself is not the problem; the train/test mismatch is. It is a
useful citation for that sentence.

**Closest precedent for our test-time gap experiment.** Fig. 16 deletes observations from fully
observed pixels at inference without retraining, which is a controlled train/test missingness
mismatch, and finds contiguous gaps hurt more than scattered ones. Our September gap is a contiguous
one-step gap of exactly this kind. The novelty report states that no SITS paper measures a gate or
sparse model under a zero-filled time step; that remains true (MCTNet has no sparse gate), but this
paper does measure an attention model under inference-time deletion, so the novelty sentence should
cite it and be narrowed to "no paper shows such a mismatch reordering a multi-model benchmark."
Note also that their training data already contain natural gaps, so their deletion test is a
milder mismatch than ours.

**A third remedy for the todo-list control.** Our planned control compares the sparse gate with
training-time month dropout. MCTNet represents a third standard option: feed a missingness mask and
mask the positional encoding. A referee may ask why we did not compare against mask-as-input. At
minimum, the Discussion should name the three families (consistent imputation, training-time
dropout or augmentation, explicit missingness indicators) and place the gate among them; if compute
allows, a mask-channel variant of L-TAE is a cheap fourth arm of the 2 x 2.

**Fill strategy interacts with architecture.** Their interpolation result (helps the no-mask model,
hurts the masked one) parallels our finding that pixel TabNet's apparent robustness came from its
median fill rather than from sparsemax: the fill rule and the model cannot be evaluated
separately. This supports reporting every model under one common fill, as the corrected Table 1 does.

**Benchmark ordering.** In-region, dense temporal attention and CNN models lead and RF is
mid-table, consistent with our corrected (consistent-fill) ordering. The two places where the ranking
moves, the mapping areas and California 2022, are the two places closest to a shift, and in both
RF gains ground. That is weak, single-run evidence, but it is in line with the transfer-ranking
literature we cite (BreizhCrops, Mohammadi 2024) in showing small, unstable gaps between RF and deep
models once the evaluation leaves the training pool.

**Feature and sensor design.** Optical-only Sentinel-2, raw ten-day composites, pixel level, no
engineered features: the same input family as our raw-input deep models. Ten bands including red
edge, against our six (`B2`, `B6`, `B11`, `B12`, `EVI`, hue). Ten-day compositing over a full year
(36 steps) versus our monthly series with May and June excluded.

**Lightweight claim.** MCTNet at 55 k parameters is smaller than L-TAE; it is a relevant comparator
for any "lightweight" framing of L-TAE-S, and it shares our observation that parameter count and
training time do not move together.

## Evaluation Caveats

**In-region random hold-out, reported as test accuracy.** All headline numbers come from random
points split within one state and year. Points from the same field, or adjacent fields, can fall in
both train and test, so the 0.968 OA and 0.933 macro-F1 measure in-distribution fit, not transfer.
They are not comparable to our holdout-tile macro-F1.

**Spatial leakage is not addressed.** No field IDs, no spatial blocking, no buffer between train and
test points. With CDL-derived labels, nearby pixels share both spectra and label errors.

**Model-derived labels.** CDL is itself a classifier output; filtering at 95 % confidence keeps the
easiest pixels, which inflates accuracy and biases toward pure, interior pixels. Map "accuracy" in
Table 7 is agreement with CDL, not with ground truth.

**Class imbalance.** Training is balanced (240 per class) while test sets keep natural frequencies
(soybeans 4,377 vs cotton 462 in Arkansas), so OA is dominated by majority crops. Macro-F1 and kappa
are reported, which helps; per-class F1 is shown only through confusion matrices.

**No uncertainty.** One run per model, no seeds, no intervals. Most between-model gaps (0.01-0.03
macro-F1) are of the size our paired bootstraps found indistinguishable from seed noise.

**Missingness experiments are partial.** Fig. 15 strata are cumulative subsets of the same test set,
so they are highly correlated and do not isolate high-missingness samples (the over-40 % group is
not shown). Fig. 16 uses one class (corn), one state, and only two models; values are not
tabulated. Neither experiment compares against the other baselines, so we cannot tell whether
MCTNet's robustness exceeds that of L-TAE or RF under the same deletions.

**Tuning asymmetry.** MCTNet's grid includes architecture depth and kernel size; RF and SVM grids are
small; GL-TAE and L-TAE are tuned on width and heads only.

**What the paper does not measure.** Cross-region transfer (despite two states, no train-one,
test-other run); field-level aggregation; training-time missingness augmentation as an alternative
to `ALPE`; behaviour when the training data have no gaps but the test data do (our case); and
statistical significance. The authors list cross-region generalization as future work.
