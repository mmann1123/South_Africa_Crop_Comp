# Benchmarking for practice: Few-shot time-series crop-type classification on the EuroCropsML dataset

**Citation:** Joana Reuss, Jan Macdonald, Simon Becker, Ekaterina Gikalo, Konrad Schultka, Lorenz Richter, and Marco Körner (2026). *ISPRS Open Journal of Photogrammetry and Remote Sensing*, 19, 100117. DOI: 10.1016/j.ophoto.2026.100117 (verified against Crossref: title, authors, journal, volume 19, article 100117 and year 2026 all match).
**Code:** https://github.com/jsreu/EurocropsML-Meta-Learning and https://github.com/jsreu/EuroCropsML-SSL. Data: EuroCropsML (Reuss and Macdonald, 2024; Reuss et al., 2025b), Sentinel-2 L1C parcel time series for Estonia, Latvia and Portugal, 2021.

## Objectives

The paper is a benchmark study rather than a methods paper.
Its stated purpose is to evaluate, under "real-world conditions", how well supervised transfer learning, optimisation-based meta-learning (MAML and its variants) and self-supervised pre-training (masked-autoencoder PRESTO variants) support few-shot crop-type classification from Sentinel-2 time series, and to establish EuroCropsML as a reproducible benchmark for that question.

The specific questions are:

1. Does pre-training on a geographically related region (Latvia) help a Transformer classify crops in a target region (Estonia) when only 1 to 500 labelled parcels per class are available?
2. Does adding a geographically and climatically distant region (Portugal) to the pre-training set help or hurt?
3. Do meta-learning algorithms (MAML, FOMAML, ANIL, TIML) beat ordinary pre-train-then-fine-tune transfer learning, and at what computational cost?
4. Does task-informed meta-learning (TIML), which injects parcel coordinates, help cross-regional transfer?
5. Can self-supervised pre-training on unlabelled Sentinel-1/Sentinel-2/ERA5 data (Pixel-Sen12MS-CR-TS) substitute for labelled pre-training data?

The paper's headline claims (Section 1, findings 1 to 7) are that MAML-family algorithms slightly outperform everything else in overall accuracy and Cohen's kappa; that TIML's location features add nothing; that meta-learning costs several times more compute; that geographic transfer "poses a significant challenge for all investigated supervised algorithms"; that pre-training on distant data lowers overall accuracy but regularises against majority-class dominance; and that SSL beats training from scratch, especially on minority classes.

For our purposes the paper is primarily a source of (a) a seven-point learning curve (1, 5, 10, 20, 100, 200, 500 parcels per class) for a deep temporal model trained from scratch versus the same model with various pre-training schemes, and (b) a quantitative demonstration of how badly overall accuracy misrepresents minority-class performance in a dataset where one class holds roughly half of all parcels.

## Methods

**Study regions and data.** EuroCropsML combines EuroCrops farmer-declared parcel labels (HCAT hierarchical taxonomy, evaluated at the finest level 6) with Sentinel-2 L1C observations for 2021.
Each parcel is represented by the per-date median of all 13 bands over the parcel polygon, giving a parcel-level, cloud-filtered, irregularly sampled annual series of up to 216 time steps.
Region sizes after preprocessing (Table 2):

| Country | Parcels | Crop classes |
|---|---|---|
| Estonia (target, fine-tuning and test) | 175,906 | 127 |
| Latvia (pre-training source) | 431,143 | 103 |
| Portugal (additional pre-training source) | 99,634 | 79 |

In total 706,683 parcels and 176 distinct classes, 35 of which are shared by all three countries.
The class `pasture meadow grassland grass` dominates: 84,104 of 175,906 Estonian parcels (about 48%), 215,026 of 431,143 Latvian parcels, and the third most frequent class in Portugal.
For supervised pre-training this class is downsampled to the median class frequency; for fine-tuning and testing it is left at its empirical frequency.

**Backbone.** Every supervised experiment uses the same Transformer encoder with sinusoidal positional encoding, a maximum sequence length of 366 days and a single linear classification head.
Band B10 is dropped.
The SSL models are PRESTO (Tseng et al., 2023, pre-trained weights), PRESTO-XTS (the authors' variant with full temporal and spectral resolution, retrained from scratch) and CrossPRESTO-XTS (a cheaper cross-attention decoder).
There is no classical baseline: no random forest, no gradient-boosted trees, no logistic regression.
All comparisons are between training schemes for one deep architecture.

**Pre-training and fine-tuning splits.** Two transfer scenarios: LV to EE (pre-train on 172,993 Latvian parcels, validate on 43,249) and LV+PT to EE (239,538 / 59,885).
Fine-tuning uses the predefined Estonian split: 60% train (105,543 parcels), 20% validation (of which 1,000 fixed parcels are actually used), 20% test (35,182 parcels, all used).
Figure 9 shows the Estonian split is a spatially uniform random split over the whole country, i.e. training and test parcels are interleaved.

**Few-shot construction.** For k in {1, 5, 10, 20, 100, 200, 500}, k parcels per class are sampled from the Estonian training split.
Because many of the 127 classes have fewer than 500 parcels, larger k shots drift back toward the empirical prior.
Every configuration is run with five random seeds; tables report mean and standard deviation.
Fine-tuning runs up to 200 epochs with batch size 16 and early stopping on validation loss (patience 5).
Three learning-rate strategies (shared, separate backbone/head, frozen backbone) are tuned per algorithm.

**Meta-learning.** MAML, FOMAML, ANIL and TIML (with and without a location encoder) are trained in an n-way k-shot regime with n in {4, 10} and k in {1, 10}, tasks sampled first by NUTS region (Latvian NUTS-3, Portuguese NUTS-2) then by class.
One inner gradient step is used except for a 4-way 1-shot sweep over s in {1, 4, 10}. 100 fixed meta-validation tasks are used for model selection.

**Metrics.** Overall accuracy `a_OA` (micro-averaged over the imbalanced test set), accuracy excluding the meadow class `a_MCA` (reported as the minority-class metric), accuracy on the meadow class alone, and Cohen's kappa (Appendix B.5).
No macro-F1 and no per-class F1 are reported.
Parent-level accuracies at HCAT levels 3 and 4 are in Appendix B.2.

**Evaluation protocol:** The test set is a random 20% parcel split within Estonia, so the few-shot accuracies in Tables 4 and 6 are in-region, parcel-disjoint, random-split accuracies.
The "transfer" in the paper is between the pre-training source (Latvia, or Latvia plus Portugal) and the fine-tuning target (Estonia); the model always sees k Estonian parcels per class before being scored on Estonian parcels.
On our spectrum this is therefore (i) parcel-wise random split within one region for the score itself, combined with (ii) cross-region pre-training as the treatment.
Nothing in the paper measures zero-shot transfer to an unseen region, and no out-of-region test set exists.
The parcel-level median aggregation removes within-field pixel leakage, but neighbouring parcels of the same farm will be in both train and test, so spatial autocorrelation is not controlled.
Single year (2021), single sensor, no cross-year evaluation.

## Key Findings

**Reference ceiling.** The from-scratch Transformer trained on all 105,543 Estonian training parcels reaches `a_OA` = 0.787 +/- 0.004 on the S2 task and 0.768 +/- 0.010 on the S1+S2+ERA5 task.
These are the only full-data numbers in the paper.

**Learning curve from scratch (Table 4a, S2 task, no pre-training).** Overall accuracy by shots per class:

| Shots per class | 1 | 5 | 10 | 20 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| `a_OA` | 0.17 +/- 0.14 | 0.33 +/- 0.11 | 0.438 +/- 0.027 | 0.490 +/- 0.020 | 0.518 +/- 0.033 | 0.573 +/- 0.019 | 0.632 +/- 0.006 |
| `a_MCA` (excl. meadow) | 0.050 | 0.037 | 0.060 | 0.142 | 0.27 | 0.455 | 0.517 |
| meadow-only accuracy | 0.30 | 0.65 | 0.850 | 0.871 | 0.78 | 0.701 | 0.757 |
| Cohen's kappa (Table B.22) | 0.004 | 0.094 | 0.206 | 0.25 | 0.367 | 0.466 | 0.529 |

The gap between `a_OA` and `a_MCA` at 10 and 20 shots (0.44 vs 0.06, 0.49 vs 0.14) shows that the from-scratch model at 10 to 20 parcels per class is essentially a majority-class predictor: its meadow accuracy is 0.85 to 0.87 while it recognises almost nothing else.
Kappa at 20 shots is 0.25.
Minority-class accuracy only becomes non-trivial at 100 shots (0.27) and approaches overall accuracy at 200 to 500 shots (0.46 to 0.52).
At 500 shots per class (roughly 25,000 parcels after capping by class size) the curve is still rising and sits 0.155 below the full-data ceiling.

**Pre-training from Latvia (LV to EE).** Standard transfer learning is worse than training from scratch at every shot count in overall accuracy (0.575 vs 0.632 at 500 shots) but better on minority classes at low shots (`a_MCA` 0.15 vs 0.05 at 1 shot, 0.295 vs 0.142 at 20 shots).
The MAML family gives the best overall accuracy from 20 shots upward: MAML 0.507 at 20, 0.552 at 100, 0.611 at 200, 0.648 at 500; ANIL 0.652 at 500.
The abstract summarises this as "MAML-family algorithms show on average a 31% higher overall accuracy for a 20-shot benchmark task" relative to transfer learning (0.367).
Kappa confirms the ordering: ANIL 0.556 and MAML 0.534 at 500 shots versus 0.529 from scratch and 0.482 for transfer learning.
The gains over scratch are modest (about 0.02 OA and 0.03 kappa at 500 shots) and come with 3.8 to 5.4 hours of pre-training versus 0.73 hours for transfer learning (Table C.24, RTX 4090).

**Adding Portugal (LV+PT to EE).** Overall accuracy drops for every algorithm; the from-scratch baseline beats all LV+PT methods at 1, 5, 10 and 20 shots.
Minority-class accuracy, however, often improves (for example ANIL 0.272 at 5 shots vs 0.200 for LV only), and Table 5b shows LV+PT models shift attention toward Estonia-only classes.
The authors attribute the overall loss to climate, crop-mix, management and phenology mismatch between Portugal and the Baltics.
Finding 5 of the paper is explicit: geographic transfer is hard for every supervised scheme tested.

**Seen versus unseen classes (Table 5a, 20-shot LV to EE).** On classes present in both Latvia and Estonia, MAML reaches 0.556; on classes exclusive to Estonia, every method is near zero (0.02 from scratch, 0.15 for transfer learning, 0.055 to 0.069 for MAML/ANIL).
Pre-training only helps classes that were seen during pre-training.

**TIML.** Injecting parcel coordinates gives the best 1-shot overall accuracy (0.247) but this is driven by meadow-class prediction (0.47 meadow accuracy, 0.042 `a_MCA`).
From 100 shots upward TIML (encoder) is the worst supervised method.
The authors argue that in a region where the dominant class is geographically ubiquitous, location features become a shortcut to the majority class and amplify region-specific overfitting.
They explicitly recommend against raw coordinate features for spatially heterogeneous classes (Section 8.1.2).

**Self-supervised pre-training.** PRESTO-XTS reaches 0.622 (S2) and 0.615 (S1+S2+ERA5) at 500 shots, with `a_MCA` 0.540 on the multimodal task, which the authors note is 14 points above the from-scratch model on minority classes.
SSL models underperform meta-learning at k <= 100 but beat ordinary transfer learning in most cells.
They are also reported as robust to missing bands (the S2 task has fewer channels than pre-training).

**Compute.** Pre-training: transfer learning 0.73 h, ANIL 3.8 h, FOMAML 4.1 h, TIML 4.3 to 5.4 h, MAML 5.3 h. 500-shot fine-tuning: 3.8 to 10 minutes per run.

**What the paper does not measure.** No classical baseline of any kind; no zero-shot out-of-region test; no cross-year test; no per-class F1 or macro-F1 (only accuracy with and without the single majority class); no field-size or spatial-autocorrelation analysis; no feature selection or channel-sparsity experiment.
Sample size is varied only for the fine-tuning target, never for the pre-training source.

## Relevance to Our Crop-Classification Study

Our revised results are: (1) with a missing month imputed consistently, dense deep temporal pixel models lead the adjacent-tile out-of-region benchmark (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63 macro-F1), TabNet 0.62, and XGBoost, LightGBM, logistic regression and MiniRocket 0.59 to 0.61, with no in-region to out-of-region ranking inversion; (2) a sparsemax channel gate gives no transfer gain on complete inputs and no consistent low-data gain, its one effect being robustness to a zero-filled month; (3) on the corrected holdout, learning curves show every model at about 0.55 macro-F1 with 10% of training fields (about 270 fields, about 30 per class), except XGBoost at 0.51; deep models pull ahead from 15 to 25%; at 100% deep models reach about 0.65 while logistic regression plateaus at 0.61, XGBoost at 0.61 and TabNet at 0.57; feature selection (MiniRocket L1, the sparse gate) gave no low-data benefit.

**The learning-curve shape is consistent with ours, once metrics are aligned.** Reuss et al. train the same kind of dense temporal Transformer we use, from scratch, at 10 to 500 parcels per class.
Our 10% draw (about 30 fields per class) sits between their 20-shot and 100-shot points.
At 20 shots their from-scratch Transformer has kappa 0.25 and minority-class accuracy 0.14; at 100 shots kappa 0.37 and `a_MCA` 0.27.
Our deep models at 30 fields per class reach macro-F1 0.55, which is far better than their minority-class numbers, but the problems differ in difficulty: five winter-crop classes with field-level aggregation versus 127 fine-grained HCAT classes with 48% majority share.
The point of contact is the trend, not the level: in both studies the deep model keeps improving monotonically with labelled samples, with no plateau through the largest fraction tested (their 500 shots is still 0.155 below the full-data ceiling; our deep models are still rising at 75 to 100%).
Their curve therefore supports our observation that dense temporal nets are the models that benefit most from going from 15% to 100% of fields.

**Their data say nothing about a classical-versus-deep crossover.** Because there is no RF, XGBoost or logistic-regression arm, the paper cannot confirm or refute our finding that classical models plateau at 0.61 while deep models continue to 0.65.
It is also silent on whether a tree model would beat the Transformer at 20 shots.
Any citation of this paper must be limited to the within-deep-learning comparison.

**Transfer difficulty, but of a different kind.** Finding 5 (geographic transfer is hard for all supervised algorithms) is often read as supporting the claim that deep models overfit their training region.
That reading is not what the paper shows.
The Transformer here is always fine-tuned on target-region parcels; the question is whether source-region pre-training helps, and the answer is "a little, if the source is geographically similar, and not at all for classes the source does not contain".
Our setting is zero-shot transfer to an adjacent tile with shared crop classes, where the deep models lead.
The two results are compatible: shared classes and adjacent geography are exactly the regime where Reuss et al. also see pre-training help (EE intersect LV classes, MAML 0.556 at 20 shots).

**Majority-class masking.** The 0.85 to 0.87 meadow-class accuracy at 10 to 20 shots alongside 0.06 to 0.14 minority accuracy is a clean demonstration of why we report macro-F1 rather than overall accuracy, and why our ablation on resampling and loss weighting mattered even though it ended up giving no gain.

**Location features as a cautionary tale for inductive-bias arguments.** TIML's coordinate injection, intended to help cross-regional adaptation, instead became a majority-class shortcut and the worst performer at 100+ shots.
This is an architectural intervention aimed at transfer that gave no transfer benefit, which is the same shape as our sparsemax-gate result (no transfer gain on complete inputs, no consistent low-data gain).
Neither paper finds that adding a transfer-motivated inductive bias to a Transformer improves generalisation; both find that the data regime and class coverage dominate.

**Feature selection and small samples.** The paper contains no feature selection, channel gating or sparsity experiment, so it cannot be cited for or against the claim that feature selection helps in the low-data regime.
The closest statement is that SSL models are "fairly robust to missing input data" because they were pre-trained with masked channels, which parallels our one positive gate result (robustness to a zero-filled month) and suggests that robustness to missing inputs is a training-regime property rather than an architecture-level transfer advantage.

**Pre-training is a lever we have not pulled.** The paper's strongest practical result is that SSL pre-training on unlabelled Sentinel data lifts minority-class accuracy by 14 points at 500 shots and beats from-scratch training on kappa at every shot count on the multimodal task.
If we ever extend the learning-curve experiment, PRESTO-style pre-training on our unlabelled tiles would be the natural next arm, and this paper gives the expected magnitude.

## Evaluation Caveats

1. **In-region parcel-random test set.** Despite the cross-regional framing, every reported score is on a random 20% of Estonian parcels interleaved with the training parcels (Figure 9).
It is not an out-of-region holdout and is not comparable to our adjacent-tile macro-F1.
2. **No spatial separation of train and test parcels.** Parcel-level medians remove within-field leakage, but adjacent parcels of the same farm and the same management regime can appear on both sides of the split.
The paper does not quantify this.
3. **Accuracy is micro-averaged on a test set where one class is 48% of parcels.** The authors mitigate this with `a_MCA` (accuracy with the meadow class removed) and kappa, but `a_MCA` is still a micro-average over the remaining 126 classes, not a macro-F1, so rare classes remain under-weighted.
4. **No classical baseline.** All methods share one Transformer backbone.
The paper cannot speak to tree ensembles, linear models or ROCKET-style transforms, which is precisely the comparison our manuscript makes.
5. **Few-shot samples are drawn from an imbalanced pool.** At k >= 100 many classes cannot supply k parcels, so the "500-shot" training set is not class-balanced and is not 127 x 500 parcels.
The effective training size per shot level is not tabulated.
6. **Five seeds, large variance at low shots.** Standard deviations at 1 and 5 shots (0.11 to 0.18) exceed most between-method differences; conclusions at k <= 10 are weak.
7. **Single year, single sensor.** No cross-year robustness is tested; the authors flag the absence of multi-year EuroCropsML data as a limitation.
8. **Hyperparameter tuning cost is unequal.** Meta-learning methods are tuned over more hyperparameters (inner/outer learning rates, steps, n-way, k-shot) than the baseline, and the authors note that this tuning burden partly offsets the marginal accuracy gain.
9. **HCAT level 6 is extremely fine-grained.** 127 classes include winter/spring splits of the same crop, which depresses every accuracy figure relative to a five-class winter-crop problem like ours; level-3 and level-4 parent accuracies (Appendix B.2) are substantially higher.
10. **Compute figures are for one hardware configuration** (16-core EPYC 9654, RTX 4090 for supervised; dual RTX 6000 Ada for SSL) and exclude hyperparameter search.
