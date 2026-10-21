# Assessing the Sentinel-2 Capabilities to Identify Abandoned Crops Using Deep Learning

**Citation:** Enrique Portalés-Julià, Manuel Campos-Taberner, Francisco Javier García-Haro, and María Amparo Gilabert (2021). *Agronomy*, 11(4), 654. DOI: 10.3390/agronomy11040654 (verified against Crossref: title, all four authors, journal, volume 11, issue 4, article 654 and year 2021 match).
**Data:** Ground truth from the Generalitat Valenciana (2018 in-situ inspections); not publicly released ("Data Availability Statement: Not applicable").

## Objectives

The paper asks whether multitemporal Sentinel-2 reflectance and derived spectral indices can separate abandoned agricultural parcels from active ones in the Valencian Community (Spain), for the purpose of automated Common Agricultural Policy (CAP) subsidy checks.
Three concrete aims:

1. Characterise the temporal behaviour of five spectral indices (NDVI, SAVI, NBR, BSI, CIre) over abandoned versus active parcels of eight crop types.
2. Quantify class separability with Jeffries-Matusita (JM) distances, for single dates and for stacked multitemporal inputs.
3. Compare classical (LDA, QDA, decision tree, k-NN, random forest) and recurrent deep (LSTM, BiLSTM) classifiers across three class configurations, and test the effect of training-set size on each.

The third aim, and in particular Section 3.4 (Effect of Dataset Training Size), is why the paper is in our corpus: it reports a classical-versus-deep crossover as training data grows.
It is a land-use rather than crop-type paper, but the classifier comparison is directly on Sentinel-2 time series.

## Methods

**Study area and labels.** 344,600 ha of the Valencian Community (tiles 30SXJ, 30SYJ), Mediterranean climate, mean parcel about 5 ha.
Ground truth was compiled by agronomists from in-situ inspections in 2018.
Two major classes (abandoned, active) each with eight sub-classes: vineyard (VI), arable land (AL), fruits-vineyard (FV), fruits (FR), olive grove (OL), citrus (CI), dried fruit (DF), vineyard-olive (VO).
Pixel counts (Table 1):

| Sub-class | Abandoned pixels | Active pixels |
|---|---|---|
| VI | 16,547 | 20,000 |
| AL | 1,619 | 20,000 |
| FV | 812 | 2,023 |
| FR | 159 | 12,707 |
| OL | 86 | 3,083 |
| CI | 1,756 | 20,000 |
| DF | 566 | 20,000 |
| VO | 188 | 1,555 |
| Total | 21,733 | 119,742 |

Several active classes are capped at exactly 20,000 pixels, indicating the authors downsampled majority classes; abandoned OL has only 86 pixels and FR 159.

**Imagery and features.** 67 cloud-free Sentinel-2 L2A scenes from 26 May 2017 to 11 March 2020 (almost three years), downloaded through Google Earth Engine.
Twelve bands (B10 excluded) plus one spectral index per experiment, giving an input tensor of n x 13 x 67.
Raw per-date reflectances, no temporal feature engineering, no SAR.
Pixel-based classification, with the authors arguing that pixel-level output is needed to detect partial abandonment.

**Classifiers.** LDA, QDA, decision tree, k-NN (k = 1, 3), random forest (10, 50, 100 trees), LSTM and BiLSTM.
The recurrent networks have an input layer of 67 steps x 13 predictors, two layers of 100 hidden units each followed by 50% dropout, a fully connected layer, softmax and output.
No hyperparameter search is described; no learning rates, epochs, optimiser or seeds are given.

**Experiments.**
Scenario 1: eight abandoned sub-classes only (which former crop was abandoned).
Scenario 2: all sixteen classes (eight active plus eight abandoned).
Scenario 3: eight active classes plus a single merged abandoned class (AB).
For each scenario and index, the pixel set is split 70% training and 30% test.
Section 3.4 then varies the training fraction from 10% to 90% for Scenario 3 with the BSI dataset, for all ten classifiers (Figure 7).

**Metrics.** Overall accuracy on the test set; confusion matrices with per-class user's and producer's accuracy for the best model in each scenario.
No F-score, no macro-F1, no kappa, no standard deviations across runs.

**Evaluation protocol:** Random 70/30 split of pixels, pooled over all parcels, within a single region and a single ground-truth campaign.
On our spectrum this is the first rung: pooled random k-fold over pixels within one scene, here a single random split rather than k-fold.
Pixels from the same parcel, which are nearly identical 67-step reflectance series, are present on both sides of the split.
There is no field-wise split, no spatially separated holdout, no second region and no cross-year test (the three years of imagery form one feature vector, not separate domains).
The reported accuracies are therefore in-scene, leakage-inflated numbers.
The training-fraction curve in Figure 7 inherits the same leakage: at 10% training, 90% of the test pixels still have same-parcel neighbours in the training set.

## Key Findings

**Separability (Table 3, Figure 3).** Single-date JM distances between most class pairs are far below 1.8 (poor), with VI, FV, DF, OL and VO heavily overlapping.
Stacking 15 dates raises most pairwise JM distances above 1.9; residual overlap remains among FV, OL, DF and VO and their abandoned counterparts (JM 1.8 to 1.9).
JM distance to the merged abandoned class keeps increasing as more dates are stacked.
This is the paper's main argument for multitemporal inputs.

**Classification accuracy (Figures 4 to 6).**
Scenario 1 (abandoned sub-classes): all classifiers except LDA and QDA exceed 89% OA; RF-100, LSTM and BiLSTM exceed 95%; best is BiLSTM with BSI at 96.4%.
NBR runs 1 to 2 points below the other indices.
Lowest producer's accuracies: DF 67.8%, FV 68.9%, AL 82.6%, due to omission into VI, the largest class.
Scenario 2 (sixteen classes): best BiLSTM with BSI at 91.8%; every other classifier is below 90%.
Scenario 3 (eight active plus merged abandoned): BiLSTM with BSI 98.2%, with CIre 97.6%; BiLSTM is about 2 points above LSTM.
Lowest producer's accuracies in Scenario 3: VO 79.6%, FV 85.6%, OL 89.8%, attributed to few training samples and spectral confusion.
BSI was the best thirteenth predictor in every scenario; red-edge CIre second.

**Training-set size (Figure 7, Section 3.4).** For Scenario 3 with BSI, training fraction from 10% to 90% of 141,475 pixels:
"when using small training datasets, RF-100 performs better than deep architectures.
However, LSTM and especially BiLSTM networks outperform the rest of algorithms when an increased number of training data is available."
The crossover is shown only graphically; no numeric OA values per fraction are given in the text, and the figure's y-axis spans 0 to 100% so the differences among the top models are visually compressed.
At 10% training the pool is still about 14,000 labelled pixels across nine classes, so "small" here means thousands of pixels, not tens of fields.
The Discussion restates it: "random forest performed even better than deep learning architectures in scenarios with limited training data, being a good alternative to DL algorithms."

**Class imbalance.** The authors note that VI commission errors in Scenario 1 stem from VI over-representation rather than spectral overlap, and that they downsampled majority classes in Scenarios 2 and 3 to correct this.
They call for more field work on minority classes.

**What the paper does not measure.** Any spatially or temporally disjoint test; per-class F-scores or a balanced summary metric; variance across runs or seeds; deep-model hyperparameters or training details; feature selection; transfer of the trained models to another region or year; the actual OA values at each training fraction.

## Relevance to Our Crop-Classification Study

Our revised findings: (1) with a missing month imputed consistently, dense deep temporal pixel models lead the adjacent-tile out-of-region benchmark (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63 macro-F1), TabNet 0.62, XGBoost/LightGBM/logistic regression/MiniRocket 0.59 to 0.61, with no in-region to out-of-region ranking inversion; (2) the sparsemax channel gate gives no transfer gain on complete inputs and no consistent low-data gain, only robustness to a zero-filled month; (3) learning curves on the corrected holdout show every model near 0.55 macro-F1 at 10% of fields (about 270 fields, about 30 per class) except XGBoost at 0.51, deep models pulling ahead at 15 to 25%, and at 100% deep models at about 0.65 versus logistic regression 0.61, XGBoost 0.61 and TabNet 0.57, with no low-data benefit from feature selection.

**Same qualitative curve shape, different regime and different mechanism.**
Portalés-Julià et al. report that RF beats LSTM/BiLSTM at small training fractions and that the recurrent nets overtake as data grows.
Our corrected learning curves also show deep models pulling ahead with more fields, so the direction of the crossover agrees.
But two things differ.
First, their x-axis is a fraction of 141k pixels within one scene, so their "small" regime (about 14k pixels) is larger than our full training set in pixel-equivalent terms, and their low-data RF advantage is observed at sample sizes where our models have already converged.
Second, their test pixels share parcels with their training pixels, so a model that memorises per-parcel reflectance series is rewarded as the training fraction grows.
BiLSTM, with 100-unit layers and no stated regularisation beyond dropout, has more capacity to memorise than RF-100, so the reported crossover is partly an artefact of leakage, not evidence of better generalisation.
Our curves are scored out of region, so the deep models' advantage at 100% cannot come from memorising test parcels.

**Our data do not show an RF-style advantage at low n.**
At our 10% point XGBoost is the worst model (0.51), and logistic regression and the deep nets sit together at about 0.55.
This is the opposite of the paper's low-data claim for tree ensembles.
The most likely reconciliation is that their low-data regime is still in the thousands of pixels and their deep models are untuned; at 30 fields per class with field-level aggregation, a boosted tree has little advantage over a regularised sequence model, and in our out-of-region setting it has a small disadvantage.
The paper should be cited as a widely repeated claim that we tested and did not reproduce under a transfer protocol, not as corroboration.

**Ranking stability.**
Within their single scene, the BiLSTM is best in every scenario, and the ordering RF-100 < LSTM < BiLSTM at full data matches our in-region and out-of-region ordering (trees below dense temporal nets).
There is no out-of-region score in the paper, so it cannot speak to ranking inversion; it is consistent with our finding that there is none.

**Feature design.**
Raw per-date reflectances for 67 dates plus a single index are fed directly to every classifier.
The RF therefore receives an 871-dimensional vector, and the authors do no feature selection.
That the RF still competes at low data is a reminder that tree models tolerate wide raw inputs, which is also what our raw-versus-xr_fresh XGBoost experiment found.
The paper has nothing on channel gating, sparsity or feature selection as an aid to transfer or to small samples.

**Index choice.** BSI (a bare-soil index) was the most useful added predictor for distinguishing abandoned from active parcels; NBR was least useful for separating abandoned sub-classes from each other.
This is specific to the abandonment task and has no direct bearing on winter-crop discrimination, though it supports our inclusion of SWIR bands (B11, B12), which BSI and NBR are built on.

**Imbalance handling.** Their observed VI commission errors from over-representation, and their remedy of downsampling majority classes, are a pixel-level version of the lucerne-dominance issue in our data.
Our ablation found no gain from resampling at the field level; their paper does not quantify the effect of their downsampling, so it adds no evidence either way.

## Evaluation Caveats

1. **Pooled pixel-level random split.** Pixels from the same parcel appear in both training and test; the 96 to 98% OA figures are inflated by within-parcel leakage and are not comparable to any field-disjoint or spatially disjoint result.
2. **No spatial holdout, no second region, no cross-year test.** The three years of imagery are stacked into one feature vector, so the model is never asked to generalise over time or space.
3. **OA only.** No F-score, macro-F1 or kappa; per-class accuracies are given only for the best model per scenario.
The abandoned class is 15% of pixels and several sub-classes have under 200 pixels.
4. **Training-size experiment reported only as a figure.** No numeric values, no standard deviations, no repeated draws; the crossover point cannot be located precisely and the y-axis compresses the top models.
5. **Deep-model training is unspecified.** No optimiser, learning rate, epoch count, early stopping or seed information; the LSTM/BiLSTM results cannot be reproduced from the paper.
6. **"Small training data" means thousands of pixels.** 10% of 141k is about 14k pixels; the low-data regime relevant to field-level crop mapping (tens of fields per class) is never reached.
7. **Majority classes capped at 20,000 pixels, minority classes not augmented.** The effective imbalance after capping is not reported, and the paper does not say whether capping was applied before or after the 70/30 split.
8. **Pixel-level classification inside known parcels with a parcel-level ground truth.** A per-parcel majority vote would likely raise all accuracies further and remove most within-parcel variance, but the paper scores raw pixels.
9. **Optical-only, cloud-free selection.** 67 cloud-free scenes over 34 months in a Mediterranean climate; gap handling is not needed and not discussed, so robustness to missing dates is untested.
10. **Land-use rather than crop-type task.** Abandoned versus active is largely a vegetation-cover and bare-soil problem; the high accuracies should not be read as evidence about discriminating phenologically similar crops such as wheat and barley.
