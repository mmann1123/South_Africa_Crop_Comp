# Crop Mapping without Labels: Investigating Temporal and Spatial Transferability of Crop Classification Models Using a 5-Year Sentinel-2 Series and Machine Learning

**Citation:** Rusňák, T., Kasanický, T., Malík, P., Mojžiš, J., Zelenka, J., Sviček, M., Abrahám,
D., & Halabuk, A. (2023). Crop Mapping without Labels: Investigating Temporal and Spatial
Transferability of Crop Classification Models Using a 5-Year Sentinel-2 Series and Machine
Learning. *Remote Sensing*, 15(13), 3414. DOI: 10.3390/rs15133414 (verified against Crossref;
title, authors, volume, issue and article number match).
**BibTeX key:** `rusnak2023cropmapping`

## Objectives

The paper asks how well pixel-level crop classifiers trained on Sentinel-2 monthly composites
generalise to an unseen year and to an unseen region in Slovakia, using a domain generalisation
approach (train on several source seasons, apply without any adaptation) rather than domain
adaptation. Four classifier families are compared under identical features so that transfer
robustness can be attributed to the algorithm: quadratic discriminant analysis (QDA), support
vector machines (SVM), shallow neural networks (NN) and random forests (RF). Secondary aims are
to identify which months and bands carry transferable information, to find the earliest month at
which acceptable within-season accuracy is reached, to test whether composite observation quality
explains year-to-year performance, and to build parcel-level confidence maps from classifier
agreement. The practical goal is nationwide crop monitoring for Slovakia using LPIS parcels as
reference.

## Methods

### Data and features

Sentinel-2 L2A bottom-of-atmosphere reflectance from Google Earth Engine, 2018 to 2022, April to
August only. Bands `B2`, `B3`, `B4`, `B8` at 10 m and `B5`, `B6`, `B7`, `B8A`, `B11`, `B12` at
20 m. Clouds, shadows and cirrus are masked with the SCL layer; a simple per-pixel median
composite is computed per calendar month (April through August), giving five monthly composites
per year, each with ten bands, so 50 features per pixel per year. Two regions: the Danubian
Lowland (`Danube`, 9,820 km2, warmest part of Slovakia, Chernozems and Fluvisols, three Sentinel-2
tiles) and the East Slovakian Lowland (`East`, more continental, heavy hydromorphic soils prone
to spring waterlogging, two tiles). Eight balanced classes: rapeseed, barley, wheat, sunflower,
maize, sugar beet, soybean, permanent grassland.

Reference labels come from LPIS parcels filtered to those with more than 90 percent of a single
declared crop. From these, 2,000 pixels per class per year are sampled at random (1,698 and 1,458
per class for 2020 and 2022 because of observational limits), giving 9,156 pixels per case,
split 50/50 into training and test for each year and class. Input is the raw monthly reflectance
stack (a short five-step sequence), pixel-level, optical only, no indices.

### Missing-data handling

Monthly medians over cloud-masked observations; the paper devotes a results section (3.1,
Fig. 4) to the number of valid observations per month per region per year, noting that spring
months and specific region-months (May 2019 in both regions, June 2020 in Danube, May 2021) have
many pixels with fewer than three observations. What the pipeline does when a pixel has zero valid
observations in a month is not stated; the sampling note that fewer pixels were available in 2020
and 2022 "due to observational limitations" suggests such pixels were dropped from the sample.
All four classifiers receive the identical 50-feature vector, so gap behaviour is shared. The
authors explicitly tested whether observation quality explains transfer performance and found no
coherent relationship (Section 4.2).

### Models

All models come from MATLAB's Classification Learner. Hyperparameters are chosen by Bayesian
optimisation with five-fold cross-validation per scenario and year (20 tasks). Typical outcomes
(Table 3): SVM mostly Gaussian kernel, box constraint mean 445; NN one to three fully connected
layers (first layer mean 154 units, second 64, third 23), tanh activation in 12 of 20 tasks,
tiny L2 regularisation; RF 32 to 500 learners (mean 360), 3 to 46 predictors per split (mean 16),
up to about 40,000 splits. QDA has no tuned hyperparameters. The NN is therefore a shallow MLP on
the flattened 50-vector, not a temporal architecture.

### Evaluation protocol

**Evaluation protocol:** Four held-out scenarios, each run for every year. Scenario 1: train on
four Danube years, test on the fifth Danube year (five folds, temporal holdout within region).
Scenario 2: the same in East. Scenario 3: train on Danube in year Y, test on East in year Y
(spatial holdout, same year, single-year training). Scenario 4: train on all five Danube years,
test on each East year separately (spatial holdout with multi-year training). The test region or
year never contributes training pixels, so the transfer numbers are free of spatial and temporal
leakage even though the within-year 50/50 split is random at pixel level (the random split only
matters for the in-year results, which are stated as "good" but "data not shown"). Metric is
overall accuracy on class-balanced test sets of 1,000 pixels per class (so OA equals balanced
accuracy here), plus per-class F1 (Fig. 12, Table S2) and Kappa for the example confusion
matrices. No in-region same-year reference score is tabulated, so the paper gives transfer
numbers only. An additional analysis adds months sequentially to measure the earliest usable
month (Figs. 15 and 16), and feature importance is computed for Scenario 1, 2022.

## Key Findings

### Temporal transfer within region (Tables 4 and 5, OA percent, mean over five held-out years)

Danube (Scenario 1): QDA 92.00, SVM 92.83, NN 92.27, RF 88.11. RF by year: 92.26 (2018), 87.61,
91.56, 86.64, 82.50 (2022, drought year). Best-model range 91.5 (2022) to 94.8 (2020).

East (Scenario 2): QDA 87.28, SVM 85.40, NN 87.14, RF 84.64. RF by year: 85.80, 89.70, 90.80,
82.80, 74.10 (2022). Best-model range 85.0 (2022) to 91.9 (2020).

RF is the worst classifier in eight of ten region-years and is "substantially worse" in the
anomalous 2022 drought year (82.5 and 74.1 against 90 to 92 and 80 to 85 for the others).

### Spatial transfer across regions, same year (Table 8, Scenario 3, Danube to East)

QDA 85.43, SVM 87.31, NN 87.02, RF 81.62 (means). Year-wise RF: 76.90 (2018), 89.29, 89.15,
80.65, 72.10 (2022). Best model per year ranges from 84.9 (2018) to 91.54 (2019); worst model
from 72.10 (RF 2022) to 88.75 (2020). Spatial transfer is less effective than temporal transfer
and shows larger year-to-year variability.

### Spatial transfer with multi-year source training (Table 9, Scenario 4)

QDA 86.98, SVM 90.18, NN 88.47, RF 85.28 (means). Using all five Danube years instead of one
improves cross-region OA in every year except 2021, by up to 5 points for 2018 and 2022 (SVM
2022: 85.29 to 88.70; RF 2018: 76.90 to 85.20). Gains concentrate in rapeseed, wheat and sugar
beet; barley, maize, sunflower and soybean do not benefit much.

### Transfer drop

Temporal to spatial (Scenario 1 mean to Scenario 3 mean): SVM minus 5.5, NN minus 5.3, QDA
minus 6.6, RF minus 6.5 OA points; with multi-year training (Scenario 4) the spatial gap shrinks
to minus 2.7 (SVM), minus 3.8 (NN), minus 5.0 (QDA), minus 2.8 (RF). Because no same-year
in-region score is given, the drop from in-region to transfer cannot be computed.

### Deep versus tree ranking

The shallow NN and the SVM (and even QDA) consistently out-transfer RF by 3 to 6 OA points, with
the gap widening to 8 to 14 points in distribution-shifted years (2022 drought, 2021 East). The
authors explicitly say RF "was less reliable in generalizing across different scenarios,
particularly when there was a significant deviation in the distribution of unseen domains", and
attribute this to RF's "sensitivity to overfitting in complex multidimensional feature spaces";
RF was the only model that improved under PCA feature reduction or when late-season drought
months were removed (Fig. 15). No in-region/out-of-region ranking inversion is reported because
in-region numbers are not shown; the authors note only that all algorithms were good in-year.

### Other findings

Rapeseed, grassland and sugar beet are stable across years and regions; wheat and barley confuse
only with each other (a merged cereals class would transfer well); soybean, maize and sunflower
confuse with each other, worst in 2022 (Danube) and 2021 (East). Feature importance (Scenario 1,
2022) ranks `B6` (May), `B8A` (August) and `B6` (July) highest; red-edge `B6`, `B7`, NIR `B8A` and
SWIR `B11` dominate, while visible bands and `B8` are unimportant; July is the most important
month, August the least, yet all months contribute. Acceptable accuracy is reached by June, with
saturation by July in a normal year. Classifier disagreement is used to build parcel-level
confidence maps.

## Relevance to Our Crop-Classification Study

This is the batch's most direct multi-classifier test of whether tree models transfer better
than neural models under identical, complete inputs, and its answer is the opposite of the
sparse-trees-transfer-best thesis: the random forest is the least transferable of four models in
both temporal and spatial holdouts, and the gap widens exactly when the test distribution shifts
most. That matches our corrected holdout ordering (deep pixel models 0.64 to 0.66, TabNet 0.62,
tree and linear 0.59 to 0.61) and the authors' mechanism (RF overfits the high-dimensional
feature space) is the same one we now attribute to the trees rather than to the deep models.

**Missingness shift and identical handling.** All models see the same monthly medians, and the
authors went further than most by tabulating composite quality per region-month-year and testing
whether it predicts transfer loss; it does not. That null result is important context for our
finding (1), because it is a case where observation-quality variation is present but is a
gradient (one to three or more observations) rather than a structural absence of a month that
never occurs in training. Their design never had a month that is present for all training pixels
and absent for 41 percent of test pixels, and their pipeline never zero-filled, so their null
result does not contradict ours; it shows that once a month is at least sparsely observed and
handled the same way everywhere, quality differences are second-order. Their drought-year result
is the mirror image of our artefact: when late-season months carried an unfamiliar signal, RF's
accuracy fell as those months were added (Fig. 15), while the NN and SVM were unaffected. Trees
were the model family most sensitive to an input shift, not the least.

**Deep versus tree ranking and inversion.** The NN here is a one-to-three-layer MLP on a 50-vector,
not a temporal network, and it still beats RF by about 4 OA points cross-year and 5 points
cross-region, with SVM and QDA doing the same. The paper cannot report an inversion because
in-region scores are not tabulated, but it states that all four were good in-year, which implies
the RF deficit emerges under transfer. For us, the citable point is that a tree model ranked last
under transfer in a 50-feature, eight-class, balanced, pixel-level Sentinel-2 setting with no
adaptation, so the tree advantage we previously reported is not a general property of the model
family.

**Transfer distance and gap magnitude.** Danube to East is roughly 300 km, same country, same
year, with different soils and a more continental climate; the spatial gap relative to the
within-region cross-year setting is about 5 to 6 OA points for the well-behaved models and the
spatial scores are 85 to 90 OA on balanced eight-class data. Our adjacent-tile gap of minus 0.06
to minus 0.14 macro-F1 is comparable in kind, with the caveat that our holdout shares a growing
season and climate more closely than their two lowlands do. Their Scenario 4 shows that adding
source years narrows the spatial gap by 2 to 3 points, a lever we do not have (single season) but
should mention as the standard mitigation.

**Sparse gate and low-data regime.** No feature-selection module and no training-size ablation.
The closest datum is that RF alone benefited from PCA dimensionality reduction, i.e. a form of
input sparsification helped the tree but not the dense models, which is loosely consistent with
our observation that the sparsemax gate gives no transfer gain on complete inputs for the deep
backbones. Training sizes (roughly 4,000 to 8,000 pixels per class) are small and fixed, so the
paper offers no learning-curve evidence for finding (3).

**Feature design overlap.** Their importance ranking (`B6`, `B7`, `B8A`, `B11`; visible bands
unimportant) overlaps with our band choice (`B6`, `B11`, `B12`, with `B2` the one visible band),
and supports the red-edge and SWIR emphasis. Their finding that all months contribute, with
mid-season most important, is consistent with a late-season month (our September) being
informative enough that its absence matters.

**What to cite it for.** (i) A four-classifier, identical-feature transfer comparison in which RF
is least robust, with the overfitting explanation. (ii) Evidence that composite observation
quality did not explain transfer loss when handled identically for all models, contrasted with
the structural missingness in our holdout. (iii) Multi-year source training as the standard
mitigation for spatial transfer, worth 2 to 5 OA points. (iv) Red-edge and SWIR band importance.

## Numbers at a Glance (for cross-paper comparison)

| Item | Value |
|---|---|
| Transfer setting | Slovakia: Danube Lowland to East Slovakian Lowland (about 300 km, same year) and held-out year within region; Sentinel-2 2018 to 2022, April to August |
| Granularity and input | Pixel-level, five monthly median composites x ten bands (50 features), eight balanced classes |
| In-region reference | Not tabulated ("good", data not shown) |
| Temporal holdout OA, Danube (mean) | QDA 92.00, SVM 92.83, NN 92.27, RF 88.11 |
| Temporal holdout OA, East (mean) | QDA 87.28, SVM 85.40, NN 87.14, RF 84.64 |
| Spatial holdout OA, single-year source (mean) | QDA 85.43, SVM 87.31, NN 87.02, RF 81.62 |
| Spatial holdout OA, five-year source (mean) | QDA 86.98, SVM 90.18, NN 88.47, RF 85.28 |
| Temporal to spatial drop (single-year source) | SVM minus 5.5, NN minus 5.3, QDA minus 6.6, RF minus 6.5 |
| Worst-case shifted years | RF 2022 Danube 82.5, East 74.1, spatial 72.1 versus 80 to 92 for the others |
| Deep versus tree direction | Shallow NN, SVM and QDA above RF by 3 to 6 points on average, 8 to 14 in shifted years |
| In-region/out-of-region inversion | Not measurable; authors state RF is least robust under distribution deviation |
| Missing data | SCL cloud mask plus monthly median, identical for all models; composite quality tabulated and found unrelated to transfer loss |
| Band importance | `B6`, `B7`, `B8A`, `B11` dominant; visible and `B8` unimportant; July most, August least |
| Learning curve / sparse gate | Not studied; RF alone improved under PCA reduction |

The numbers to carry forward are the RF deficit under both holdouts (3 to 6 OA points, larger
under shift) and the 2 to 5 point gain from multi-year source training, both under identical,
complete inputs.

## Evaluation Caveats

- No in-region, same-year reference numbers are reported ("data not shown"), so the transfer
  drop from in-region and any ranking inversion cannot be quantified from this paper.
- The within-year 50/50 split is random at pixel level, so pixels from the same parcel can sit on
  both sides; this does not affect the transfer scenarios (test year or region is fully unseen),
  but it means the unreported in-year scores would be leaky.
- Test sets are artificially balanced (1,000 pixels per class), so OA is effectively balanced
  accuracy and says nothing about performance under the real, highly imbalanced class
  proportions in the landscape. Per-class F1 is available (Fig. 12, Table S2) but not tabulated
  in the main text.
- Pixel-level evaluation on pixels drawn from parcels with more than 90 percent purity; parcel
  aggregation is applied only to produce maps, not to compute accuracies, so the reported scores
  are pixel scores on clean pixels.
- Only April to August, five monthly composites; late-season and winter phenology are excluded by
  design, which is a different temporal window from our winter-crop season.
- The "NN" is a shallow MLP from a GUI toolbox, not a sequence model, and there is no deep
  temporal architecture in the comparison; the paper cannot speak directly to CNN, LSTM or
  attention models.
- Behaviour for pixels with zero valid observations in a month is not stated; sample counts were
  reduced in 2020 and 2022 for "observational limitations", implying such pixels were discarded
  rather than imputed.
- Hyperparameters are re-optimised per scenario and year by five-fold CV on the training data,
  which is sound, but means the four models are not fixed configurations across scenarios.
- The two regions are both Slovak lowlands with a shared LPIS and management culture; the
  "spatial" shift is moderate.
- Single run per configuration; no seeds or confidence intervals, and differences of 1 to 2 OA
  points between QDA, SVM and NN should be read as ties.
- The draft text retains an editorial placeholder ("Add examples.") in Section 4.5, a minor sign
  that the discussion was not fully finalised.
