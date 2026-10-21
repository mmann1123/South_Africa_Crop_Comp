# Selecting Relevant Features for Random Forest-Based Crop Type Classifications by Spatial Assessments of Backward Feature Reduction

**Citation:** Talha Mahmood, Muhammad Usman, and Christopher Conrad (2025). *PFG - Journal of Photogrammetry, Remote Sensing and Geoinformation Science*, 93(2), 173-196. DOI: 10.1007/s41064-024-00329-4 (verified against Crossref: title, all three authors, journal, volume 93, issue 2, pages 173-196 and publication date 9 January 2025 match).
**Data:** Sentinel-1 and Sentinel-2 are public; field polygons available from the authors on request.

## Objectives

The paper studies sequential backward feature elimination for a random forest (RF) crop-type classifier, with the novelty being a spatial rather than purely tabular assessment of what feature reduction does to the map.
Stated goals:

1. Use RF Gini importance to rank 122 Sentinel-1 and Sentinel-2 time-series features and eliminate them one at a time.
2. Use McNemar's test between the all-feature reference map and each reduced-feature map to find a "critical point" beyond which the reduced map differs significantly, rather than relying on overall accuracy alone.
3. Monitor two spatial metrics during reduction, spatial confidence (fraction of successive maps agreeing with the reference at a pixel) and spatial instability (number of distinct classes a pixel takes during reduction).
4. Repeat the whole process over ten stratified sampling splits to test whether the selected features depend on the training sample.
5. Describe which temporal windows and sensors the retained features come from, and how reduction affects three cropping zones of differing heterogeneity.

The question that matters for us is narrow: does removing features improve, preserve or degrade RF accuracy, and does the paper say anything about feature selection helping transfer or small samples.

## Methods

**Study area.** Lower Chenab Canal (LCC) command area, Punjab, Pakistan, about 12,400 km2 of irrigated cropland in Rechna Doab, arid to semi-arid.
Kharif (summer) season 2017.
Five crop classes: rice, cotton, sugarcane, maize, fodder; plus orchard/forest, urban/bare and water as non-crop classes for a first-stage cropland mask.
Three cropping zones defined from prior MODIS work: continuous homogeneous zone (CHZ, rice-dominated), continuous mixed cropping zone (CMZ, all crops, sugarcane dominant, small patchy fields), discontinuous mixed zone (DMZ, sparse cropland among forest and bare land).

**Reference data.** 894 crop polygons collected by GPS in October 2017 in proportion to regional crop shares:

| Crop | Polygons | Share | Training | Validation |
|---|---|---|---|---|
| Rice | 274 | 30.6% | 192 | 82 |
| Cotton | 84 | 9.4% | 59 | 25 |
| Sugarcane | 287 | 32.1% | 201 | 86 |
| Maize | 135 | 15.1% | 95 | 41 |
| Fodder | 114 | 12.8% | 80 | 34 |

Plus 134 orchard/forest, 135 urban/bare and 115 water polygons from Google Earth.
A stratified 70/30 polygon split is made first; then one to six random pixel points are drawn inside each polygon according to its size (1-2 pixels: 1 point; more than 20 pixels: 6 points), giving 2,250 crop training plus validation points.
Points from a training polygon stay in training and vice versa, so the split is polygon-disjoint.
The authors state they chose random points inside polygons rather than polygon means because polygon-mean features "can increase the confusion among crops with similar spectral behavior".

**Features (122 total).**
Sentinel-2: six cloud-free dates (6 May, 26 May, 15 June, 9 August, 18 September, 8 October 2017), bands 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12 plus NDVI, all at 20 m, giving 66 optical features.
No Sentinel-2 in July because of monsoon cloud.
Sentinel-1: fourteen 12-day dates from 6 May to 21 October, VV and VH backscatter in dB with incidence-angle normalisation, plus GLCM mean texture (9 x 9 window) for each polarisation, giving 56 SAR features.
Raw per-date values; no temporal statistics, no indices beyond NDVI, no engineered phenology metrics.

**Classifier and elimination.** RF via the R package `ranger`, 500 trees, mtry = sqrt(number of remaining features), re-set at every step.
Gini importance ranks features; the least important is dropped; RF is retrained; repeat until one feature remains.
Hierarchical: cropland is first separated from non-crop with all features, then crop types are classified within the cropland mask.
Reference map = classification with all 122 features.
Ten different stratified random sampling splits, each repeated ten times through the elimination, yielding 100 elimination rankings per feature.
Kruskal-Wallis tests whether rankings differ across splits.

**Metrics.** Overall accuracy, kappa, per-class F-score (from user's and producer's accuracy), McNemar's chi-square between reference and reduced maps (significant at 3.84), spatial confidence and spatial instability averaged over validation pixels, all also computed per cropping zone.
Critical point = first step at which two consecutive McNemar values exceed 3.84.

**Evaluation protocol:** A stratified random 70/30 split of field polygons within one irrigation district and one season, polygon-disjoint, with random pixel samples drawn inside polygons.
On our spectrum this is field-wise random split within one scene.
The authors explicitly draw a sparse subset of points "to reduce spatial autocorrelation", but training and validation polygons remain interleaved across the same canal commands, so neighbouring fields of the same crop are on both sides.
There is no spatially disjoint holdout, no second year and no second region.
The paper closes with the statement that "the spatial and temporal transfer of the classification model remains an open task", so it does not claim transfer results.
Ten sampling splits give a credible estimate of split-to-split variance for the in-region numbers, which most papers in this corpus lack.

## Key Findings

**Reference accuracy and its stability.** With all 122 features, OA is about 0.81 and kappa 0.75 (repetition 6, chosen as the reference because it is closest to the mean of ten repetitions).
Across ten repetitions, OA and kappa vary only slightly; the maximum standard deviations over all reduction steps are 2.1% (OA) and 2.9% (kappa).
McNemar's tests between the ten reference maps find only four of 45 pairs significantly different.

**Accuracy during reduction (Figure 5, Table 5).** OA ranges from 0.82 (all features) down to 0.42 (one feature).
OA is essentially flat from 122 features down to the critical point and then falls.
Critical points occur after 76 to 98 features have been removed in nine of ten repetitions, i.e. 24 to 46 features retained; repetition 5 is an outlier with an early double-significant step at 7 removed and otherwise behaves like the others.

| Repetition | Features removed at critical point | Features retained |
|---|---|---|
| 1 | 96 | 26 |
| 2 | 80 | 42 |
| 3 | 96 | 26 |
| 4 | 76 | 46 |
| 5 | 7 | 115 |
| 6 | 98 | 24 |
| 7 | 95 | 27 |
| 8 | 90 | 32 |
| 9 | 76 | 46 |
| 10 | 82 | 40 |

**Reference versus reduced map (Table 7, repetition 6, 24 features retained):**

| Metric | All 122 features | 24 selected features |
|---|---|---|
| Overall accuracy | 0.81 | 0.79 |
| Kappa | 0.75 | 0.71 |
| F-score rice | 0.93 | 0.92 |
| F-score cotton | 0.82 | 0.75 |
| F-score sugarcane | 0.85 | 0.83 |
| F-score maize | 0.64 | 0.62 |
| F-score fodder | 0.51 | 0.46 |
| Spatial confidence | 1.00 | 0.95 |
| Spatial instability | 1.00 | 1.28 |

The unweighted mean of the five crop F-scores (a macro-F1 the paper does not compute) is 0.75 with all features and 0.72 with 24 features.
The 2-point OA loss is not statistically significant by McNemar's test, but kappa and mean F-score drop about 4 points, cotton drops 7 and fodder 5.
The authors are explicit that "in contrast to previous studies (e.g. Löw et al. 2015b; Cui et al. 2020), reduced number of features did not significantly exceed the classification accuracy when using all features", and cite Gilbertson and van Niekerk (2017) for the view that feature reduction need not raise accuracy.
The benefit claimed is computational, plus a diagnostic of where the map becomes unstable.

**Which features survive.** 18 optical and 6 SAR features.
Optical survivors: SWIR (B11, B12), red edge (B05), red (B04) and NDVI, spread over May, June, August, September and October; four of the six NDVI dates are retained.
The two most important features overall are B11 on 9 August and 18 September.
All six SAR survivors are VH polarisation (four of them GLCM mean texture), five of them from July, when no Sentinel-2 scene exists.
SAR features are on average eliminated earlier (mean modal rank 47 versus 74 for optical) and with more variability.
Ten of the 22 dated survivors are from July and early August, the full-development phase.

**Sample dependence.** Kruskal-Wallis finds no significant difference in feature rankings across the ten sampling splits (p > 0.05).
Features retained after the critical point have small rank spread across repetitions (20 of 24 within 3 positions); features eliminated before it vary more.
Five highly correlated features show rank spreads of 20 or more.

**Zone effects (Figure 10).** At the critical point OA is 0.84 in the homogeneous rice zone (CHZ), 0.82 in the sparse DMZ and 0.76 in the mixed CMZ; the CMZ never reaches 0.80 at any step.
Spatial instability at the critical point: CHZ 1.21, DMZ 1.26, CMZ 1.37; spatial confidence 0.96, 0.95, 0.94.
Fodder in the DMZ has the lowest confidence (0.85) and highest instability (1.8 classes per pixel).
Rice is the most stable class everywhere.

**What the paper does not measure.** Any classifier other than RF; any transfer across space or time; any variation of training-set size; any engineered temporal features; any test of whether the 24 features generalise to another season or district; any deep model.
It also does not report the reference-map confusion matrix as a table.

## Relevance to Our Crop-Classification Study

Our revised findings: (1) with a missing month imputed consistently, dense deep temporal pixel models lead the adjacent-tile out-of-region benchmark (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63 macro-F1), TabNet 0.62, XGBoost/LightGBM/logistic regression/MiniRocket 0.59 to 0.61, with no in-region to out-of-region ranking inversion; (2) the sparsemax channel gate gives no transfer gain on complete inputs and no consistent low-data gain, only robustness to a zero-filled month; (3) learning curves on the corrected holdout show every model near 0.55 macro-F1 at 10% of fields (about 270 fields, about 30 per class) except XGBoost at 0.51, deep models pulling ahead at 15 to 25%, and at 100% deep models at about 0.65 versus logistic regression 0.61, XGBoost 0.61 and TabNet 0.57, with no low-data benefit from feature selection.

**Feature selection does not raise accuracy, even in-region, even for a tree model.**
This is the paper's most useful result for us.
With a careful, ten-split, importance-ranked backward elimination, the best the authors can do is remove 80% of the features at a cost of 2 OA points, 4 kappa points and 3 macro-F points, with cotton and fodder losing 5 to 7 points.
They state directly that reduced feature sets did not exceed the all-feature accuracy.
Our sparse-gate and MiniRocket-L1 experiments reached the same conclusion from the other side: selecting or gating channels did not improve transfer on complete inputs and did not help at low data.
The two results together undercut the earlier framing in our manuscript that sparse, feature-selecting models have a transfer advantage; the only thing feature reduction reliably buys, in both studies, is compute (theirs) or robustness to a missing input block (ours).

**No transfer claim.** The authors are careful to say that spatial and temporal transfer "remains an open task" and that the accuracy "usually refers to the area and the year in which it was trained".
The paper therefore cannot be cited for the proposition that fewer features transfer better; it does not test it.

**No sample-size claim.** Training-set size is held fixed (626 training polygons, 2,250 crop points) throughout.
The paper's only data-quantity observation is indirect: fodder and maize F-scores are higher in the CMZ, where they have more ground points, than in the CHZ, which the authors read as "large training datasets are desirable for improving classifier accuracy" in mixed-cropping regions.
That is consistent with our learning curves but is not a controlled result.

**Instability as a hidden cost of selection.** Spatial instability rises from 1.0 to 1.28 classes per pixel at the critical point, and to 1.37 in the mixed zone, with OA barely moving.
This is the paper's central methodological point: OA can stay flat while the map changes.
It parallels the reason we report per-seed variance and five-seed ensembles for the deep models: a single headline score can hide decision-level churn.
Our sparse gate was also evaluated on complete inputs and on a zero-filled month, which is a form of the same question (how much does the prediction change when inputs are perturbed).

**Which features matter.** SWIR (B11, B12), red edge and NDVI dominate the retained optical set, spanning the whole season.
Our six-band selection (B2, B6, B11, B12, EVI, hue) overlaps on B11, B12 and a red-edge band, which gives some external support for that choice, with the caveat that the LCC Kharif season and the Western Cape winter season have different phenology and the paper's features are raw per-date values rather than xr_fresh statistics.

**SAR as gap filler.** The retained SAR features are almost all VH from July, the month with no optical scene.
This is an operational argument for SAR fusion in cloudy windows rather than an argument that SAR improves accuracy generally (optical features are consistently ranked higher).
It is relevant to our missing-month problem (months 05 and 06 excluded) as an alternative to imputation, which we have not tested.

**Ranking stability across splits.** The Kruskal-Wallis result that rankings do not depend on the sampling split is a reassurance that RF importance is a stable in-region signal at this sample size (about 2,250 points).
It says nothing about whether the same features are important in another region.

**Metric choice.** The paper reports OA, kappa and per-class F-score, so a macro-F can be computed (0.75 to 0.72); the authors treat the McNemar test on OA as decisive, which understates the per-class losses.
We should continue to report macro-F1 as primary.

## Evaluation Caveats

1. **In-region, single season.** Random polygon split within one irrigation district in Kharif 2017; no spatial holdout, no second year, no second region.
The authors say so explicitly.
2. **Polygon-disjoint but spatially interleaved.** Random points inside polygons reduce within-field autocorrelation, but training and validation polygons share the same canal commands and villages.
3. **Small ground truth for minor crops.** 84 cotton and 114 fodder polygons; cotton is absent from CHZ validation and maize/fodder nearly absent from DMZ validation, so zone-level F-scores for those crops are not reported.
4. **RF only.** No other classifier; conclusions about feature reduction are specific to Gini-importance backward elimination with RF.
5. **OA as the significance criterion.** McNemar's test is applied to OA; kappa and mean F drop about 4 points at the "insignificant" critical point, and cotton drops 7.
6. **Critical point varies with the split.** Retained feature count ranges from 24 to 46 across repetitions (and 115 for one outlier); the 24-feature set is from one chosen repetition.
7. **Raw per-date features, no temporal statistics.** The 122 features are dated band and backscatter values; the paper does not test engineered temporal summaries, so it cannot say whether reduction behaves differently on xr_fresh-style inputs.
8. **No training-size variation.** Sample efficiency is never tested; the only data-quantity remark is a cross-zone comparison with confounded field sizes and crop mixes.
9. **Nesting effect of backward elimination.** Once removed, a feature cannot return; correlated features (five with rank spread of 20 or more) make the elimination order partly arbitrary.
10. **Hierarchical classification.** Crop types are classified inside a cropland mask built with all features; mask errors are not propagated into the reported crop accuracies.
11. **No per-class confusion matrix for the reference map.** Only summary F-scores and zone-level plots are given, so commission versus omission errors for cotton and fodder cannot be separated.
