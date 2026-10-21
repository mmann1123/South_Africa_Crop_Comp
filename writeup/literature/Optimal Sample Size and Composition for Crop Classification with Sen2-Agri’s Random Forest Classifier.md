# Optimal Sample Size and Composition for Crop Classification with Sen2-Agri's Random Forest Classifier

**Citation:** Urs Schulthess, Francelino Rodrigues, Matthieu Taymans, Nicolas Bellemans, Sophie Bontemps, Ivan Ortiz-Monasterio, Bruno Gérard, and Pierre Defourny (2023). *Remote Sensing*, 15(3), 608. DOI: 10.3390/rs15030608 (verified against Crossref: title, all eight authors, journal, volume 15, issue 3, article 608 and year 2023 match).
**Data:** Schulthess and Ortiz-Monasterio (2021), "Crop types of the Yaqui Valley during the 2016-2017 winter growing season", CIMMYT Research Data Repository, https://hdl.handle.net/11529/10548637.

## Objectives

This is a practitioner-oriented sample-size study for the Sen2-Agri operational crop-classification system, which wraps a random forest (RF) over Sentinel-2 time series.
The authors had an unusually large labelled set (6,084 digitised fields in one irrigation district) and used it to answer four questions:

1. Does proportional ("Ratio") or equal ("Equal") representation of crop types in the training set give better classification?
2. What is the optimal number of training fields, overall and per crop?
3. How does accuracy evolve as the season progresses (in-season classification)?
4. How well does a binary one-crop-versus-everything-else classification work, and with how few fields?

The explicit goal is to replace Sen2-Agri's expert rule of thumb (75 to 100 samples per major crop, 20 to 30 per minor crop) with empirically derived numbers.
Questions 1 and 2 are the ones that matter for our learning-curve experiment.
The paper is RF-only and single-region, single-year; it contains no deep learning and no transfer experiment.

## Methods

**Study area and labels.** Yaqui Valley, Sonora, Mexico, winter 2016-2017, an irrigated wheat-dominated district in the Sonoran desert.
Field boundaries were hand-digitised on a 13 March 2017 Sentinel-2 image and labelled from farmer declarations to the irrigation district.
The paper gives both 6,084 (Tables 1 and 2) and 6,048 (text) as the field count; Table 2 is used here.
Mean field size 11.5 ha.
Nine classes:

| Crop | Fields | Share of fields | 20% validation | 80% training pool |
|---|---|---|---|---|
| Wheat | 4,291 | 70.5% | 858 | 3,433 |
| Maize | 366 | 6.0% | 73 | 293 |
| Vegetable | 343 | 5.6% | 69 | 274 |
| Chickpea | 338 | 5.6% | 68 | 270 |
| Orchard | 193 | 3.2% | 39 | 154 |
| Forage crop | 166 | 2.7% | 33 | 133 |
| Dry bean | 151 | 2.5% | 30 | 121 |
| Asparagus | 131 | 2.2% | 26 | 105 |
| Potato | 105 | 1.7% | 21 | 84 |
| Total | 6,084 | | 1,217 | 4,867 |

Wheat covers about 75% of validation pixels (1,059,143 of 1,395,251).

**Imagery and features.** 12 cloud-free Sentinel-2A images of tile 12RXR between 13 November 2016 and 22 April 2017.
Sen2-Agri applies MAJA atmospheric correction, masks clouds, linearly interpolates masked pixels, and resamples to a 10-day composite series.
Features per date: bands 2, 3, 4, 8 (10 m), 5, 6, 7 and 11 (resampled to 10 m), plus NDVI, NDWI and brightness.
Raw per-date reflectances and indices, not engineered temporal statistics.
Classification is pixel-level inside known field polygons; no cropland mask is generated.

**Classifier.** The Sen2-Agri RF with default settings; the authors note they could not tune hyperparameters or try other algorithms because they were using the operational toolbox.
Sen2-Agri pools training pixels from all supplied polygons without stratifying by class, so classes with more pixels get proportionally more training pixels.
Internally Sen2-Agri does its own 75/25 split; the authors instead fixed an external 20% validation set so all scenarios score against the same pixels.

**Scenarios.**
Ratio (Scenario 1): random draws of 8%, 10%, 12%, 18%, 31% and 64% of the 80% training pool, preserving class proportions.
Total fields 389, 487, 584, 876, 1,509 and 3,018.
Minority-crop counts at Ratio-0.08 range from 7 (potato) to 23 (maize); at Ratio-0.18 from 15 to 52; wheat ranges from 275 to 2,128 fields.
Equal (Scenario 2): 10, 20, ..., 80 fields per crop, i.e. 90 to 720 fields total.
In-season (Scenario 3): Ratio-0.31 rerun with imagery truncated at end of January, February, March or April.
Binary (Scenario 4): wheat-vs-rest and maize-vs-rest with 50, 100 or 200 fields per class, in-season.
Every Ratio and Equal treatment was run six times with different random seeds.

**Metrics.** Pixel-level confusion matrix on the fixed validation set: overall accuracy (OA), per-class precision, recall and F-score.
A 1-pixel inner buffer is applied to training fields only.
No Cohen's kappa, no macro-F1; per-class F-scores are plotted as Michaelis-Menten fits versus number of training fields.
Breakpoint definition: the number of fields at which the fitted F-score gains less than 2.5% per additional ten fields of that crop.

**Evaluation protocol:** A random 80/20 split of fields per crop within a single Sentinel-2 tile and a single season, with the 20% validation fields fixed across all scenarios.
On our spectrum this is field-wise (FID-disjoint) random split within one scene.
Training and validation fields are interleaved across the same 2 x 2 km irrigation blocks, so spatial autocorrelation between adjacent fields of the same crop is uncontrolled.
There is no spatially disjoint holdout, no second tile, no second year and no cross-sensor test.
All reported accuracies are therefore in-region and should be read as upper bounds on what the same RF would achieve on a different district.

## Key Findings

**Overall accuracy versus number of fields (Figure 4, Section 3.2).**
OA rises with training-set size for both schemes and reaches "about 0.9" where it levels off.
The three largest Ratio treatments (876, 1,509 and 3,018 fields) exceed the best Equal treatment (Equal-80, 720 fields).
In the 360 to 630 field range, Equal gives slightly higher OA than Ratio at matched total field count.
Ratio shows only a slight improvement above 876 fields; Equal levels off after about 540 fields (60 per crop).
Equal-10 to Equal-30 and Ratio-0.08 to Ratio-0.12 show a wide spread of OA across the six seeds, which the authors interpret as unstable classifiers.
Their stated minimum: fewer than 30 fields per crop yields "not only low but also unstable classification accuracies".

**Ratio-0.18 versus Equal-80 (Table 3).** Both give OA 0.92 but with different error structure:

| Crop | Ratio-0.18 precision | Ratio-0.18 recall | Equal-80 precision | Equal-80 recall |
|---|---|---|---|---|
| Wheat | 0.98 | 0.96 | 0.99 | 0.94 |
| Maize | 0.86 | 0.84 | 0.76 | 0.91 |
| Potato | 0.62 | 0.59 | 0.66 | 0.79 |
| Dry bean | 0.74 | 0.72 | 0.78 | 0.85 |
| Chickpea | 0.84 | 0.90 | 0.89 | 0.88 |
| Forage crop | 0.44 | 0.59 | 0.42 | 0.79 |
| Vegetable | 0.52 | 0.56 | 0.64 | 0.58 |
| Asparagus | 0.48 | 0.79 | 0.45 | 0.98 |
| Orchard | 0.77 | 0.76 | 0.79 | 0.83 |

Averaged over the eight minority crops, precision is 0.66 (Ratio) versus 0.67 (Equal) and recall 0.72 versus 0.83.
A 2-point drop in wheat recall (0.96 to 0.94) under Equal releases enough wheat pixels into other classes to account for 96% of maize's commission error and more than 60% of the commission error in vegetable, forage and chickpea.
The authors' conclusion is that the dominant crop must be classified accurately first, because its omissions become the minority crops' commission errors; proportional sampling does this, and its benefit to minority-crop precision outweighs the recall gain from equal sampling.

**Per-crop breakpoints (Figure 7, Section 3.3).**
Fitted F-score curves gain less than 2.5% per ten added fields at, on average, 40 fields (SD 15.7) under Equal and 46 fields (SD 9.1) under Ratio.
Vegetable needed more than 50 fields under both schemes; forage and wheat exceeded 50 under Ratio (the wheat fit of 51 is below the tested range and flagged as unreliable).
Ratio reached the breakpoint with fewer fields for asparagus, orchard and vegetable; Equal reached it with fewer fields for the other crops.
For a given number of fields of a crop, Ratio F-scores were mostly above Equal, except for wheat.
Equal failed to detect asparagus or dry bean at all in four low-count runs.
The recommendation is at least 40 to 45 fields per minority crop and more for the dominant crop, giving a total of about 500 fields; this is close to Congalton's pre-machine-learning rule of 50 units per class and above Sen2-Agri's 20 to 30 for minor crops.

**In-season (Figure 8).** With Ratio-0.31, wheat (F 0.97) and forage (0.55) plateau by end of February; maize rises from 0.73 (January) to 0.88 (April); orchard from 0.71 to 0.79; potato and dry bean dip in the March window after harvest.
Near-maximal accuracy is reached by mid-season, when all crops except chickpea have peaked in NDVI.

**Binary (Figure 9).** Wheat-vs-rest with 50 fields per class gives OA, precision and recall above 0.9 and F 0.91 to 0.97 for windows ending February or later.
Maize-vs-rest gives recall above 0.8 but precision only 0.2 to 0.4, because maize is spectrally close to wheat and the merged "non-maize" class is 90% wheat.

**Main error source.** Wheat-maize confusion; chickpea is the second most accurate crop because of its narrow sowing window.
Forage, vegetable, asparagus and potato remain below 0.65 precision at any training size.

**What the paper does not measure.** No deep learning or other classifier (toolbox constraint); no hyperparameter tuning; no transfer to another district, tile or year; no macro-F1 or kappa; no feature selection; no report of total training pixel counts per scenario (pixels, not fields, are what the RF sees).

## Relevance to Our Crop-Classification Study

Our revised findings: (1) with a missing month imputed consistently, dense deep temporal pixel models lead the adjacent-tile out-of-region benchmark (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63 macro-F1), TabNet 0.62, XGBoost/LightGBM/logistic regression/MiniRocket 0.59 to 0.61, with no in-region to out-of-region ranking inversion; (2) the sparsemax channel gate gives no transfer gain on complete inputs and no consistent low-data gain, only robustness to a zero-filled month; (3) learning curves on the corrected holdout show every model near 0.55 macro-F1 at 10% of fields (about 270 fields, about 30 per class) except XGBoost at 0.51, deep models pulling ahead at 15 to 25%, and at 100% deep models at about 0.65 versus logistic regression 0.61, XGBoost 0.61 and TabNet 0.57, with no low-data benefit from feature selection.

**Their 30-field instability threshold coincides with our 10% point.**
Schulthess et al. find that RF classifiers trained on fewer than 30 fields per crop are both less accurate and unstable across seeds, and that per-crop F-scores stop improving meaningfully at 40 to 45 fields.
Our 10% draw provides about 30 fields per class, i.e. exactly at their instability boundary, and it is there that XGBoost is the weakest model (0.51) and all models cluster near 0.55.
Our 15 to 25% draws (about 45 to 75 fields per class) straddle their breakpoint, and this is where our deep models begin to separate from the classical ones.
The paper therefore gives an independent, RF-based reason to expect tree models to look worst at our 10% point and to expect their curve to flatten soon after; it does not predict the continued rise of the deep models, because it did not test any.

**Their RF plateau is in-region; ours is out-of-region.**
Their OA levels off near 0.9 at about 500 fields on a same-tile validation set dominated by wheat.
Our XGBoost and logistic regression plateau at macro-F1 0.61 on an adjacent tile.
These are different quantities, and the paper's plateau cannot be used to argue that 500 fields "saturate" a tree model for transfer; it can only be cited for the in-region saturation of RF on a nine-class problem with one dominant crop.
The fact that our deep models keep improving to 100% while the classical ones flatten is a transfer-regime observation that this paper has no analogue for.

**No deep-versus-classical comparison at matched sample size.**
The paper varies sample size only for a single RF.
It cannot be cited for the claim that tree models are more sample-efficient than deep temporal nets, which was part of the earlier (now withdrawn) framing of our manuscript.

**Composition result bears on our resampling ablation.**
Their finding that proportional sampling beats class-balanced sampling, because accurate majority-class classification suppresses minority-class commission errors, is consistent with our ablation in which resampling and loss reweighting gave no gain over the natural class distribution.
It also supplies the mechanism: in a wheat-dominated (or lucerne-dominated) landscape, balancing the training set starves the dominant class of examples and the resulting omissions are redistributed as minority-class false positives.

**Majority-class masking of OA.**
OA 0.92 coexists with precision 0.44 to 0.52 for forage, vegetable and asparagus.
A majority-class-only predictor would score 0.75 OA on their validation set.
This is the same artefact that motivates our macro-F1 reporting; the paper is a clean example to cite when explaining why OA is not used.

**Feature design.**
Sen2-Agri uses raw 10-day composited reflectances plus three indices, and the authors note that the dense, cloud-free Sonoran winter series (five cloud-free images in January-February) probably contributed to the high accuracies.
This is the opposite of our situation (months 05 and 06 excluded for cloud), and it is a reminder that the value of engineered temporal statistics such as xr_fresh features is operational robustness to gaps rather than accuracy on clean series, which is consistent with our raw-versus-xr_fresh experiment.

**Feature selection and transfer.** The paper contains no feature selection and no transfer experiment, so it is silent on whether sparse models transfer better or learn faster from few samples.

## Evaluation Caveats

1. **In-region, single-tile, single-season.** All results are from a random field split inside one Sentinel-2 tile in one winter; no spatial holdout, no second year, no second district.
The authors themselves call for testing "in regions with more than one dominant crop or with frequent cloud cover".
2. **Spatial autocorrelation uncontrolled.** Training and validation fields are randomly interleaved within the same irrigation blocks; neighbouring fields of the same crop and management calendar sit on both sides of the split.
3. **Pixel-level scoring inside polygons.** Accuracy is counted per pixel within hand-drawn boundaries, so the 1.06 million wheat validation pixels dominate OA; a 1-pixel inner buffer is applied to training fields only, not validation fields.
4. **OA is the headline metric.** Per-class precision, recall and F-score are given, but no macro-F1 or kappa.
OA 0.92 masks minority precision as low as 0.42.
5. **Sample size is counted in fields, but the RF is trained on pixels.** Sen2-Agri draws training pixels from a pooled bag without stratification, so larger fields contribute more pixels; the effective pixel counts per scenario are not reported.
6. **RF only, untuned.** The operational toolbox prevented hyperparameter tuning or alternative classifiers, so the breakpoints are specific to the Sen2-Agri RF defaults.
7. **Breakpoint definition is arbitrary.** The 2.5%-per-ten-fields threshold and the Michaelis-Menten fit are the authors' choice; Equal curves for several crops have outliers the authors excluded from fitting.
8. **Six seeds only.** The spread at low counts is wide; the authors do not test whether differences between Equal and Ratio at matched counts are statistically significant.
9. **Field-count inconsistency.** The paper states 6,048 digitised fields in the text and 6,084 in the tables; the tables are internally consistent.
10. **Optical-only, dense cloud-free series.** Results depend on five cloud-free acquisitions in the two months of rapid canopy growth; transfer of the 40 to 45 field recommendation to cloudier regions is untested.
