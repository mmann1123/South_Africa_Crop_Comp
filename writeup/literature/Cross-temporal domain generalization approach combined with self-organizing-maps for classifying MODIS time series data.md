# Cross-temporal domain generalization approach combined with self-organizing-maps for classifying MODIS time series data

**Citation:** Daniel H. Shibuya, Júlio C. D. M. Esquerdo, João P. S. Werner, André S. Tavares, and Filipe C. Felix (2025). *International Journal of Remote Sensing*, 46(15), 5802-5831. DOI: 10.1080/01431161.2025.2527988 (verified against Crossref: title, all five authors, journal, volume 46, issue 15, pages 5802-5831 and publication date 13 July 2025 match).
**Data:** MODIS MOD13Q1/MYD13Q1 from LP DAAC; TerraClass 2020 maps (www.terraclass.gov.br); Embrapa ground samples available on request. Software: the `sits` R package.

## Objectives

The paper addresses label scarcity across years rather than across regions.
It trains land-use/land-cover (LULC) classifiers on one crop year (2019/2020) of MODIS time series in the Brazilian state of Mato Grosso, using training labels extracted from an official map product (TerraClass), and then applies the frozen models to other crop years without retraining or adaptation, which the authors call cross-temporal domain generalization (DG).
Specific aims:

1. Show that TerraClass maps, cleaned with Self-Organizing Maps (SOM), can substitute for field-collected training labels at scale (40,000 samples over 903,357 km2).
2. Use SOM to characterise intra-class variability and remove inconsistent samples before training.
3. Compare random forest (RF), TempCNN and L-TAE trained on the 2019/2020 data (a) on a same-year validation split and (b) on independent ground samples from three other crop years.
4. Apply the best model to all 18 crop years (2004 to 2022) and inspect interannual LULC transitions at four locations.

For our purposes the paper is the corpus's cleanest head-to-head of RF against two dense temporal deep models (TempCNN and L-TAE, both of which we also run) under a genuine domain shift, with the important qualification that the shift is temporal within the same region, not spatial.

## Methods

**Study area and classes.** Mato Grosso, Brazil (Cerrado, Amazon and Pantanal biomes; Pantanal excluded), the country's main grain and fibre producer, with a well-established soybean then maize/cotton double-cropping system on fields of 30 to over 1,000 ha.
Crop year defined as September to August.
Five thematic classes consolidated from 16 TerraClass classes: single crop (SC), double crop (DC), semi-perennial crop (SP, mainly sugarcane), pasture (PA, herbaceous plus shrub pasture), forest vegetation (FV, primary, secondary and plantation forest).
These are land-use types, not crop types; a soybean field and a cotton field are both "single crop".

**Imagery and features.** MODIS MOD13Q1 and MYD13Q1 16-day composites, 250 m, 2004 to 2022 (18 crop years), from LP DAAC.
Six channels: BLUE, RED, NIR1, NIR2 bands plus NDVI and EVI.
Gaps flagged by the pixel-reliability layer are filled by bilinear interpolation.
Raw per-composite values as the time series; no engineered statistics, no SAR.

**Training samples.** TerraClass 2020 (crop year 2019/2020), built from 10 m Sentinel-2 with OA 0.85 (Amazon) and 0.88 (Cerrado), reprojected to 250 m, eroded by two pixels (about 500 m) to avoid edge mixing, and randomly sampled with the QGIS AcATaMa plugin.
Original sampling (OS): DC 10,000, SC 5,000, SP 5,000, PA 10,000, FV 10,000; total 40,000 pixels.
SOM (30 x 30 grid, 100 iterations, Euclidean distance, 60% conditional and posterior probability thresholds) removed 8,188 samples (20.5%), most severely from single crop (5,000 to 1,966) and semi-perennial (5,000 to 4,289), giving the SOM sampling (SS) of 31,812.
SOM also revealed two distinct SC sub-patterns (early-sown soybean/maize versus late-sown cotton) and persistent PA/SP overlap.

**Classifiers.** RF with 500 trees and mtry = sqrt(p).
TempCNN (Pelletier et al., 2019): three 1-D convolutional layers of 64 units, dense layer of 256, softmax.
L-TAE (Garnot and Landrieu, 2020): lightweight self-attention temporal encoder with positional encoding.
Deep models trained with AdamW, learning rate 0.001, weight decay 1e-6, 150 epochs, batch 64, via the `sits_tuning` defaults; no architecture-specific search is described.
SS is split 80/20 stratified (25,449 train, 6,363 validation), with five-repetition k-fold on the training set and a t-test on OA for the same-year comparison.

**Test data.** Embrapa field campaign ground samples from three other crop years:

| Class | 2016/2017 | 2020/2021 | 2021/2022 | Total |
|---|---|---|---|---|
| Double crop | 2 | 3 | 1,623 | 1,628 |
| Single crop | 74 | 27 | 975 | 1,076 |
| Semi-perennial | 162 | 567 | 176 | 905 |
| Pasture | 616 | 7 | 542 | 1,165 |
| Forest vegetation | 295 | 0 | 459 | 754 |
| Total | 1,139 | 604 | 3,775 | 5,528 |

All 5,528 samples are pooled for the DG test; each model is run five times and compared by paired t-test on OA.
Interannual classification over 18 years is assessed at only four ground points plus the RF probability layers.

**Metrics.** OA and per-class F1 on both the same-year validation split and the pooled cross-year ground samples; confusion matrices with omission errors.
No macro-F1 or kappa is reported, but per-class F1 for all five classes allows a macro-F1 to be computed.

**Evaluation protocol:** Two levels.
(a) Same-year: a stratified random 80/20 pixel split of SOM-cleaned TerraClass samples from 2019/2020, i.e. pooled random split within one domain, with the additional caveat that both sides are map-derived labels from the same product.
(b) Cross-year DG: the frozen 2019/2020 models are scored on 5,528 field-collected ground samples from 2016/2017, 2020/2021 and 2021/2022, pooled.
On our spectrum (b) is cross-year transfer within the same region, the fourth rung, and it also involves a label-source shift (map-derived training labels versus field-collected test labels) and a sampling-design shift (random map pixels versus campaign sites).
It is not cross-region: training and test samples come from the same state and the same biomes.
Spatial overlap between training pixels and test sites is not controlled or reported; some ground samples may fall on or next to training pixels, but the different years make this a temporal rather than a spatial leak.
The 18-year interannual application is qualitative (four points, visual inspection, probability layers) and is not an accuracy assessment.

## Key Findings

**Same-year validation (Table 6, Table B1).** All three models are near-perfect on held-out TerraClass pixels from the training year:

| Model | OA | F1 DC | F1 SC | F1 SP | F1 PA | F1 FV | Mean F1 |
|---|---|---|---|---|---|---|---|
| RF | 0.994 | 0.998 | 0.994 | 0.984 | 0.989 | 0.998 | 0.993 |
| TempCNN | 0.981 | 0.996 | 0.982 | 0.942 | 0.966 | 0.996 | 0.976 |
| L-TAE | 0.962 | 0.980 | 0.923 | 0.913 | 0.941 | 0.992 | 0.950 |

Differences are statistically significant (p < 0.05) but all exceed 0.96 OA.

**Cross-year domain generalization (Tables 7 and 8).** On the 5,528 pooled ground samples from other years:

| Model | OA | F1 DC | F1 SC | F1 SP | F1 PA | F1 FV | Mean F1 (computed) |
|---|---|---|---|---|---|---|---|
| RF | 0.917 | 0.972 | 0.918 | 0.900 | 0.850 | 0.922 | 0.912 |
| TempCNN | 0.886 | 0.960 | 0.914 | 0.826 | 0.801 | 0.901 | 0.880 |
| L-TAE | 0.859 | 0.944 | 0.903 | 0.753 | 0.766 | 0.898 | 0.853 |

Paired t-tests on five runs find the three models significantly different (p < 0.05).
OA drops under transfer by 7.7 points for RF, 9.5 for TempCNN and 10.3 for L-TAE.
Mean F1 drops by 8.1, 9.6 and 9.7 points respectively.
The ranking RF > TempCNN > L-TAE is identical in-domain and out-of-domain; there is no inversion.
The gap between RF and the deep models widens slightly under transfer (OA gap RF minus L-TAE: 3.2 points in-year, 5.8 points cross-year).

**Where the deep models lose.** Double crop, single crop and forest are above 0.90 F1 for all models under transfer.
The loss is concentrated in semi-perennial (sugarcane) and pasture, which the SOM analysis had already flagged as overlapping in MODIS spectral-temporal space.
SP omission errors are 10.06% (RF), 20.88% (TempCNN) and 32.04% (L-TAE); pasture is the main source of confusion for every class and every model.
The authors attribute the deep models' weaker SP/PA separation to the limited information in four MODIS bands at 250 m and to mixed pixels introduced by reprojecting a 10 m map to 250 m, and note that prior successes of TempCNN and L-TAE used Sentinel-2 or Landsat with more indices.

**SOM sample cleaning.** 20% of map-derived samples removed; the authors describe SOM as "crucial" to the DG result but do not report an ablation without SOM.
The cited precedent (Santos et al., 2021) reports a 4-point OA gain from SOM cleaning for RF on the Cerrado.

**Interannual application.** RF applied to 18 years reproduces plausible transitions (forest to pasture to single crop; pasture to sugarcane; sugarcane to double crop) at four sites, with RF class probabilities below 0.7 in transition years and in persistent SP/PA confusion at one site.
This is illustrative, not quantitative.

**Comparison to prior cross-temporal DG.** The authors' RF OA of 0.917 exceeds the 0.78 (Massey et al., 2017, CDL to MODIS, USA) and 0.82 (Luciano et al., 2022, 33-year sugarcane, São Paulo) reported for earlier RF-based cross-year generalization, while noting their own test set is unbalanced and their classes coarser.

**What the paper does not measure.** Cross-region transfer; any variation of training-set size; any feature selection or sparsity experiment; any deep-model tuning beyond `sits` defaults; an ablation of SOM cleaning; per-year accuracy (the three test years are pooled; 2020/2021 has no forest samples and 2016/2017 has two double-crop samples); kappa or macro-F1; spatial proximity of test sites to training pixels.

## Relevance to Our Crop-Classification Study

Our revised findings: (1) with a missing month imputed consistently, dense deep temporal pixel models lead the adjacent-tile out-of-region benchmark (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63 macro-F1), TabNet 0.62, XGBoost/LightGBM/logistic regression/MiniRocket 0.59 to 0.61, with no in-region to out-of-region ranking inversion; (2) the sparsemax channel gate gives no transfer gain on complete inputs and no consistent low-data gain, only robustness to a zero-filled month; (3) learning curves on the corrected holdout show every model near 0.55 macro-F1 at 10% of fields (about 270 fields, about 30 per class) except XGBoost at 0.51, deep models pulling ahead at 15 to 25%, and at 100% deep models at about 0.65 versus logistic regression 0.61, XGBoost 0.61 and TabNet 0.57, with no low-data benefit from feature selection.

**The one corpus paper where RF beats TempCNN and L-TAE under a real domain shift, and why it does not contradict us.**
Shibuya et al. find RF 0.917 > TempCNN 0.886 > L-TAE 0.859 OA (computed mean F1 0.912 > 0.880 > 0.853) on ground samples from other years.
We find TempCNN 0.65 and L-TAE 0.65 above XGBoost/LightGBM 0.59 to 0.61 macro-F1 on an adjacent tile.
Four differences explain the reversal of winner without any contradiction in mechanism.
First, the shift is different: theirs is cross-year within the same state, with the same fields and the same biomes; ours is cross-region within the same season.
Second, the inputs are different: four MODIS bands plus two indices at 250 m with 16-day compositing versus six Sentinel-2 channels at 10 m with monthly aggregation; the authors themselves say the deep models may lack the information to separate sugarcane from pasture at that resolution.
Third, the classes are different: five coarse land-use types (where three of five are nearly saturated at F1 > 0.9 for every model) versus five phenologically similar winter crops.
Fourth, the training labels are map-derived and SOM-cleaned, which removes exactly the ambiguous samples a deep model would otherwise have to fit, and the deep models are run at `sits` defaults with no architecture-level tuning.

**Both studies agree on the structural point: no in-domain to out-of-domain ranking inversion.**
In Shibuya et al. the model that is best on the same-year split is also best on other years, and the order is preserved all the way down.
In our corrected results the same holds across in-region k-fold and the adjacent tile.
This is the claim we should make, and this paper is a clean cross-year corroboration of it from a setting where the winner happens to be the tree model.
The earlier framing that in-region validation "systematically misranks" models is supported by neither study.

**Deep models lose slightly more under shift, in both studies' numbers, but the effect is small.**
Their OA drop is 7.7 (RF) versus 9.5 to 10.3 (deep); the gap widens by about 2.5 points.
That is the same order of magnitude as the standard deviations we see across seeds, and it does not change the ranking in their data or in ours.
It can be cited as weak evidence that dense temporal nets are marginally more sensitive to distribution shift, with the caveat that their deep models were untuned.

**No learning curve, no feature selection.** Training-set size is fixed at 31,812 pixels (after SOM) and never varied, so the paper offers no data points on accuracy versus number of training samples and nothing on whether deep and classical models are compared at matched small sizes.
No feature selection or gating is tested; SOM is a sample-cleaning step, not a feature-selection step, and its effect on transfer is asserted rather than ablated.
The paper therefore cannot be cited for or against feature selection helping transfer or small samples.

**Sample cleaning as an alternative to architecture.** Removing 20% of noisy map-derived labels before training is the paper's main methodological contribution.
We have not done the equivalent on our field labels, which come from the AI4FoodSecurity ground truth and are presumably cleaner, but SOM-style screening of the pixel-level training set (for mislabelled or mixed-boundary pixels inside fields) is a cheap experiment that this paper suggests could matter more than model choice.

**Gap filling.** Bilinear interpolation of cloud-flagged MODIS composites is their missing-data strategy, applied identically to train and test years.
That consistency is exactly the condition under which our deep models' earlier out-of-region deficit disappeared (the missing-month imputation artefact).
The paper does not test any inconsistency between training and test gap patterns, so it does not bear on our zero-filled-month robustness result.

**Metric.** OA on an unbalanced pooled test set (double crop 29%, pasture 21%, forest 14%) is the headline; computed mean F1 tracks OA closely here because all classes are reasonably large, but we should cite the per-class F1 for SP and PA when discussing where temporal nets fail.

## Evaluation Caveats

1. **Cross-year, not cross-region.** Training and test samples are from the same state and biomes; the result says nothing about transfer to a new region, which is our setting.
2. **Label-source shift confounds the year shift.** Training labels are TerraClass map pixels (OA 0.85 to 0.88), test labels are field campaign points; part of the 8 to 10 point drop may be label noise or sampling-design mismatch rather than temporal shift.
3. **Test years pooled and unbalanced.** 68% of test samples come from 2021/2022, 2020/2021 has no forest and 2016/2017 has two double-crop points; per-year accuracies are not reported, so year-to-year variability is unknown.
4. **Spatial proximity of test sites to training pixels not reported.** Ground sites could coincide with training pixels of the same field in a different year.
5. **Coarse land-use classes, not crop types.** Single crop pools soybean, maize and cotton; double crop pools all rotations.
Three of five classes exceed 0.90 F1 for every model, so the comparison is decided by sugarcane versus pasture.
6. **MODIS 250 m with four bands.** Mixed pixels from reprojecting a 10 m map to 250 m, and a narrow spectral set, limit what any temporal model can extract; the authors concede this.
7. **Deep models at library defaults.** TempCNN and L-TAE use `sits_tuning` defaults (150 epochs, batch 64, lr 0.001); no architecture search, no ensembling, no early stopping are described, so the deep results may be undertrained.
8. **SOM effect not ablated.** The claim that SOM cleaning was "crucial" rests on a cited prior study, not on a with/without comparison in this paper.
9. **Same-year validation is map-to-map.** The 0.96 to 0.99 OA figures compare two random draws of the same map product and should not be read as ground-truth accuracy.
10. **Interannual classification is anecdotal.** Four points over 18 years with visual inspection; no accuracy is computed for the 15 years without ground samples.
11. **OA on an unbalanced set, no kappa or macro-F1 reported.** Per-class F1 is given and allows the computed means above, but the authors' statistical tests are on OA only.
