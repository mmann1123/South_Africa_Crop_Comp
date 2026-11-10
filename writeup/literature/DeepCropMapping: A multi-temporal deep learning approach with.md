# DeepCropMapping: A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping

**Citation:** Jinfan Xu, Yue Zhu, Renhai Zhong, Zhixian Lin, Jialu Xu, Hao Jiang, Jingfeng Huang,
Haifeng Li, and Tao Lin (2020). "DeepCropMapping: A multi-temporal deep learning approach with
improved spatial generalizability for dynamic corn and soybean mapping." *Remote Sensing of
Environment*, 247, 111946. DOI: `10.1016/j.rse.2020.111946` (printed on the PDF first page; verified
against Crossref on 2026-11-09: title, journal, volume 247, article number 111946 and the nine-author
list Xu, Zhu, Zhong, Lin, Xu, Jiang, Huang, Li, Lin all match). Haifeng Li and Tao Lin are the
corresponding authors. Code: [Lab-IDEAS/DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping).
Sources: the full text in this folder (read in full, 16 pages plus references). The Supplement
(Tables S1 to S3, Figs. S1 to S3) was not available, so statements about it come from the main text.

## Objectives

The paper asks whether an LSTM-based multi-temporal classifier generalizes spatially, which it
frames as a scientific challenge "due to the spatial heterogeneity of meteorological conditions,
hybrid selection, topographical features, and farming practice". It proposes `DeepCropMapping`
(DCM), a two-layer bidirectional LSTM with an attention layer over time steps (`AtBiLSTM`), and
poses three research questions: (RQ1) end-of-season spatial generalization against Transformer,
random forest (RF) and MLP; (RQ2) in-season (early) mapping; (RQ3) what the model learns as the
season progresses, via classification confidence and t-SNE of learned features.

The task is three-class (corn, soybean, other) at 30 m in the US corn belt. Labels come from the
USDA Cropland Data Layer (CDL), which the authors treat as ground truth (95 percent or better
accuracy for corn and soybean).

## Methods

### Sites, years and inputs

- **Six sites**, each 51 km by 51 km (1700 by 1700 pixels, 2.89 million pixels), in Iowa, Minnesota,
  South Dakota, Nebraska, Illinois and Indiana (Fig. 2), each inside one Landsat ARD tile. Sites A to
  D are mostly cropland. Sites E and F are mainly non-cropland (other = 87 and 53 to 59 percent,
  Table 1), so class balance differs sharply between sites.
- **Sensor:** USGS Landsat Analysis Ready Data (ARD) surface reflectance, Landsat 7 ETM+ and
  Landsat 8 OLI combined, six optical bands (blue, green, red, NIR, SWIR1, SWIR2), 22 April to
  23 September. Optical only, no SAR, no indices in the main model (EVI and NDVI are suggested as
  future work).
- **Validity and gap handling (p. 5):** "We discarded any pixel observations labeled cloudy,
  shadow, filled or unclear. Any pixel with less than seven valid observations after May 15 was
  considered to lack sufficient multi-temporal information and excluded from the dataset." Gaps
  are then filled by "linear interpolation based on the nearest valid values before and after the
  target time step for each pixel", giving 23 weekly steps. The model has no mask input, so it
  cannot tell a measured value from an interpolated one.
- **Per-domain statistics:** Table 1 (p. 5) lists, for every site and year, the number of valid
  pixels and a "Validity percentage" (the share of the 2.89 million pixels that survive the
  seven-observation rule), next to class proportions. Fig. 8 (p. 12) prints the percentage excluded
  for the 2018 test maps. A threshold-sensitivity analysis is in Supplement Table S3.

### Models

DCM (4.5 M parameters on average), a Transformer encoder (18.9 M), RF and MLP (both on the
flattened `S x T` vector, 138 features). Hyperparameters were searched per model. DCM uses
cross-entropy with L2 and 50 percent dropout, Adam.

### Experimental design

Train on one base site using 2015, 2016 and 2017 (8.6 million pixel samples for Site A), test on
2018. The "local test" applies the model to the same site in 2018. The "spatial transfer test"
applies it to the other five sites in 2018. Every site takes a turn as base ("full cross
validation"), and the transfer score for a site averages the models trained at the other five.
In-season mapping fixes Site A as base and truncates the series week by week. Metrics: Cohen's
kappa (primary), macro-averaged user's and producer's accuracy, per-class F1. Significance by
paired t-test on kappa across sites (Table 2).

**Evaluation protocol:** Cross-year same-site ("local test", training 2015 to 2017, test 2018)
plus a true spatially disjoint, cross-site test in the transfer year 2018, with a time-forward
split throughout. This is two to three rungs above random k-fold, and the transfer test sits at
the spatially disjoint-holdout rung of this manuscript's spectrum. There is no pooled pixel
k-fold, so the pixel-adjacency leakage typical of the first rung is avoided. Caveats: the local
test reuses the same pixel locations in training (2015 to 2017) and test (2018), so it is
temporal but not spatial, and rotation makes training years and the target year share crop
rotation structure. Seeds and run variance are not described beyond the t-test across sites, and
the t-test treats sites as samples, which share base-site training.

## Key Findings

- **Headline scores (kappa):** DCM 85.8 percent mean in local tests and 82.0 percent mean in
  spatial transfer, against RF and MLP 83.9 percent in local tests (p. 9). Transformer is
  "comparable" locally (p = 0.285) and less stable across transfer sites, second in Sites A, D and
  F, higher than DCM in Site C, lower than RF in Sites B and E. DCM is significantly better than
  all three in transfer (Table 2, p < 0.05 to 0.01).
- **Transfer drop:** "All four models showed certain performance drop for transfer analysis as
  compared to local tests" (p. 11). The mean decrease is 11.5 percent in Site E against under 3
  percent elsewhere, and the largest DCM gain over RF in transfer is 6.5 percent (Site F).
- **Authors' explanation of the drop (p. 11):** "The magnitude of performance decrease in different
  sites could be explained by the degree of similarity of phenological cycles indicated by ARD
  time-series profiles", "the distinct phenological characteristics of Site E might account for the
  result", plus low crop proportion, scattered parcels and "sample impurity such as mixed pixels".
  Phenology desynchrony ("leads and lags") also explains the unstable in-season curves in Sites D,
  E and F (p. 13 to 14). Validity percentage is not invoked for any drop, although Table 1 shows
  Site E at 38.9 percent in 2018 (lowest) and 47.8 percent in 2017.
- **Class pattern:** all models score higher on "other" than on corn or soybean, and DCM's edge is
  larger for the crops (Fig. 9).
- **In-season:** DCM overtakes RF and MLP after about week 12 (8 July), when corn and soybean
  SWIR and NIR curves separate. Confidence rises from under 70 percent at week 2 to 91 percent at
  season end (base site) and above 84 percent (transfer sites).
- **Cost:** DCM trains in 56 min on average, Transformer in 304 min, MLP in 19 min, RF in 12 min
  (RF memory-heavy).
- **Limitation stated by the authors (p. 16):** "the requirement of a complete time series is one
  major limitation of the current workflow", the threshold "had a great impact on the proportion of
  valid pixels as shown in Table S3", and sensitivity "depended on the quality of raw remote sensing
  data for the specific site and year. Less cloud and shadow and higher revisit frequency improved
  data quality and reduced sensitivity."

## Relevance to Our Crop-Classification Study

### Same direction as our headline, with a milder drop

A sequence model with attention, trained on one site, loses about 3.8 kappa points on average
when moved to other sites in the same year (85.8 to 82.0), and the paper's own ranking (DCM over
RF and MLP) comes from the transfer test, not the local one, where the models are statistically
tied or close. That agrees with our finding that in-region comparisons understate the differences
that appear under a spatial holdout. It does not agree with our tree-model result: here RF is the
weakest transfer model, but RF and MLP receive a flattened 138-feature vector with no engineered
temporal features, so the comparison is between a sequence model and raw flattened inputs, not
between deep nets and well-tuned gradient boosting. The paper never tests XGBoost or LightGBM.
The transfer gap is also small because the six sites share a corn-belt climate and a single crop
calendar, with two dominant summer crops. Our Western Cape holdout is a winter-cereal system with
five confusable crops and sparse labels, and the abstract's "mean average kappa of 82.0 percent"
should not be compared with our macro-F1.

### Missingness handling and the audit

The paper is one of the few in this literature that reports a per-site, per-year validity
statistic (Table 1) and a threshold sensitivity (Table S3). The model is blind to the fill, but the
statistic exists and the authors discuss its dependence on site and year. Dropping pixels with
fewer than seven valid observations after 15 May removes cloudy pixels rather than feeding them,
so the target test set is thinner (Site E 2018: 38.9 percent retained) and the pixels that remain
are the cleanest. This is a selection effect on the test set rather than a distribution shift in
the input. In `reports/Missingness confound audit of transfer papers.md` this paper had been
ranked provisionally from code and a conference precursor. Reading the text changes that: see the
rank recommendation in that report's follow-up. Our own finding (zero-filling September in the
holdout drove most of the CNN-BiLSTM drop) is a case where the validity statistic would have
mattered, and Table 1 of this paper is a model for reporting it.

### Feature and sensor design

Optical-only raw reflectance time series at pixel granularity, with weekly interpolation, as
opposed to our xr_fresh-style statistical features at field level and our month-level gaps. Weekly
interpolation to a dense grid is the opposite of our approach of leaving months missing and
filling from training-tile medians, and the paper gives a direct test of neither.

### What the paper does not measure

No missingness ablation (it never drops observations at test time or varies the seven-observation
threshold in the main text), no per-class transfer scores beyond Fig. 9, no minority-class
analysis beyond three coarse classes, no comparison with gradient-boosted trees, no per-run
variance, no cross-region transfer beyond one agroclimatic zone, and no spatial split inside a
site (training pixels are reused in the local test).

## Evaluation Caveats

- **Local test is not spatially held out.** Pixels from 2015 to 2017 train the model and the same
  pixels in 2018 test it, so only the transfer experiment measures spatial generalization.
- **Reference labels are CDL**, itself a satellite classification, so reported kappa measures
  agreement with CDL, not with field truth.
- **Class balance varies by site** (other = 17.8 percent in Site A, 85 to 87 percent in Site E), and
  kappa on a three-class problem with a dominant "other" class can mask weak soybean or corn
  recall. Per-class F1 is shown in Fig. 9 only as bars.
- **Excluded pixels are not scored.** A site's kappa is computed on pixels that pass the
  seven-observation rule only, so sites with low validity (Site E, 2017 and 2018) are scored on a
  filtered subset, and cross-site kappa values are not computed on equal footing.
- **Statistics across sites are not independent.** The paired t-test treats the five or six sites
  as samples while all models share the same base-site training years.
- **Supplement not read.** Tables S1 to S3 were unavailable, so hyperparameters and the threshold
  sensitivity are not checked here.
