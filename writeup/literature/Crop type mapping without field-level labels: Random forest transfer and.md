# Crop type mapping without field-level labels: Random forest transfer and unsupervised clustering techniques

**Citation:** Sherrie Wang, George Azzari, and David B. Lobell (2019). "Crop type mapping without
field-level labels: Random forest transfer and unsupervised clustering techniques." *Remote Sensing
of Environment*, 222, 303-317. DOI: `10.1016/j.rse.2018.12.026` (extracted from the PDF first page;
verified against Crossref on 2026-11-09: title, journal, volume 222 and pages 303-317 all match).

Source for this briefer: the full text PDF in this folder (main text only). The supplement (Figs.
S1-S7, Tables S1-S3, including the per-pixel clear-observation maps in Fig. S1) was not available
and is not summarised here.

## Objectives

The paper asks how to map crop types at field resolution where no local field labels exist. It tests
two routes in the US Midwest, where USDA Cropland Data Layer (CDL) labels allow a full check:
(1) train a random forest in one state or year and transfer it to other states or years, and
(2) cluster pixels without labels (k-means, Gaussian mixture model, GMM) and assign crop names to
clusters from county-level NASS planted-area statistics. The headline claim is that transfer works
when source and target are similar in growing degree days (GDD), and that GMM works when crop
diversity is low.

## Methods

### Data and features

- **Study area and years.** Nine Midwestern states (IL, IN, IA, NE, ND, SD, MN, WI, MI), 2010-2016,
  1.62 million km2. A uniform random sample of CDL cropland pixels per state and year gives 912,081
  pixels in total (126,563 for 2016).
- **Imagery.** Landsat 5, 7 and 8 Surface Reflectance Tier 1 in Google Earth Engine, 23,166 images,
  January 1 to December 31 of each year, six shared reflectance bands plus ten vegetation indices.
- **Cloud handling.** Pixels are kept only where the Landsat `QA` band (Cloud Cover Assessment
  algorithm) says "clear"; "water", "cloud shadow", "snow" and "cloud" are dropped. Harmonic fits
  then run on whatever clear observations remain at each pixel.
- **Features.** A second-order harmonic regression (n = 2, omega = 1.5) per band or index per pixel,
  five coefficients each. The fit "needed at least five cloud-free observations at a pixel." Feature
  selection (correlation, random forest importance, mutual information) reduces 13 bands and indices
  (65 features) to four (`NIR`, `SWIR1`, `SWIR2`, `GCVI`) and 20 features.
- **Classes.** Crops above 10 percent of state crop area are "major"; the rest merge into "other."
  Most states have three classes (corn, soybean, other); ND and WI have four.
- **Models.** `scikit-learn` `RandomForestClassifier`, 100 trees, defaults otherwise. Unsupervised:
  k-means and GMM (tied covariance), k set from the number of major crops, outliers beyond 2.58
  Mahalanobis units removed, features standardised per state.

### Evaluation protocol

**Evaluation protocol:** a mix of random pixel split within state and year, and genuine cross-state
and cross-year transfer. (1) Within-state baseline: random 80/20 pixel split, with no minimum
distance between train and test pixels ("we did not enforce a minimum distance ... as our sampling
coverage is quite sparse at ... 0.019%"). This is pixel-level and not field-disjoint, so it is an
in-region random split that rests on sparse sampling for its defence against leakage. (2)
Cross-state transfer: train on one state in a year, test on the other eight, repeated for each year.
(3) Cross-year transfer: train on one year in a state, test on the other six. (4) GDD-bin transfer:
pool all states, split into ten bins of 150 GDD, train on 80 percent of one bin, test on the 20
percent held out and on the other bins. Cross-state and cross-year are spatially or temporally
disjoint holdouts with the label source unchanged, so this sits near the "disjoint region and
cross-year" end of the spectrum, but only within one country, one sensor family and two main crops.
Reported metric is overall accuracy, with a majority-class baseline quoted for context.

## Key Findings

- **Within-state, within-year (random split):** accuracy above 87 percent in six of nine states
  (2016); ND 82, WI 80, MI 80. Majority-class baselines are 56, 58, 51, 58, 33, 37, 43, 51 and 38
  percent for IL, IA, IN, NE, ND, SD, WI and MI (as listed in the text).
- **Cross-state (2016):** out-of-state accuracies "near or over 70%"; ND is the exception. Models
  trained on any state except ND reach 80 percent or more on IL, IN, IA and NE; models trained
  anywhere but MN "transfer poorly to North Dakota." Values are in Fig. 4, which is a bar chart, so
  per-pair numbers are not given in the text.
- **Cross-year (train 2016):** "generally over 70%," ND again the exception, read by the authors as
  "high year-to-year variability."
- **GDD:** transfer accuracy "correlates significantly with the difference in GDD"; it falls
  "steeply" beyond about 300 GDD of difference. Abstract: above 80 percent when GDD is similar.
  Resampling classes to equal proportions improves transfer but accuracy still falls with GDD.
- **Unsupervised:** GMM reaches 85 to 90 percent in IL, IA, IN and NE, and falls to chance level in
  ND and MI. Clustering accuracy against crop entropy has R2 = 0.602 over 180 GDD zones. The
  assignment step recovers the best mapping 90 percent of the time once clustering accuracy exceeds
  65.5 percent.
- **Harmonic versus percentile features:** harmonic coefficients beat percentile features in the
  authors' development tests (detail only in the supplement).

## Relevance to Our Crop-Classification Study

Wang et al. is an early, clear case of transfer degradation measured across space and time with a
single feature pipeline, which supports our premise that in-region accuracy does not predict
holdout accuracy. Several points bear directly on our design. First, they find that transfer loss is
organised by phenological and compositional similarity (GDD, class proportions), which is the same
family of explanation our Western Cape holdout invites. Second, their harmonic coefficients are a
compact, engineered time-series representation of the kind `xr_fresh` provides, and a random forest
on them is a sparse, tree-based model, consistent with our finding that tree-like models transfer
best. Third, and unusually, they connect a transfer drop to acquisition density: 2012 has a mean of
12 clear observations per pixel against 25 overall (Landsat 5 ended in November 2011, Landsat 8
launched in 2013), and they attribute lower 2012 accuracy partly to "fewer Landsat images and
therefore worse quality features." That is a rare explicit link between observation count and
performance, though it is asserted, not isolated by an ablation. Fourth, the class-composition
argument (a forest trained on 50/40/10 corn/soybean/other does worse on a 10/30/60 target) is a
direct parallel to our lucerne-dominated training regions.

For the missingness-confound audit specifically: the gap handling (QA-clear masking, harmonic fit
with a five-observation minimum, no fill value) is applied identically to source and target and
changes no input value to a constant. The paper reports pixel clear-observation counts per year
(Fig. S1, and mean and median counts in the text), so it does give a per-domain statistic.

## Evaluation Caveats

- **Pixel-level random split for the baseline.** The within-state "upper bound" splits pixels, not
  fields, with sampling sparsity as the only leakage argument. Treat it as an in-region random
  split.
- **CDL labels are model output, not ground truth.** Evaluation inherits CDL error (reported
  producer and user accuracies 86 to 99 percent), and CDL is itself built from Landsat, so
  labels and features share a sensor. The authors note CDL accuracies are likely an upper limit for
  places without ground samples.
- **Accuracy only.** No macro-F1, Kappa, or per-class recall is reported for transfer; with two
  dominant crops plus "other," overall accuracy hides minority-class behaviour. Minor crops
  (below 10 percent) are excluded from evaluation.
- **Distance and time are confounded with composition.** GDD tracks climate and crop mix together.
  The class-resampling control narrows but does not close the gap, so how much loss is
  phenology versus composition is only partly separated.
- **Acquisition density is not isolated.** The 2012 explanation (fewer images) and the sowing-date
  explanation (early sowing in IL and IA) are offered together and not separated. No experiment
  subsamples clear observations to test the effect.
- **No field-level units.** Despite the title, all evaluation is at the Landsat pixel; there is no
  field aggregation.
- **Not measured:** other sensors or Sentinel-2 (Sentinel-2 is mentioned only as future work),
  smallholder or Global-South data, winter-cereal systems, computational cost, and cross-country
  transfer. The Midwest is described by the authors as "a perfect testing ground," so the
  reported transfer is probably optimistic for our setting.
