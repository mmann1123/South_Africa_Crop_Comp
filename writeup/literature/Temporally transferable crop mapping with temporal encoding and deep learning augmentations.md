# Temporally transferable crop mapping with temporal encoding and deep learning augmentations

**Citation:** Pham, V.-D., Tetteh, G., Thiel, F., Erasmi, S., Schwieder, M., Frantz, D., van der Linden, S. (2024). Temporally transferable crop mapping with temporal encoding and deep learning augmentations. *International Journal of Applied Earth Observation and Geoinformation* 129, 103867. DOI: [10.1016/j.jag.2024.103867](https://doi.org/10.1016/j.jag.2024.103867) (verified against Crossref, title match). Open access, CC BY 4.0. PDF from the Thünen Institute repository (`literatur.thuenen.de/digbib_extern/dn068235.pdf`).

Suggested cite key: `pham2024temporal`.

## Objectives

The paper asks whether a pixel-level deep crop classifier trained with one year of reference data can map crop types in other years whose optical time series have very different observation densities. The authors name two obstacles:

1. **No common representation for irregular observations.** Annual clear-sky observations (CSO) vary with sensor constellation (Landsat 5/7/8, Sentinel-2A/B) and with cloud cover, and the usual aggregation methods (compositing, spectral-temporal metrics, smoothing, linear interpolation) either need a fixed density or break when it changes.
2. **Multi-year reference data are scarce**, so the model has to be trained in a single year and transferred.

Their answer is a fixed-length daily encoding of all observations, **Temporal Encoding (TE)**, combined with two training-time augmentations:

- **Random Observation Selection (ROS)**, which simulates sparser years.
- **Random Day Shifting (RDS)**, which simulates phenology shifts between years.

The stated aim is temporal transferability: training on a recent, dense year and mapping back in time.

## Methods

**Study area and labels.** The study covers Brandenburg, Germany (29,640 km², about 45% agricultural, mostly rainfed single-crop systems). Labels come from the IACS/LPIS parcel polygons for 2010 to 2021, with about 30 regional crop classes aggregated into 14, grassland included.

**Imagery.** The paper uses all Landsat 5/7/8 scenes below 75% cloud (2010 to 2021) and all Sentinel-2A/B scenes below 85% cloud (2017 to 2021).

- **Processing:** FORCE to Level-2 surface reflectance, with Sentinel-2 resampled to 30 m through an approximated point spread function.
- **Inputs:** six shared reflectance bands (`Blue`, `Green`, `Red`, `NIR`, `SWIR1`, `SWIR2`) plus `NDVI`, `NDWI` and `SAVI`, so nine channels.
- **Cloud masking:** Fmask as improved by Frantz et al. (2018).
- **Observation counts:** reported per year (Fig. 1b, distribution in Supplement 1). Density varies strongly: about 80 or more CSO per pixel in 2018, far fewer in the Landsat-only years 2010 to 2016.

**Temporal Encoding.** Each pixel becomes a `365 x 9` array, one row per day of year.

- Each clear observation is written into the row of its acquisition day.
- **Days without an acquisition and days flagged cloud or cloud shadow are set to 0.** Same-day observations from several sensors are averaged, and day 366 folds into day 365 when that row is empty.
- Whole FORCE tiles (30 x 30 km, `1000 x 1000 x 365 x 9`) are initialised to zero and filled image by image, so every pixel has the same feature length regardless of density.
- **The zero is the encoding's "no observation" token and is used identically in training and inference.** Even in dense 2018, more than 75% of a pixel's daily cells are zero.
- The paper does not say whether inputs are standardised before the network, so whether a 0 lies inside the range of real values (for example `NDVI` = 0) is not addressed.

**Augmentations, applied on the fly during training.**

- **ROS:** keep a uniformly random fraction of a pixel's annual CSO, between 5% and 100%, and set the rest to 0. The authors present it as an adaptation of Sainte Fare Garnot et al.'s (2022) temporal dropout.
- **RDS:** shift each CSO independently by a random −16 to +16 days. It is inspired by TimeMatch (Nyborg et al. 2022), but applied per observation rather than to the whole series.
- **Application:** ROS then RDS, generating new augmented samples until accuracy plateaus (Fig. 5).

**Models.**

- **1D-CNN:** three blocks of `Conv1d` (kernel 16, 64 filters), BatchNorm, ReLU and max-pool 2, then a dense softmax over 14 classes.
- **Transformer:** the encoder-only model of Zhang, Luo and Li (2024, "CRIT"), with its input and output dimensions changed.
- **Five model variants:**

| Variant | Input | ROS | RDS | Classifiers |
|---|---|---|---|---|
| Baseline | TE | Yes | Yes | 1D-CNN and Transformer |
| TE-only | TE | No | No | 1D-CNN and Transformer |
| TE-ROS | TE | Yes | No | 1D-CNN |
| TE-RDS | TE | No | Yes | 1D-CNN |
| LI | Daily linear interpolation of CSO, same 365-day length | No | No | 1D-CNN |

**Training data.** Training uses 2018 only, chosen because it "had the highest number of clear sky observations", with 4,000 randomly placed pixels per class (56,000 in total). The training-set size comes from a pre-analysis in Supplement 4. Training pixels lie in the northern part of the state.

**Single-sensor test.** For each year, the Baseline model is also run on data from one sensor only (for example Landsat 7 alone in 2019). This tests robustness to artificially reduced density in the same year.

**Evaluation protocol.** This is a **spatially disjoint holdout combined with cross-year transfer.**

- **Spatial split:** Brandenburg is split into a northern training area and a southern test area (Fig. 1a), described as done "to allow for an unbiased evaluation".
- **Test set:** wall-to-wall maps of the southern area are compared against all LPIS parcels there for each year from 2010 to 2021.
- **In-year result:** the 2018 result is therefore already a north-to-south spatial transfer inside one state and year, not a k-fold estimate. The other eleven years add a change of year, with different sensor constellation, cloud climatology and phenology.
- **Leakage:** training is pixel-sampled, but train and test areas are disjoint, so there is no pixel or field leakage between them. How far the two areas are separated, and whether they differ in landscape, is not quantified.
- **Metric:** the headline metric is **overall accuracy (OA) only.** Class-wise results are in Supplement 6, not in the main text. No macro-F1, Kappa, confidence intervals or repeated runs are reported.
- **Class balance:** training is class-balanced, but the test maps are dominated by the area-dominant classes (grassland, maize, winter rye, winter wheat). Wall-to-wall OA is therefore close to area-weighted accuracy and can hide poor minority-crop recall.

## Key Findings

**Transformer vs 1D-CNN (Fig. 6).**

- Without augmentation, the Transformer beats the 1D-CNN by about 10 to 25 OA points in most transfer years.
- With ROS and RDS, the two converge: both are above 60% OA in every year, and the Transformer is ahead by only 1 to 3 points, except about 7 points in 2010 and about 4 points behind in 2012.
- The Transformer maps a 1,000 x 1,000 tile in about 60 minutes against about 18 for the 1D-CNN, so the rest of the analysis uses the 1D-CNN.
- The authors attribute the unaugmented Transformer's advantage to its masking, which can ignore the zero cells. They write that "applying dropout and masking to the input data could be equivalent to using ROS with CNN-based models".

**Accuracy follows the training year's observation density.**

- In 2018, Baseline and LI both reach about 90% OA.
- In other years, "the level of decrease was closely related to the number of annual CSOs" (Section 4.1).
- Baseline beats LI by 2 to 10 points in 2019 to 2021 and by 4 to 12 points in 2015 to 2017.
- In the sparse years 2010 to 2014, **LI falls to 30 to 54% OA while Baseline stays above 60% in the first four years.** Linear interpolation across long gaps produces "a high number of unrealistic features" (citing Roy and Yan 2020).
- In the LI maps for 2010 and 2012, most crops are misclassified as fallow (Fig. 8).

**Without augmentation the encoding overfits the training year's density (Fig. 7).**

- "The TE-only model is highly overfitted without the use of augmentation methods, i.e., only performed well in the training year and poorly in unseen years", and its loss is "severely weakened in relation to the sparsity of annual CSO" (Section 4.2).
- ROS and RDS together add about 20 OA points in 2016 and 2019 to 2021, and nearly 30 in 2017.
- Before 2016 they roughly double OA (2011 excepted).
- **In the training year they add only about 5 points.** Augmentation buys transfer, not in-domain accuracy.

**ROS matters more than RDS.**

- TE-ROS beats TE-RDS in most years, by about 20 points in 2010, 2012, 2014 and 2015, and by nearly 30 in 2012.
- The authors conclude that "the impact of low annual CSO counts being simulated by the ROS augmentation is higher than that of RDS simulating shifts in phenologies between years". They thus attribute most of the cross-year loss to observation density rather than phenology.

**Single-sensor robustness (Fig. 9).**

- **Same year, single sensor:** the Baseline model trained on 2018 with all sensors keeps similar accuracy when given only one sensor. In 2018: Landsat 7 81.5%, Landsat 8 89%, all sensors 91%.
- **Later years:** Landsat 8 alone is 3 to 10 points below all sensors in 2017 to 2021.
- **Landsat 5:** the model handles Landsat 5, which is absent from training, but with low OA (42.2% in 2010, 53.9% in 2011). The authors attribute this to the sensor's degradation at the end of its mission.
- **Landsat 7 stripes:** ROS largely removes the stripe artifacts from Landsat 7's failed scan-line corrector.

**Authors' caveat on direction.** ROS teaches a model trained on a dense year to handle sparse ones. The authors "expect the impact of ROS on improving transferability to be less effective or maybe even negative" in the reverse case (training on a sparse year, mapping a dense one).

**Code and data.** Data are "available on request". No code is released.

## Relevance to Our Crop-Classification Study

**This is the closest published demonstration of our mechanism, and the paper must engage with it directly.**

Pham et al. show three things:
1. A pixel deep model's transfer loss to another domain scales with how different that domain's observation density is from the training domain's.
2. A model that has never seen sparse inputs "only performed well in the training year and poorly in unseen years".
3. Training-time observation dropout (ROS) recovers most of that loss, with far more effect than augmenting phenology. A plain change of input representation does not recover it.

That is our month-dropout result (2 x 2 in `experiments/missingness/month_dropout/`), found earlier, in a cross-year rather than cross-region setting, and with its own independent attribution of transfer loss to density rather than phenology. It also matches the 2×2 finding that dropout restores accuracy under zero fill, while the in-domain gain stays small (their "about 5 points in the training year"; our near-zero differences under the training-median fill).

**What is not in Pham et al., and what defines our contribution relative to it:**

1. **Which domain shift.** Their shift is cross-year, with sensor constellation changing in step. Ours is cross-region in a single year and season: adjacent Sentinel-2 tiles, the same sensor and nominally the same acquisition dates, with the gap sub-tile and invisible in granule metadata. Their density difference is therefore anticipated by design; ours was not.
2. **Encoding and fill.** In TE, zero is a consistent "no observation" token. It saturates even the training data (over 75% of daily cells), so zeros are always inside the training support, and only the sparsity pattern shifts. In our case a whole month that is never missing in training arrives at inference zero-filled after a training-fitted `StandardScaler`, i.e. an out-of-distribution value in one slot. Our step-deletion and fill-sweep results separate the effect of the inserted value from the information lost. Pham et al. cannot separate them, because their only non-zero comparator, LI, changes the representation as well.
3. **Model ranking.** Pham et al. compare two deep architectures and show that augmentation closes the Transformer–CNN gap. We show that missingness shift reorders a 27-model benchmark spanning trees, linear models, TabNet and dense deep models, and creates an apparent in-region versus out-of-region inversion. Their result that the Transformer's masking gives it an edge without augmentation fits our finding that L-TAE step deletion equals the training-median fill.
4. **Sparse gating.** They do not test sparse feature selection; we do. Our month-dropout 2 x 2, in which the gate adds nothing once dropout is in, is consistent with their claim that augmentation is what matters.
5. **Metrics and statistics.** They report only OA from single runs. We report macro-F1 with field-level bootstrap intervals, McNemar tests and five-seed ensembles. Our September-observed versus September-missing subset analysis is an imputation-free check they have no counterpart to.

**How to cite it.**

- **Introduction:** as the published demonstration that observation density drives cross-year transfer loss in crop mapping, and that observation dropout during training is the remedy.
- **Discussion:** in the month-dropout section, as prior evidence that the dropout fix generalises beyond our setting.
- **What not to claim:** that we are first to show missing observations can pass for a transfer failure. Claim instead that we are first to show it across regions within one season, from a single fill inconsistency, reordering a multi-family benchmark.

The user's audit (`reports/Missingness confound audit of transfer papers.md`) ranks this paper as a rank-1 counter-example. It also pairs it with DENETHOR: same region (Brandenburg), with DENETHOR training on 2018, which Pham et al. identify as the clearest year of 2010 to 2021, and attributing its out-of-year drop to weather. That pairing is a strong illustration for the Introduction.

**Bearing on the second-dataset to-do item.** Brandenburg has public LPIS labels, a documented spread of densities across years, and a published density-transfer baseline. It is a natural candidate for testing generality, alongside BreizhCrops.

**Feature and sensor design.** The design is optical only, using Landsat and Sentinel-2 harmonised by FORCE at 30 m. Inputs are pixel-level raw daily sequences of six bands and three indices; there are no engineered time-series statistics, no field aggregation and no SAR. Their bands overlap ours only partly (we use `B2`, `B6`, `B11`, `B12`, `EVI` and hue on a monthly grid).

## Evaluation Caveats

- **OA only, on wall-to-wall maps dominated by a few area-dominant classes.** Minority-crop behaviour is in Supplement 6, not the main text, and the numbers are not comparable with our macro-F1. Even the 2018 "about 90%" is an area-weighted spatial-transfer OA.
- **No repeated runs, seeds or intervals.** Differences of a few points between Baseline and Transformer or between ROS and RDS in a given year cannot be judged for significance.
- **Year, density, sensor constellation and phenology all change together.** ROS beating RDS is the paper's evidence that density dominates, but the two augmentations have different strengths (5–100% retention against ±16 days), so the comparison depends on those parameters, as the authors concede for RDS.
- **The TE-only versus Baseline comparison bundles ROS with RDS.** The density attribution rests on the separate TE-ROS and TE-RDS arms, which were run only with the 1D-CNN.
- **LI changes the representation as well as the gap handling.** "Interpolation fails on long gaps" therefore cannot be separated from "interpolated daily series differ in distribution from training".
- **Input scaling is not described**, so it is unknown whether the zero token falls inside the range of real values after normalisation.
- **Spatial separation is coarse.** The north/south split is one division without distance or landscape statistics, so how much of the 2018 result is spatial transfer is not measured.
- **Training is pixel-sampled, at 4,000 per class**, with no field-level aggregation and no field-disjoint validation inside the training area. Neither the hyperparameter selection nor the "train until sufficient accuracy" stopping rule is described in enough detail to reproduce.
- **No code, and data on request.** The density statistics per year (Fig. 1b, Supplement 1) are the one auditable artefact, and they are what make the paper a counter-example in the user's audit.
