# Learning the variations in annual spectral-temporal metrics to enhance the transferability of regression models for land cover fraction monitoring

**Citation:** Pham, V.-D., Thiel, F., Frantz, D., Okujeni, A., Schug, F., van der Linden, S. (2024). Learning the variations in annual spectral-temporal metrics to enhance the transferability of regression models for land cover fraction monitoring. *Remote Sensing of Environment* 308, 114206. DOI: [10.1016/j.rse.2024.114206](https://doi.org/10.1016/j.rse.2024.114206) (verified against Crossref, title match). Open access, CC BY 4.0.

Suggested cite key: `pham2024fractions`. Companion to `pham2024temporal` (same first author, same lab, same Random Observation Selection idea applied to crop-type classification).

## Objectives

The paper studies annual land-cover fraction mapping (regression-based unmixing) from Sentinel-2 spectral-temporal metrics (STM) over six years. It asks three questions:

1. How much does annual clear-sky observation (CSO) density affect regression accuracy under temporal and spatial transfer?
2. Does simulating lower density during training improve transfer?
3. Does annual mapping support consistent multi-year monitoring?

The paper's motivation is that STM, being per-pixel annual percentiles, are often treated as density-robust. Fig. 1 shows that the same forest pixel's percentiles move from year to year with how many observations fall in which season.

## Methods

**Study area and data.** The study covers northern Germany and Poland (405,000 km², seven Sentinel-2 orbits, 450 FORCE tiles of 30 x 30 km).

- **Imagery:** all Sentinel-2A/B L1C scenes below 75% metadata cloud cover, 2017 to 2022.
- **Processing:** FORCE to Level-2, with Fmask cloud masking (Frantz et al. 2018).
- **Bands:** ten (`B2`, `B3`, `B4`, `B8`, `B5`, `B6`, `B7`, `B8A`, `B11`, `B12`; the text lists "10" among the 20 m bands, presumably a typo for `B12`), with the 20 m bands sharpened to 10 m by ImproPhe.
- **Features:** 25th, 50th, 75th and 90th percentiles per band, so 40 features per pixel-year.

**Observation density is reported per year and is the paper's central variable** (Fig. 2, mean CSO per pixel):

| Year(s) | Mean CSO per pixel | Reason given |
|---|---|---|
| 2017 | 13 ± 4 | Wet year; Sentinel-2B only became operational late in the year |
| 2018 | 35 ± 9 | Extreme hot, dry summer |
| 2019, 2020, 2022 | about 30 ± 8 | |
| 2021 | low | Wet year |

Density also varies in space: areas under a single orbit receive fewer CSO than areas where orbits overlap.

**Labels and model.** Training uses regression-based unmixing with synthetic mixtures (Okujeni et al. 2013).

- **Spectral library:** 966 pure pixels from five German tiles (impervious 298, woody 208, low vegetation 362, background 98), checked for no change from 2017 to 2022 against VHR imagery, CLMS layers and NDVI thresholds.
- **Synthetic mixtures:** for each year, 128,000 mixtures of 1 to 4 pure STM profiles from that year with random fractions that sum to 1.
- **Network:** multi-output neural-network regression (two hidden layers of 512, 40 inputs, 4 outputs, MAE loss, Adam at learning rate 1e-3, TensorFlow 2.10).
- **Single-year models:** M-17 to M-22.
- **Multi-year model:** M-All, trained on all six years.

**Random Observation Selection (ROS).** For each pure pixel and year:

- Draw random subsets of 80%, 60%, 40% and 20% of that year's CSO, keeping at least four observations.
- Recompute the four percentiles on each subset, giving four augmented STM sets alongside the original.
- Mix from the enlarged library for the ROS models, M-17-R to M-All-R.

ROS is used in training only; inference uses the original STM. Unlike the crop paper (`pham2024temporal`), nothing is zero-filled. Missing observations here only change the summary statistics.

**Evaluation protocol.** This design is **cross-year transfer crossed with a spatially disjoint holdout**.

- **Validation sites:** 1,000 sites of 3 x 3 pixels (30 x 30 m), labelled by annotating a 10 x 10 grid of 3 m cells on VHR imagery and stratified by fraction interval. All sites are invariant from 2017 to 2022, so one reference serves every year.
- **Local tiles:** 527 sites in the five German training tiles. These test temporal transfer at the same location, plus spatial generalisation within those tiles, because validation sites are not the library pixels.
- **External tiles:** 473 sites in five eastern tiles (Poland, including Warsaw and Gdansk), chosen to be harder, and three of the five under a single orbit, so sparser. These test spatial and temporal transfer together.
- **Train-test independence:** training uses synthetic mixtures of library pure pixels, validation uses real mixed pixels, so there is no direct leakage. Whether any validation site sits next to a library pixel on the local tiles is not stated.
- **Metrics:** MAE, RMSE, slope, intercept and R² per class. No intervals, no repeated runs.

## Key Findings

**Same-year baselines are accurate.**

| Class | MAE | R² |
|---|---|---|
| Impervious | 8.5 to 10.9 | at least 0.89 |
| Woody | 11.0 to 12.1 | 0.87 to 0.91 |
| Low vegetation | 12.1 to 13.7 | 0.87 or above |

**A model trained on a dense year transfers worst (Fig. 8).**

- "Models trained in years with high data density performed well in that same year but poorly when transferred to years with lower density."
- M-18, the densest year, has the poorest temporal transferability: woody MAE 15.1 to 18.3 and low-vegetation MAE 15.1 to 18.5 in other years, against roughly 11 to 13 in 2018.
- Impervious stays about 10 or below everywhere, because surfaces without phenology have stable STM.
- Models trained on sparse years degrade less, and sometimes improve: M-17 maps other years better than its own, and M-21 is close to its baseline in most years.
- The authors' explanation is that dense years give stable STM, so the model learns less variation.

**The year being mapped matters too.**

- Every model is most accurate when mapping 2018, the densest year, and least accurate on 2017, the sparsest.
- The authors note that density is confounded with weather and land-cover condition between years. They state that isolating density "would need constant land cover conditions under varying data density", which they did not have.

**ROS improves transfer, mostly for models trained on dense years (Fig. 9).**

- **Same year, local sites:** gains are small, −1% to −8% relative MAE, with slight losses in some cases.
- **Same year, external sites:** gains are larger, up to −16% (impervious: M-18 9.36 to 7.88, M-20 9.92 to 8.36). The authors attribute this to density differences between the west and east of the area, including single-orbit tiles.
- **Unseen years:** M-18-R reduces MAE by up to −10%, −17% and −14% on local sites (impervious, woody, low vegetation) and −19%, −14% and −16% on external sites, about −15% on average. Gains for M-19-R, M-20-R and M-22-R are −3% to −8%; for M-17-R and M-21-R they are mostly small.
- **By mapped year:** ROS helps most when mapping sparse 2017 and least when mapping dense 2018.

**Training on all years is best, and ROS still helps it.**

- M-All beats every single-year model. Average six-year MAE is 10.38 on local sites and 10.82 on external sites.
- With ROS, M-All-R reaches 10.17 local and 10.10 external, so local and external accuracy become almost identical.

**Cost and recommendation.**

- ROS multiplies training time by about 4, and by 24 for M-All-R.
- **Recommendation:** ROS is clearly worthwhile when labels exist only for a single dense year and sparser years must be mapped. When labels come from a year as sparse as the targets or sparser, the gain is positive but small. With labels for all years it matters least.

**Their stated future work** includes "more challenging land cover categories such as crop types" and "regions with more challenging observation availability".

**Code and data.** Neither a code nor a data availability statement was found in the extracted text.

## Relevance to Our Crop-Classification Study

This is the second, independent Pham et al. demonstration that a difference in observation density between training and target domains is enough to produce an apparent transfer failure, and that training-time observation removal fixes it.

**What it adds to `pham2024temporal`.**

1. **Spatial as well as temporal transfer.** ROS improves same-year transfer to the external tiles (up to −16% MAE), which the authors attribute to orbit-driven density differences between areas in the same year. This is the paper's closest point to our setting, where the shift is spatial within one season. It is still only a few tiles, and confounded with Poland's more continental phenology, which the authors also invoke.
2. **The mechanism does not need a fill value.** STM are NaN-free percentiles over whatever observations exist, yet density still shifts the feature distribution enough to cost accuracy. Our xr_fresh classical models are the analogue: they take NaN-aware annual statistics, and moved only 0.02 to 0.05 under the consistent fill. This paper says those statistics are still density-sensitive. Our fill sweep and the September-observed versus September-missing subset analysis are the right checks for that, and the paper should say so rather than calling the statistic features immune.
3. **The direction asymmetry.** Training on the densest year transfers worst, and training on a sparse year transfers best. Our training tiles have no September gaps and the holdout misses September in about 41% of pixels, the worst-case direction in their terms.

**Where our case differs.**

- **Task:** theirs is regression of land-cover fractions, not crop classification.
- **Features and model:** theirs uses annual percentiles with an MLP; we use monthly sequences with temporal deep models plus tabular models.
- **Size of the density difference:** theirs is a matter of degree (13 against 35 CSO). Ours is one specific month that is never missing in training, encoded as an out-of-range value after scaling.
- **What we add:** a fill-value contrast (step deletion), a model-ranking reversal across a 27-model, multi-family benchmark, field-level macro-F1 with bootstrap intervals and McNemar tests, and the sparse-gate comparison.

**How to cite it.** Cite it alongside `pham2024temporal` wherever the manuscript says density or missingness differences can masquerade as transfer failure. Its "train-dense, transfer-poorly" asymmetry belongs in the Discussion of why our holdout was hit so hard. Its external-tile ROS gain is the one published hint of a spatial, same-year version of the effect.

**Bearing on the to-do list.**

- **Per-domain reporting:** the paper is a model for reporting density per domain (mean ± SD of CSO per year and per orbit configuration), the statistic the manuscript should publish for each tile and month.
- **Second dataset:** it supports a Germany-based second dataset with documented year-to-year density variation.

## Evaluation Caveats

- **Not crop classification.** The metric is per-class fraction MAE, so the numbers are not comparable with macro-F1. "Low vegetation" pools crops, grassland and wetland.
- **Single runs.** There are no seeds, intervals or significance tests. Several of the ROS gains on individual models are a few percent of MAE and may be within run-to-run noise.
- **Density is confounded with year.** Year-to-year density moves with weather (2018 hot and dry, 2017 and 2021 wet) and land-cover condition. The authors concede this; ROS beating the standard model is the stronger evidence, because it varies density with everything else fixed.
- **Spatial transfer is confounded too.** The external tiles differ in country, climate (more continental), landscape structure and orbit coverage at once, and were deliberately chosen to be harder.
- **ROS is a fixed four-level subsampling** (80/60/40/20%, minimum four CSO), not a random rate. Whether the gain comes from the specific rates or from the extra library size (5x the training spectra) is not separated.
- **One validation reference per site for all years.** This assumes six-year invariance, checked on VHR imagery but limited by its availability, and excludes changing areas from quantitative validation.
- **The validation sample is stratified on the 2022 map**, made by a model trained with these data, a mild circularity in the sampling design.
- **No code or data statement** was found in the extracted text.
