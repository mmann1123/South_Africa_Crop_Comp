# Smallholder Maize Area and Yield Mapping at National Scales with Google Earth Engine

**Citation:** Jin, Z., Azzari, G., You, C., Di Tommaso, S., Aston, S., Burke, M., & Lobell, D. B. (2019). Smallholder maize area and yield mapping at national scales with Google Earth Engine. *Remote Sensing of Environment*, 228, 115–128. DOI: 10.1016/j.rse.2019.04.016 (verified via Crossref).
**BibTeX key:** `jin2019smallholder`

## Objectives

The study builds "wall-to-wall" 10 m maps of (i) cropland presence, (ii) maize presence, and (iii) maize yield for the 2017 season across the entirety of Kenya and Tanzania (>1.5 million km²) using Google Earth Engine (GEE). The explicit goal is *scalability* — methods that work in any smallholder region in any year with minimal dependence on cloud-free imagery or local ground calibration — addressing the data scarcity of Global-South smallholder systems where official crop maps do not exist. Yield mapping aims to be accurate without any in-region ground calibration.

## Methods

Feature design fuses **Sentinel-1 (radar backscatter) and Sentinel-2 (optical reflectance)** via seasonal *median composites* per pixel — a deliberately parsimonious representation that sidesteps the cloud-cover problem endemic to tropical growing seasons. A custom Sentinel-2 cloud/shadow mask was built because the default mask performed poorly in the region. Two Random Forest classifiers were trained sequentially: crop/non-crop (`CRL`), then maize/non-maize (`MZL`) among cropped pixels, on several thousand GEE-labeled crop/non-crop points and thousands of crop-type labels (2015–2017). Yield mapping used the Scalable Crop Yield Mapper (SCYM): a crop-model simulation trains a regression predicting yield from satellite vegetation indices, advanced here by grouping simulations into Global Agro-Environmental Stratification zones, gridded soil/sowing/harvest inputs, and harmonic-regression fits over all available observations.

**Evaluation protocol:** This is the strongest spatial protocol among our assigned papers. Classification accuracy is reported as **out-of-sample** (>85% cropland accuracy in both countries; maize 79% Tanzania, 63% Kenya), evaluated on held-out labels at the national scale across heterogeneous landscapes — effectively a **spatially extensive, cross-landscape holdout** rather than a single-scene random split. Yield estimates are validated against *independent objective ground-based crop cuts* at the district level in Western Kenya (SCYM captured ~50% of district-level yield variance) — an external, uncalibrated validation. Soil-constraint analysis used independent soil databases (explaining 72% of predicted-yield variation). The protocol approaches genuine cross-region transfer and uses external ground truth, not in-region resubstitution.

## Key Findings

- Cropland (`CRL`) classifier exceeded 85% out-of-sample accuracy in both countries; **Sentinel-1 radar was particularly useful** for cropland detection — a data-fusion result.
- Maize (`MZL`) accuracy was country-dependent: 79% Tanzania vs 63% Kenya, exposing how transfer degrades with landscape heterogeneity and label quality.
- SCYM yields were accurate *without any ground calibration*, capturing ~50% of district-level yield variation against crop-cut ground truth.
- Independent soil data explained 72% of variation in predicted yields, with soil nitrogen and organic carbon most strongly associated — a downstream scientific application of the maps.
- GEE + Sentinel composites enabled processing ~1 trillion pixel-observations per season, demonstrating operational national-scale feasibility.

## Relevance to Our Crop-Classification Study

This is a high-water-mark example of the evaluation rigor our manuscript advocates, in a directly relevant Global-South, smallholder, cloud-limited optical context. Three lessons transfer to our Western Cape study. First, its **honest out-of-sample, cross-landscape accuracy reporting** (and the visible Kenya-vs-Tanzania gap) is exactly the spatially-disjoint-holdout discipline we argue for — in-region k-fold would have hidden that 63% Kenya maize number. Second, the maize accuracy *dropping* across regions is the empirical signature of the transfer penalty our paper quantifies, and it validates using a separate region/tile as the real test. Third, it shows the payoff of **sparse tree-based classifiers (Random Forest) on parsimonious seasonal-composite features** — a feature-reduction philosophy aligned with our engineered xr_fresh statistics and with the manuscript's claim that compact, feature-selecting models transfer best. The one design divergence worth noting: this paper *adds Sentinel-1 SAR* and finds it valuable for cropland detection, whereas our study is optical-only (`B2,B6,B11,B12,EVI,hue`); their result is a flag that SAR fusion could be a future axis for improving our cloud-robustness and transfer.

## Evaluation Caveats

- **Coarse semantic target.** The classification task is binary-ish (cropland vs not; maize vs not), not multi-class crop discrimination. The strong out-of-sample numbers do not translate directly to our 9-class macro-F1 problem, where minority small-grain confusion dominates.
- **Class balance / metric.** Accuracy is the primary reported classification metric; no macro-F1 or Kappa for crop classes, so minority-class performance within "cropland" is not resolved. For a 2-class task this matters less, but it is not a balanced-metric protocol.
- **Yield validation is aggregated.** SCYM is validated at *district* level (~50% variance explained), not at field level; pixel/field-scale yield accuracy is not directly demonstrated, and ~50% leaves substantial unexplained variance.
- **Region-dependent transfer is real and unmitigated** — the Kenya maize result (63%) shows the method does not transfer uniformly; landscape heterogeneity and label scarcity cap performance, a caveat consistent with our transfer findings.
- **What it does not measure:** field-level aggregation for crop *type*, deep-learning comparators, cross-*year* stability of the classifiers, or per-crop balanced metrics — its contribution is scalability and external validation, not model ranking.
