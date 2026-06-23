# Analysis of Time-Series MODIS 250 m Vegetation Index Data for Crop Classification in the U.S. Central Great Plains

**Citation:** Wardlow, B. D., Egbert, S. L., & Kastens, J. H. (2007). Analysis of time-series MODIS 250 m vegetation index data for crop classification in the U.S. Central Great Plains. *Remote Sensing of Environment*, 108(3), 290–310. DOI: 10.1016/j.rse.2006.11.021 (verified via Crossref).
**BibTeX key:** `wardlow2007analysis`

## Objectives

The paper investigates whether time-series MODIS 250 m vegetation-index (VI) data —
specifically the Enhanced Vegetation Index (`EVI`) and Normalized Difference
Vegetation Index (`NDVI`) — carry enough spectral–temporal information to
discriminate the major crop types and crop-related land-use practices of the U.S.
Central Great Plains (Kansas). Three research questions are posed: (1) Do the 16-day
250 m VI time series have sufficient spatial, spectral, and temporal resolution to
separate the region's major crops (alfalfa, corn, sorghum, soybeans, winter wheat)
and practices (double-crop, fallow, irrigation)? (2) Are regional climate/management
variations (e.g., planting-time gradients) detectable in the VI signatures? (3) How
do `EVI` and `NDVI` differ in response across crop types, and how informationally
distinct are they?

Crucially, this is a **separability/feasibility study**, not a classification-
accuracy study. It establishes the phenological basis for crop mapping with MODIS
VI time series rather than reporting a trained classifier's accuracy.

## Methods

A 12-month time series of 16-day MODIS 250 m `EVI` and `NDVI` composites (MOD13Q1
V004, 23 periods, calendar year 2001) was built for Kansas from three MODIS tiles,
mosaicked and reprojected to Lambert Azimuthal Equal Area. A field-site database of
**2,179 fields** (>=32.4 ha) was assembled from USDA Farm Service Agency annotated
aerial photos across 48 counties, labeled by crop and irrigated/non-irrigated status
(Table 1). For each field, a **single, "maximally interior" 250 m pixel** lying
fully inside the field boundary was extracted to minimize mixed-pixel contamination,
and its `EVI`/`NDVI` time series was used.

Analyses were graphical and statistical: (1) visual MODIS-vs-Landsat ETM+ comparison
at landscape/field scale; (2) class-averaged multi-temporal VI profiles compared to
documented crop calendars (phenology); (3) **Jeffries–Matusita (JM) distance**, a
statistical separability measure ranging 0–2, computed pairwise between crop classes
both over the full growing season and period-by-period; (4) Agricultural Statistics
District (ASD)-level profiles to characterize intra-class regional variation across
Kansas's precipitation/planting gradients; and (5) correlation/crossplot analysis of
the `EVI`–`NDVI` relationship across greenup and senescence phases.

**Evaluation protocol.** There is **no classifier and no accuracy metric** — no
k-fold, no holdout, no macro-F1, no Kappa. Generalization is assessed only as
**class separability via JM distance** on field-mean VI profiles within a single
state and single year. On our spectrum (pooled-pixel k-fold -> field-wise k-fold ->
spatial holdout -> cross-year/cross-sensor), this paper does not occupy a
classification rung at all; it is upstream of classification. Its "test" of transfer
is whether crop classes remain JM-separable across Kansas's four corner ASDs (a
spatial-variation probe, not a train/test transfer experiment) and whether 2001
matched the climatological average (it did, per USDA — so cross-year robustness is
explicitly *not* tested; an average year was deliberately chosen). One detail that
favors clean methodology: a single interior pixel per field removes within-field
pixel autocorrelation, so the JM distances are not inflated by adjacent-pixel
leakage. The flip side is that per-field sample size is one pixel, so the study
cannot speak to pixel-level vs field-level aggregation tradeoffs — our central
variance-reduction lever.

## Key Findings

- Each major crop class showed a **unique, well-defined multi-temporal VI profile**
  consistent with its crop calendar: summer crops (corn, sorghum, soybeans) peak in
  July–August; winter wheat peaks in late April–early May; alfalfa shows
  multiple growth-and-cut cycles across a broad season; fallow stays low.
- Most crop classes were **spectrally separable at some point in the season** by JM
  distance, with separability changing over the growing season — i.e., the
  discriminating information is concentrated in specific phenological windows, not
  uniform across the year.
- **Irrigated crops** had higher peak `NDVI` and sustained higher `NDVI` than their
  non-irrigated counterparts; management practice is detectable in the VI signal.
- **Regional (ASD-level) intra-class variation** of up to ~1 month temporal offset
  in VI profiles was found, driven by Kansas's east–west precipitation and
  planting-time gradients — a direct demonstration that the *same crop looks
  different in a different sub-region*, which is the seed of our spatial-transfer
  concern.
- `EVI` and `NDVI` tracked similar seasonal responses and were highly correlated
  across most of the season, but **diverged most during senescence**, suggesting the
  two indices are partly complementary at specific phases rather than redundant.

## Relevance to Our Crop-Classification Study

This is a foundational, heavily-cited (>800) demonstration that **multi-temporal
optical VI signatures carry phenological information sufficient to separate crops** —
the premise our entire feature pipeline rests on. It validates our use of `EVI` as a
core band and our reliance on time-series structure rather than single-date imagery.
Several points map directly onto our manuscript:

- **Phenological windows matter more than full sequences.** Separability is
  concentrated in specific composite periods. This supports our use of engineered
  time-series statistics (xr_fresh features) and feature-selecting models that can
  zero in on discriminating windows, over dense models that weight the whole sequence
  uniformly.
- **The same crop differs across sub-regions.** The ASD-level intra-class variation
  (~1 month offset across the precipitation gradient) is precisely the mechanism by
  which a model fitted in one tile can mis-rank out-of-sample — a different
  geography shifts the phenology. This paper is an early, concrete piece of evidence
  for *why* our spatially disjoint holdout reorders models.
- **Field-mean / interior-pixel design.** Their single-interior-pixel-per-field
  choice is a manual variance-reduction move analogous to our field-level
  aggregation; it foreshadows our finding that aggregating to the field level
  stabilizes classification.
- **EVI vs NDVI complementarity at senescence** is a small but useful note for our
  optical-only, index-based feature design.

It is best cited as motivation/premise ("crops are phenologically separable in
multi-temporal optical VI data, but intra-class signatures vary regionally"), not as
a comparable accuracy benchmark.

## Evaluation Caveats

- **No accuracy, no classifier.** Cite this for *separability* (JM distance) and
  phenological structure, never as a crop-classification accuracy result. There is
  no macro-F1, Kappa, or OA to compare against our spatial-holdout numbers.
- **Single year, deliberately average.** 2001 was chosen because it matched
  Kansas's climatological average; cross-year robustness is explicitly outside scope.
  A drought or wet year could collapse the clean separability shown here.
- **Single interior pixel per field.** This avoids within-field pixel
  autocorrelation (good — no spatial leakage in the JM estimates) but also means the
  study cannot inform pixel-vs-field aggregation, our key variance lever, and
  discards the within-field heterogeneity real classifiers must handle.
- **Large fields only (>=32.4 ha).** Sites were restricted to fields large enough for
  multiple clean 250 m pixels; smaller/edge fields (more common in our Western Cape
  setting at Sentinel-2 resolution) are under-represented, so separability here is an
  optimistic upper bound for finer-grained landscapes.
- **Class imbalance present but not a metric issue.** Field counts are skewed
  (corn 609, fallow 73; Table 1), but because no classifier is trained, no
  minority-recall artifact arises — though it also means the study gives no guidance
  on minority-crop classification under imbalance.
- **Optical-only, MODIS 250 m.** Coarser than our Sentinel-2 work and no SAR; the
  spectral/temporal richness here (`EVI`/`NDVI` only) is narrower than our
  `B2,B6,B11,B12,EVI,hue` design.
