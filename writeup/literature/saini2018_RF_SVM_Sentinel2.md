# Crop Classification on Single Date Sentinel-2 Imagery Using Random Forest and Support Vector Machine

**Citation:** Saini, R., & Ghosh, S. K. (2018). Crop Classification on Single Date Sentinel-2 Imagery Using Random Forest and Support Vector Machine. *The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, XLII-5, 683–688 (ISPRS TC V Mid-term Symposium, Dehradun, India). DOI: 10.5194/isprs-archives-XLII-5-683-2018 (verified via Crossref).
**BibTeX key:** `saini2018single`

## Objectives

The paper sets out a deliberately narrow comparator question: which of the two most widely used classical remote-sensing classifiers — Random Forest (`RF`) or Support Vector Machine (`SVM`) — produces more accurate crop and land-cover maps from a *single-date* Sentinel-2 acquisition over Roorkee, Uttarakhand, India. The authors aim to demonstrate (1) that Sentinel-2's medium resolution is adequate for vegetation/crop discrimination, and (2) that tuned `RF` and `SVM` both perform "well," with `RF` expected to edge ahead. It is a baseline methods comparison, not a methodological advance.

## Methods

A single Sentinel-2 scene from 19 February 2018 (growing season) covering 1049 km² was used. Only four 10 m bands were stacked — Near-Infrared (`B8`), Red (`B4`), Green (`B3`), Blue (`B2`) — so each sample is a 4-dimensional spectral vector with no temporal, red-edge, or SWIR information. Eleven LULC classes were defined (High-/Low-Density Forest, Sandy area, Water, Fallow, Built-up, Orchard, Wheat, Sugarcane, Fodder, Other crops). Reference labels came from GPS field survey plus Google Earth interpretation. `RF` was tuned to `ntree=350`, `mtry=1`; `SVM` used an RBF kernel with grid-searched `C=64`, `gamma=1`. Both were implemented in R.

**Evaluation protocol:** Stratified random sampling with a 70/30 train/test split and 10-fold cross-validation, with the explicit statement that "partitioned training and testing pixels are mutually exclusive." This is **pooled/random pixel-level k-fold cross-validation within a single scene** — the weakest rung of the spatial-rigor spectrum. There is no field/parcel (FID) grouping, no spatial blocking, and no separate holdout tile. Because pixels are split randomly, neighboring pixels of the same field almost certainly appear in both train and test sets, producing **spatial autocorrelation leakage** that inflates accuracy. The reported metrics are overall accuracy, Cohen's Kappa, and per-class F1.

## Key Findings

- `RF` achieved 84.22% overall accuracy and 83.05% Kappa; `SVM` achieved 81.85% and 79.13%. `RF` outperformed `SVM` by +2.37% overall.
- Per-class F1 (the only balanced metric reported) was highest for High-Density Forest (`RF` 92.93%) and lowest for Fodder (`RF` 61.22%, `SVM` 59.21%), attributed to spectral confusion between Wheat and Fodder on a single date.
- Feature importance: `NIR` was the single most important band for both classifiers; Green and Blue contributed almost nothing. With only four bands and one date, the discriminative signal is thin.
- `RF` improved every class relative to `SVM` (or tied), and the authors conclude `RF` has "better potential" and that Sentinel-2 is suitable for crop mapping.

## Relevance to Our Crop-Classification Study

This paper is a useful *negative control* and historical comparator for our pipeline. It anchors the classical `RF`/`SVM` baseline that our gradient-boosted-tree and TabNet models are meant to surpass, and it confirms the general finding that sparse, tree-based ensembles transfer at least as well as margin-based `SVM` on optical crop data. However, its evaluation is exactly the kind our manuscript argues against: single-scene, single-date, random pixel k-fold. The Wheat/Fodder confusion it documents is directly analogous to our Wheat/Barley/Small-grain-grazing confusions, reinforcing that single-date optical reflectance is insufficient to separate spectrally similar small grains — motivating our multi-temporal, engineered-feature design (`B2,B6,B11,B12,EVI,hue` across months). Its `NIR`-dominant importance also contrasts with our use of SWIR (`B11`, `B12`) and hue, which the temporal setting makes informative. Most importantly, it is a textbook example of the in-region-k-fold inflation our paper shows misranks models: a 84% number here says nothing about out-of-tile transfer.

## Evaluation Caveats

- **Spatial leakage is severe and unaddressed.** Random pixel-level splitting within one scene means same-field pixels straddle the train/test boundary; reported accuracies are optimistically biased and are *not* indicative of out-of-region or out-of-field performance. No FID-disjoint or spatially blocked split.
- **In-region k-fold, not OOS.** There is no holdout tile, no cross-year, no cross-sensor evaluation. Transferability is entirely unmeasured.
- **Class imbalance partly mitigated by per-class F1**, which is reported alongside overall accuracy and Kappa — better than accuracy-only — but no macro-F1 is computed, so the headline 84% remains a prevalence-weighted (accuracy) figure. Minority classes (e.g., Fodder) drag heavily but are not summarized in a single balanced number.
- **Single date, four bands, no temporal dimension** — phenological separability that drives modern crop mapping is entirely absent; many "crop" classes here are really stable land cover (Forest, Water, Built-up), inflating overall accuracy.
- **What it does not measure:** generalization across space or time, field-level aggregation, the value of SWIR/red-edge bands, or any deep-learning comparator. Silence on transfer is the key omission relative to our thesis.
