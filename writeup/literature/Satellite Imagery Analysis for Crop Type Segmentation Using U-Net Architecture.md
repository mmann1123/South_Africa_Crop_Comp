# Satellite Imagery Analysis for Crop Type Segmentation Using U-Net Architecture

**Citation:** Ayushi, & Buttar, P. K. (2024). Satellite Imagery Analysis for Crop Type Segmentation Using U-Net Architecture. *Procedia Computer Science*, 235, 3418–3427 (ICMLDE 2023). DOI: 10.1016/j.procs.2024.04.322 (verified via Crossref).

## Objectives

The paper proposes a U-Net fully convolutional encoder-decoder for *semantic segmentation* of crop types from multi-temporal Sentinel-2 imagery in the difficult smallholder setting (irregular small fields, sparse labels, heavy cloud cover). The stated aim is to recast pixel-wise crop classification as a segmentation task and show that a patch-based, spatially-convolutional deep net outperforms both classical pixel classifiers (`RF`, `CatBoost`, `KNN`) and competing segmentation nets (`FCN`, `SegNet`) on the CV4A Kenya crop type dataset.

## Methods

Data: the openly available CV4A Kenya crop type detection dataset (Radiant Earth / PlantVillage), 4 tiles, 4,688 fields, 13 multi-band Sentinel-2 L2A observations over the growing season (12 bands + cloud-probability layer, 10 m grid). Seven crop classes — three pure (maize, cassava, common bean) and four intercropped combinations. Preprocessing pipeline (eo-learn): cloud filtering via Otsu-thresholded cloud-probability mask (drop scenes >70% cloud), NDVI computation, temporal linear interpolation to gap-fill, and rasterization of vector crop polygons into ground-truth label masks. The U-Net (23 conv layers, contracting/expanding paths with skip connections) takes 64×64×8 input patches, trained with Adam (lr 1e-3), Dice loss, batch 16, 50 epochs (PyTorch/FastAI). Spatial context is exploited via 2D convolutions — this is a *patch/spatial* model, not a pixel or field model.

**Evaluation protocol:** Tiles are chipped into 64×64 patches, then the patch set is split **80/10/10 train/validation/test by random patch assignment**. This is a **pooled/random patch-level split within the same four tiles** — there is no spatially disjoint holdout tile, no FID-disjoint grouping, and no cross-region/cross-year evaluation. Because chips are drawn randomly from the same scenes and neighboring chips share field boundaries and phenology, the protocol carries substantial **spatial autocorrelation leakage**: this is in-region validation, not out-of-sample transfer. Metrics: overall accuracy, precision, recall, and F1 (pixel-level, aggregated).

## Key Findings

- U-Net reached 95.3% accuracy, 80.2% precision, 68.1% recall, 73.6% F1, beating `FCN` (F1 69.26), `SegNet` (63.33), `KNN` (48.89), `CatBoost` (42.74), and `RF` (31.35).
- All three deep segmentation nets (U-Net, FCN, SegNet) beat all classical pixel classifiers by a wide margin on this dataset — the paper's headline claim that spatial-context CNNs dominate.
- Per-crop accuracy was decent for single crops (common bean 70.4%, maize 69.1%, cassava 64.0%) but collapsed on the four intercropped classes (15.6–24.9%), the dominant failure mode.
- The huge accuracy/F1 gap (95.3% vs 73.6%) signals heavy class imbalance: background/majority pixels are easy, minority intercrop classes are not.
- The conclusion explicitly notes that crop algorithms calibrated for one region "cannot be readily extended to another" — yet transferability is named only as future work.

## Relevance to Our Crop-Classification Study

This is a direct comparator for the *patch-based / dense-CNN* arm of our model spectrum and a clean illustration of the failure mode our manuscript diagnoses. The authors report a dominant in-region result (U-Net F1 73.6%, beating RF by 42 F1 points) under a random-patch split from the *same four tiles* — exactly the in-region-k-fold setting that, in our experiments, misranks dense spatial nets above sparse tree/TabNet models. Our central finding is that this ranking *reverses* on a spatially disjoint holdout because dense patch nets overfit scene-specific spatial/phenological texture. The Kenya smallholder context (small irregular fields, sparse labels, cloud-dominated optical, intercropping) closely parallels our Western Cape setting and our preprocessing (cloud handling, temporal interpolation, optical-only Sentinel-2). The intercrop-class collapse mirrors our minority-class (small grains) difficulties, and the catastrophically low `RF` number here (31% F1) is suspicious — likely a weakly-tuned pixel baseline rather than evidence that trees are intrinsically worse, a caution our manuscript's careful classical baselines address.

## Evaluation Caveats

- **No spatially disjoint holdout.** The 80/10/10 split is random over patches chipped from the same four tiles; neighboring chips leak spatial/phenological information across the split. Reported numbers are **in-region, not out-of-sample**, despite the paper invoking smallholder-transfer motivations. The authors themselves flag transferability as untested future work — a silence that is itself a finding for us.
- **Patch model with no FID-disjoint splitting.** Field identity is not used to group the split, so same-field pixels/chips appear in both train and test — leakage in the field-aggregation sense too.
- **Accuracy-dominated headline.** 95.3% overall accuracy vs 73.6% F1 betrays class imbalance; F1 is reported (good) but is not macro-averaged transparently, and per-class numbers show minority intercrop classes near random. No Kappa.
- **Optical-only, no SAR**; uses NDVI plus Sentinel-2 bands (input depth 8), consistent with our optical-only design but offering no cloud-penetrating signal in a cloud-heavy region.
- **Weak classical baselines.** `RF`/`CatBoost`/`KNN` results are implausibly low, suggesting the comparison is stacked toward the proposed CNN; treat the "deep beats classical" claim as protocol- and tuning-dependent, not a transfer-robust result.
- **What it does not measure:** field-level aggregation accuracy, cross-tile or cross-year generalization, sensitivity to the random-split seed, or whether the U-Net advantage survives a spatial holdout — the exact questions our study foregrounds.
