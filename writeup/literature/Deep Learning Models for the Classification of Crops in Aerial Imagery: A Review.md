# Deep Learning Models for the Classification of Crops in Aerial Imagery: A Review

**Citation:** Teixeira, I.; Morais, R.; Sousa, J.J.; Cunha, A. (2023). Deep Learning Models for the Classification of Crops in Aerial Imagery: A Review. *Agriculture*, 13(5), 965. DOI: 10.3390/agriculture13050965

> Review/theory paper. Per authoring rule 7, the formal evaluation-protocol question is skipped; the Relevance section below focuses on methodological and conceptual implications for our study (inductive bias, training-data dependence, transfer, and what the corpus systematically does NOT measure).

## Objectives

A PRISMA-guided systematic review of deep-learning (DL) approaches for crop classification from aerial imagery (satellite, UAV, and aircraft), covering peer-reviewed studies published 2020-2022. The review is organized around four research questions:

1. Which DL architectures are commonly employed for crop classification?
2. How does DL performance compare to classical machine learning (ML)?
3. What aerial-imagery sources and sensors are used to train models?
4. How many classes are used, and does class count affect performance?

The motivating goal is to map the state of the art, identify which models achieve high performance, and assess the influence of data availability, spatial/spectral resolution, sample quality, and the inclusion of non-crop classes.

## Methods

The authors searched Google Scholar and Scopus (via Harzing's Publish or Perish) with keyword combinations of "image", "crop classification", and "deep learning" plus synonyms, restricted to English peer-reviewed work from 2020-2022. Of 262 initial records, deduplication and title/abstract/full-text screening reduced the pool to 36 studies. Papers focused on segmentation, non-aerial imagery, weeds/disease/leaves/trunks, or non-DL methods were excluded. Included studies are grouped by capture system: satellite (Table 1, the largest group, dominated by Sentinel-2), UAV (Table 2), and aircraft (two studies).

The review tabulates, per study, the architectures used, data source, number of spectral bands, and number of crop classes (2 to 22). Architectures span 1D/2D/3D-CNNs, LSTM/Bi-LSTM/GRU recurrent nets, transformers, hybrid CNN-RNN and CNN-transformer models, PSE+LTAE temporal-attention encoders, GANs, and CNN-CRF spatial-context models. Cross-cutting techniques catalogued include data augmentation, transfer learning, and optical-SAR multimodal fusion.

**Evaluation protocol.** This is a secondary synthesis, not a primary experiment, so it inherits whatever protocol each of the 36 source studies used and does NOT impose a uniform one. The reported metrics across the corpus are overwhelmingly overall accuracy (OA), user's/producer's accuracy, average accuracy (AA), Cohen's Kappa, and occasionally F1 or mIoU. Critically, the review does not interrogate the spatial validation design of its sources: most cited studies use random/pooled sampling of pixels or parcels within a single scene (e.g., one study trains on 1% and tests on 99% but does so within the same images; another uses stratified 10-fold cross-validation on NDVI features). On the leakage spectrum this corpus sits almost entirely at the **pooled/random-split-within-one-scene** end. A few exceptions reach toward genuine generalization: ref. [33] (PSE+LTAE) models inter-annual crop rotation, ref. [38] tests a Landsat GAN on a new county (OA dropping from 86% to 81%), and ref. [32] uses multi-year training to reduce year-specific dependence. The review treats high in-scene OA as the headline signal and does not flag in-region-k-fold-disguised-as-OOS risk.

## Key Findings

- Of 36 studies, DL outperformed classical ML in all but one; the single ML win was attributed to a small training set (the review's clearest acknowledgment that DL's advantage is data-quantity-contingent).
- CNNs and LSTM/Bi-LSTM dominate; hybrids (CNN-LSTM, CNN-transformer, 3D-CNN+LSTM "CropNet", CNN-RF) and temporal-attention encoders (PSE+LTAE, two-stream TCN) report the highest in-scene OA, frequently >94% and as high as 99%.
- Sentinel-2 is the most-used source; optical-SAR fusion (Sentinel-1 + Sentinel-2) repeatedly improves OA over single-source optical, especially in cloud-prone regions and for mountainous/heterogeneous terrain.
- Class count matters: performance degrades when classes share phenology, and adding non-crop classes (water, built-up, barren) generally raises overall accuracy.
- Large training datasets are repeatedly identified as a precondition for DL success; data augmentation and transfer learning are the standard remedies for limited samples.
- Interpretability is noted as an open need; only isolated cited works (e.g., Grad-CAM on UAV imagery) attempt explainability.

## Relevance to Our Crop-Classification Study

This review is the canonical "landscape" citation for our manuscript and simultaneously a foil for our central thesis. Its corpus demonstrates the very pathology we critique: near-universal reliance on in-scene OA/Kappa, with almost no spatially disjoint holdout evaluation, yet headline accuracies routinely reported at 94-99%. When a review of 36 papers can conclude that dense DL "tends to outperform" ML while measuring transfer in essentially zero cases, that silence is itself a finding our paper foregrounds — these scores are likely upward-biased by spatial autocorrelation between train and test pixels/parcels within a single tile.

Three threads connect directly to our argument. First, the corpus's strongest reported generalization signals come from sparse, feature-selecting, or temporally-structured models (PSE+LTAE rotation modeling, CNN-RF hybrids, multi-year ANN training), consistent with our finding that sparse tree-like and attentive models transfer best. Second, the recurring "DL needs large training data" caveat is the flip side of our overfitting observation: dense temporal/patch nets that excel in-scene are exactly those most exposed to training-region collapse out-of-sample. Third, the review's emphasis on non-crop classes and class-phenology confusion maps onto our lucerne/medics-dominated, imbalanced setting where macro-F1 (not OA) is the honest metric.

The review also catalogs the architecture families we benchmark (CNN-BiLSTM, L-TAE, TempCNN, 3D-CNN, plus the tree/TabNet classical side), giving us a citable provenance for each. Its optical-SAR fusion emphasis is a useful contrast point: our study is deliberately optical-only (`B2`, `B6`, `B11`, `B12`, `EVI`, `hue`, no SAR), so we can position our work as showing that careful evaluation, not added sensors, is the lever that re-ranks models.

## Evaluation Caveats

- **Inherited, unexamined protocols.** The review does not standardize or critique the validation design of its sources. Reported OA/Kappa values cannot be compared across studies and almost certainly conflate in-scene memorization with genuine skill.
- **Spatial leakage unaddressed.** The corpus is dominated by random/pooled within-scene splits; same-field and adjacent-pixel contamination between train and test is not flagged. No source is evaluated on a spatially disjoint tile or with FID-disjoint splitting as a stated criterion.
- **Class-imbalance artifacts.** Most cited metrics are OA/AA/Kappa; balanced per-class F1 is rare. High OA in imbalanced settings (as in our lucerne-heavy data) can mask majority-class dominance — a distortion the review does not quantify.
- **Cross-year / cross-sensor transfer barely measured.** Only a handful of cited works test inter-annual or new-region generalization, and even those report modest drops; the corpus provides essentially no evidence on the train-region-to-holdout collapse our study documents.
- **Publication-bias toward high scores.** A 2020-2022 keyword-screened review of "successful" DL papers will systematically over-represent optimistic in-scene results, reinforcing the misranking problem our manuscript exists to correct.
- **Scope exclusions.** Segmentation and non-DL methods are excluded, so the review cannot speak to the classical-vs-DL transfer comparison that is the heart of our study.
