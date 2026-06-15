# Multimodal Deep Learning Based Crop Classification Using Multispectral and Multitemporal Satellite Imagery

**Citation:** Krishna Karthik Gadiraju, Bharathkumar Ramachandra, Zexi Chen, Ranga Raju Vatsavai (2020). "Multimodal Deep Learning Based Crop Classification Using Multispectral and Multitemporal Satellite Imagery." In *Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)*, pp. 3234–3242. DOI: `10.1145/3394486.3403375` (verified via Crossref).

## Objectives

Classify six crop types (corn, soy, cotton, spring wheat, winter wheat, barley) by *fusing* two complementary modalities: a **spatial-spectral stream** from very-high-resolution single-date imagery and a **temporal stream** from coarse-resolution multitemporal imagery. The motivating problem is crop classification's high intra-class variability (same crop looks different across its growth cycle) and low inter-class variability (corn vs soy, spring wheat vs barley look alike). VHR imagery captures spatial detail but is acquired ~once a year; high-temporal MODIS captures phenology but at 250 m. The thesis: jointly exploiting both beats either alone.

## Methods

Two-stream late-fusion network:

- **Spatial stream** (`F1`): a pretrained CNN backbone (VGG16 chosen over ResNet50/DenseNet201 after grid search) on **1 m NAIP, 3 bands (RGB), single date**, 240×240 patches, plus global average pooling and FC layers.
- **Temporal stream** (`F2`): one of t-LSTM / t-biLSTM / t-1D-CNN on a **23-point biweekly MODIS NDVI** time series (250 m, full year); t-LSTM chosen for parameter economy.
- **Fusion** (`H`): late fusion by concatenation or averaging of the two FC outputs.
- **Classifier** (`G`): MLP (ST-C, ST-A) or, best, an **SVM (ST-SVM)** over the softmax outputs of the two streams.

Dataset: >60,000 points across US states, labels from the USDA CDL (30 m); 8×8 CDL patches matched to 240×240 NAIP patches (same 240 m footprint). Adam, softmax cross-entropy, early stopping, ImageNet-pretrained backbones (unfrozen). Baselines reimplement Zhong et al. (z-LSTM, z-1D-CNN) plus RF and SVM on NDVI.

Sensor/feature design: optical-only, **two heterogeneous optical sources** (NAIP RGB VHR + MODIS NDVI coarse-temporal), no SAR. Granularity is **patch/point-level** — each labeled instance is a 240 m footprint with an associated NDVI series; not field-parcel aggregation. Raw imagery for the spatial stream; a single engineered index (NDVI) for the temporal stream.

**Evaluation protocol:** **random train/validation/test split** of the >60,000 collected patches (e.g. corn 6737/2245/2247). Care was taken that *patches do not spatially overlap*, but splitting is otherwise random across patches drawn from the same handful of states/crops — there is **no spatially disjoint holdout region**, no cross-year test. Patches of the same crop come from the same source states (corn/soy from Iowa+Illinois, barley from Montana), so train and test share geography and acquisition conditions. Metrics: **test accuracy, Cohen's Kappa, and macro-average F1** — a balanced metric is reported alongside accuracy, which is good given the moderately uneven class counts (barley ~half the size of corn).

## Key Findings

- Fusion wins: best model **ST-SVM reaches accuracy 98.41, Kappa 98.08, macro-F1 98.44**, versus best single-stream spatial (VGG16, 95.55 / 94.62 / 95.44) and best temporal (t-1D-CNN, 95.65 / 94.75 / 95.89). The authors frame this as a ~60% reduction in prediction error over the state-of-the-art single-modality baseline.
- The dominant confusions in single-modality models are exactly the spectrally/temporally similar pairs (corn↔soy, spring wheat↔barley); fusion via ST-SVM substantially clears them.
- A *non-linear* fusion classifier (SVM over the two streams' softmax probabilities) beat linear/MLP fusion (ST-C, ST-A) — the best combiner is again a margin-based classifier rather than another dense layer.
- Purely temporal baselines (RF, SVM, z-1D-CNN on NDVI) were all within ~2 points of each other (~95 accuracy), which the authors attribute to the task's simplicity and limited data — i.e. on this easy, in-region split the model choice barely mattered.

## Relevance to Our Crop-Classification Study

This is a crop-specific, multimodal, dense-deep-net counterpoint to our study, and several details map onto our argument. First, the **near-saturated metrics (98%+ macro-F1)** are a textbook symptom of the in-region, same-source split our manuscript warns about: train and test patches are drawn from the same states with no spatial holdout, so very high scores need not imply transfer. We would expect these numbers to drop sharply on a disjoint region — the misranking/collapse phenomenon our paper documents. Second, the authors' own observation that **RF, SVM, and a 1D-CNN all tie (~95%)** on the temporal stream undercuts any in-region ranking of architectures and is consistent with our claim that conventional validation fails to separate models. Third, the winning combiner is an **SVM over softmax outputs** — once more a sparse/margin-based model is the most effective consumer of dense features, echoing our finding that tree-like/feature-selecting models carry the transfer load while dense nets supply representations.

Design contrasts to note for our setup: their temporal feature is a *single* index (NDVI) at 250 m, whereas we use six bands/indices (`B2`,`B6`,`B11`,`B12`,`EVI`,`hue`) at Sentinel-2 resolution; their VHR spatial stream (NAIP RGB) has no analog in our optical-only Sentinel-2 pipeline (we have no 1 m source). Their granularity is fixed-size patches rather than our field-level (FID) aggregation, so their model does not exploit the variance-reduction lever we rely on. The two-stream fusion idea is orthogonal to our within-Sentinel-2 comparison but reinforces that combining heterogeneous evidence reduces the corn/soy-type confusion that also plagues our wheat/barley distinctions.

## Evaluation Caveats

- **Random patch split, same source geography — not out-of-sample.** No spatially disjoint holdout tile and no cross-year test. Non-overlap of patches prevents pixel-identical leakage but not regional/atmospheric domain sharing; reported 98% almost certainly overstates transferable performance.
- **Spatial leakage at the landscape scale:** each crop is sourced from a fixed small set of states, so train and test share climate, soil, and acquisition dates — the classifier can exploit region-specific cues that would not generalize.
- **Possible label/footprint coupling:** labels come from the 30 m CDL while the NDVI series is from the same 250 m MODIS pixel used to define the patch center, so spatial and temporal labels are tightly co-registered to one footprint, which can ease the task relative to operational settings.
- **Easy 6-class problem with abundant data**; the authors themselves note baselines saturate, limiting what the architecture comparison reveals about hard, imbalanced, transfer settings like ours (9 classes, majority lucerne/medics).
- **Silences:** no spatial transfer, no cross-year robustness, no evaluation when one modality is missing (acknowledged as future work), and no training-set-size ablation. The paper does not test whether multimodal fusion's in-region advantage survives a domain shift — the exact gap our study probes.
