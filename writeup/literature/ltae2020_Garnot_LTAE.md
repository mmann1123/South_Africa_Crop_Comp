# Lightweight Temporal Self-Attention for Classifying Satellite Images Time Series

**Citation:** Vivien Sainte Fare Garnot, Loic Landrieu (2020). "Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series." In *Advanced Analytics and Learning on Temporal Data (AALTD 2020)*, Lecture Notes in Computer Science, Springer, pp. 171–181. DOI: `10.1007/978-3-030-65742-0_12` (verified via Crossref title match). The PDF is the arXiv preprint `2007.00586`.

## Objectives

Introduce the **Lightweight Temporal Attention Encoder (L-TAE)**, a compact self-attention module for encoding satellite image time series (SITS) into a single per-parcel feature vector for crop classification. It is a parameter- and compute-efficient redesign of the authors' earlier Temporal Attention Encoder (TAE), targeting continent-scale processing where the temporal encoder must be cheap. Goal: match or beat heavier temporal models (TAE, GRU, TempCNN, Transformer, ConvLSTM) at a fraction of the parameters and FLOPs.

## Methods

L-TAE adapts multi-headed self-attention with three efficiency moves: (i) **channel grouping** — the `E` input channels are split across `H` heads (group size `E/H`), so each head operates on its own channel slice rather than the full embedding, removing `H` from the key/output complexity; (ii) **query-as-parameter** — each head's master query `q_h` is a learned model parameter instead of the output of a linear layer, cutting parameters further; (iii) values are bypassed (`v = e`), only keys are linearly projected. A positional encoding of elapsed days (characteristic scale `τ = 1000`) is added per group. Attention masks are the scaled softmax of key·query; head outputs are concatenated to size `E` and passed through an MLP. The L-TAE is trained end-to-end inside a spatio-temporal classifier: a Pixel-Set Encoder (PSE) maps each date's parcel pixels to an embedding, the L-TAE aggregates over time, an MLP decodes class logits.

Sensor/feature design: optical-only Sentinel-2, **10 spectral bands**, 24 acquisitions Jan–Oct, 10 m. Granularity is **parcel/field-level** — the PSE pools a *set* of pixels per parcel per date (order-invariant), and the temporal encoder produces one vector per field. This is a learned dense temporal encoder over raw multispectral sequences, not engineered features.

**Evaluation protocol:** **5-fold cross-validation on the single open-access `Sentinel2-Agri` dataset** (191,703 parcels, 20 crop classes, one French region, one season). This is **within-region, within-year field-level k-fold** — there is no spatially disjoint holdout tile, no cross-year, no cross-sensor test. Folds are over parcels within the same geography; spatial autocorrelation between neighboring parcels across folds is not controlled. Metrics: Overall Accuracy (OA) and **mean IoU (mIoU)**, with the authors explicitly preferring mIoU because the dataset is unbalanced (4 classes = 90% of samples) — a balanced-metric choice we note approvingly.

## Key Findings

- At ~150k parameters matched across methods, PSE+L-TAE reaches **OA 94.3, mIoU 51.7**, beating PSE+TAE (50.9), CNN+GRU (48.1), CNN+TempCNN (47.5), Transformer (42.8), ConvLSTM (42.1), and Random Forest (32.5 mIoU). OA differences are small; the mIoU gains are the meaningful ones.
- Drastic parameter efficiency: a **9k-parameter** L-TAE beats a 110k TAE, a 700k+ TempCNN, and a 3M GRU on mIoU. Channel grouping plus query-as-parameter is the lever.
- Attention masks are **class-specific and temporally sparse**: different heads focus on narrow, distinct time windows (e.g. Spring vs Summer cereals), and masks adapt to crop phenology — interpretable temporal feature selection.
- Random Forest is by far the weakest on this benchmark (mIoU 32.5), under in-region 5-fold CV.

## Relevance to Our Crop-Classification Study

We use L-TAE as one of our deep temporal encoders, so this paper documents the exact architecture and its *original within-region evaluation*. Critically, every number here is produced under in-region field-level 5-fold CV on one French tile and one season — precisely the protocol our manuscript argues *misranks* models. On that benchmark L-TAE tops dense competitors and Random Forest finishes last (mIoU 32.5), the opposite of our headline finding that sparse tree-like models transfer best and dense temporal nets collapse out-of-sample. This is a direct illustration of the misranking thesis: the validation regime that crowns L-TAE and buries RF is the one we replace with a spatially disjoint holdout tile (`34S_20E_259N`), under which the ordering reverses.

Two nuances cut both ways. First, L-TAE's attention is *temporally sparse and class-specialized* (Fig. 3) — a soft inductive bias that may give it more transfer robustness than fully dense nets (CNN-BiLSTM, 3D-CNN); our results should be read against whether L-TAE collapses as hard as the others or partially resists. Second, the **field-level PSE aggregation** is itself the variance-reduction lever our paper identifies as recovering transfer for dense models — L-TAE is born field-level, so it already benefits from the pooling we apply to pixel-level dense models. The paper's preference for mIoU under heavy imbalance mirrors our use of macro-F1/Kappa for a lucerne/medics-dominated problem.

## Evaluation Caveats

- **In-region field-level 5-fold CV only**, single region, single year, single sensor — flag as *not* an out-of-sample test. No spatially disjoint holdout, the benchmark cannot detect the spatial overfitting our study targets.
- **Spatial leakage not controlled across folds**: parcels are split by ID but neighboring fields in the same landscape can straddle folds, so reported mIoU/OA likely embed spatial autocorrelation advantage.
- **Single dataset.** All competitor numbers (taken from Garnot et al. CVPR 2020) come from the same `Sentinel2-Agri` tile; the ranking is benchmark-specific and, as our work shows, does not survive a domain shift.
- **Metric note (positive):** mIoU is reported precisely because of 4-class/90% imbalance — a balanced metric, unlike accuracy-only papers; OA alone would have hidden the minority-class story.
- **Silences:** no cross-tile transfer, no cross-year robustness, no training-set-size ablation, no calibration, and no test of whether attention sparsity translates into better generalization under covariate shift — the L-TAE's transferability (vs its in-region dominance) is exactly the open question our manuscript answers empirically.
