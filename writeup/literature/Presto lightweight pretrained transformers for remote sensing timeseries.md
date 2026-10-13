# Lightweight, Pre-trained Transformers for Remote Sensing Timeseries (Presto)

**Citation:** Gabriel Tseng, Ruben Cartuyvels, Ivan Zvonkov, Mirali Purohit, David Rolnick, and
Hannah Kerner (2023; v4 revised 5 Feb 2024). *Lightweight, Pre-trained Transformers for Remote
Sensing Timeseries.* arXiv preprint `arXiv:2304.14065` [cs.CV] (labelled "Preprint" in the PDF; no
journal or proceedings venue is printed). DOI: not found in PDF. A Crossref title search
(`query.title=Lightweight, Pre-trained Transformers for Remote Sensing Timeseries`, rows=3) returned
no matching record (top hits were unrelated 2024-2026 papers), so no DOI is recorded. Not present in
`writeup/literature/references.bib` or `writeup/literature/README.md` at the time of writing.
Code: `https://github.com/nasaharvest/presto`.
**Suggested BibTeX key:** `tseng2023presto`

## Objectives

The paper introduces Presto (Pretrained Remote Sensing Transformer), a self-supervised foundation
model that operates on *single-pixel* multi-sensor time series rather than on image patches. The
authors argue that existing remote sensing self-supervised models borrowed from computer vision
(SatMAE, ScaleMAE, ConvMAE, SeCo, GASSL) have four deployment problems: they ignore the temporal
dimension that matters for crops and seasonal land cover, they ingest a single sensor (usually
Sentinel-2 only), they are large and expensive (ViT-Large is about 300M parameters), and they cannot
natively handle point or irregular-polygon labels or missing inputs.

Presto's stated goals are four operational requirements:

1. Computational efficiency for wall-to-wall mapping over millions of pixels (the model has about
   0.4M encoder parameters, up to 1000x fewer than ViT baselines).
2. Ability to ingest inputs of varying shape (with or without spatial context, one or many timesteps).
3. Ability to ingest many raw and derived products (S1, S2, ERA5, NDVI, DEM, Dynamic World,
   coordinates).
4. Robustness to missing data (missing months, missing sensors), achieved through structured masking.

The central empirical claim is that a tiny pixel-timeseries model, pre-trained globally with a masked
autoencoder objective, is competitive with or better than much larger image models and task-specific
state-of-the-art models across a heterogeneous suite of downstream tasks, and that it can be used
either as a frozen feature extractor for simple models (linear/logistic regression, random forest)
or fine-tuned end to end.

## Methods

**Pre-training data.** 21,535,000 pixel samples at 10 m, drawn with the Dynamic World stratified
sampling scheme (Earth split into three regions, then ecoregions, stratified by land-cover class).
Each of the sampled 510 x 510 tiles contributes 2,500 grid-sampled pixels. Each pixel has 24 monthly
timesteps spanning 2020-2021, from which contiguous 12-month windows are drawn. The sample map
(Figure 6) is global and includes southern Africa, so the unlabeled pre-training corpus plausibly
covers Western Cape cropland, although for 2020-2021 rather than any specific competition year.

**Inputs (per monthly timestep, 15 dynamic channels plus categorical).**

| Channel group | Bands | Notes |
|---|---|---|
| Sentinel-1 | `VV`, `VH` | SAR, cloud-independent |
| S2 RGB | `B2`, `B3`, `B4` | one token |
| S2 Red Edge | `B5`, `B6`, `B7` | one token |
| S2 NIR 10 m | `B8` | one token |
| S2 NIR 20 m | `B8A` | one token |
| S2 SWIR | `B11`, `B12` | one token |
| NDVI | from `B4`, `B8` | derived |
| ERA5 | precipitation, 2 m temperature | monthly reanalysis |
| Dynamic World | land-cover class (monthly mode) | categorical, embedded |
| Topography (static) | SRTM elevation, slope | static token |
| Location (static) | 3D Cartesian from lat/lon | never masked |

The 60 m Sentinel-2 bands are dropped. Each channel group at each timestep becomes one token via a
group-specific linear projection; encodings concatenate a sinusoidal position encoding, a cyclic
month encoding (sin/cos of month/12), and a learned channel-group encoding.

**Architecture and pre-training.** Masked autoencoder (He et al. style) with encoder and decoder both
2 transformer layers, 8 heads, embedding size 128, MLP ratio 4 (about 0.81M parameters including the
decoder; the encoder alone is about 402K). Masking ratio 0.75, with the mask strategy sampled per
instance from four options: random (timestep x channel), whole channel groups, contiguous timesteps,
and random timesteps. Loss is MSE on continuous values plus a count-balanced cross-entropy on the
Dynamic World categories (lambda = 2). Trained 20 epochs, batch size 4096, AdamW, cosine schedule,
about 43 hours on one V100. Pre-training validation used CropHarvest FAO crop-class classification,
excluding the downstream evaluation points.

**Downstream adaptation (three modes).**

`Presto_R`: frozen encoder, global-pooled tokens fed to linear or logistic regression (only about 129
adapted parameters for CropHarvest).

`Presto_RF`: frozen encoder, embeddings fed to a scikit-learn random forest with default
hyperparameters (`class_weight="balanced"` on CropHarvest).

`Presto_FT`: full fine-tuning of the encoder plus a linear head (AdamW, lr 3e-4, weight decay 0.05,
same hyperparameters on every task), using a slice of downstream training data for validation.

For image or parcel tasks, each pixel is encoded independently and the per-image representation is
the concatenation of the per-dimension mean and standard deviation of the pixel embeddings (a 256-d
vector), passed to the downstream classifier. This is a field/patch aggregation by pooling, conceptually
close to our mean-pooling and majority-vote aggregation by `FID`.

**Downstream tasks.**

| Dataset | Task | Region | Input [H,W,T,C] | Train n |
|---|---|---|---|---|
| CropHarvest Kenya | binary maize | Kenya | [1,1,12,18] | 1,345 |
| CropHarvest Brazil | binary coffee | Brazil | [1,1,12,18] | 203 |
| CropHarvest Togo | binary cropland | Togo | [1,1,12,18] | 1,319 |
| S2-Agri100 | crop-type classification | France (one S2 tile) | [5,5,24,10] | 1,500 |
| TreeSatAI | multi-label tree species | Germany | [6,6,1,2] or [6,6,1,11] | 45,337 |
| EuroSAT | 10-class land cover | Europe | [64,64,1,3] or [64,64,1,11] | 21,600 |
| Fuel moisture | regression | western USA | [1,1,3,19] | 1,578 |
| Algae blooms | regression | midwestern USA | [1,1,12,19] | 777 |

Baselines: random forest on raw inputs, MOSAIKS-1D (random temporal convolution features with RF or
regression heads, the authors' own adaptation), TIML (task-informed meta-learning, prior SOTA on
CropHarvest), linear regression, LightGBM and MLP (TreeSatAI benchmark baselines), SITS-Former
(pre-trained and random init) on S2-Agri100, and SatMAE / ScaleMAE / ConvMAE / SeCo / GASSL on
EuroSAT. All downstream heads are run with three seeds (0, 42, 84).

**Evaluation protocol:** Heterogeneous and task-dependent; there is no single protocol, and only some
tasks are spatial holdouts. The two regression tasks (fuel moisture and algae blooms) use what the
authors explicitly call a "geographically separated test set (a test set covering a different
geographic area than the training set)", deliberately replacing the 5-fold CV used by the original
fuel-moisture paper. These are genuine spatially disjoint holdouts, but they are regression tasks,
not crop classification. The CropHarvest Kenya/Brazil/Togo tasks use the CropHarvest benchmark's
predefined test sets; the Presto paper itself does not describe how those test points relate
spatially to the training points (the CropHarvest benchmark paper should be consulted before calling
them spatial holdouts), so in this briefer they are classed as benchmark-provided held-out sets of
unstated spatial separation. They are binary tasks within one country each, not cross-region crop-type
transfer. S2-Agri100 is a within-tile parcel holdout: S2-Agri comes from a single Sentinel-2 tile,
100 parcels per class are used for training and 100 for validation, and all remaining parcels are the
test set. This is parcel-disjoint (the analogue of our field-wise `FID` split) but explicitly
in-region, so it sits at the field-wise k-fold level of our spectrum, not the spatial-holdout level.
TreeSatAI and EuroSAT use dataset-provided splits (EuroSAT via Neumann et al. 2019), which are
random image-level splits within the same European/German domain, and these tasks are not
crop-type. No task is cross-year (pre-training is 2020-2021; downstream years are not discussed), and
no task trains a crop-type classifier in one region and tests it in a disjoint region with no target
labels, which is the setting of our holdout tile `34S_20E_259N`.

## Key Findings

**CropHarvest (Table 2, Table 12).** Mean binary F1 across the three tasks: random forest 0.441,
MOSAIKS-1D_R 0.738, TIML 0.802, `Presto_R` 0.835 (0.836 without Dynamic World), while adapting only
about 129 parameters versus 91K for TIML. Per task, `Presto_R` F1 is 0.816 (Kenya), 0.891 (Brazil),
0.798 (Togo). The raw-input RF baseline scores F1 0.000 on Brazil coffee (203 training samples,
heavily imbalanced) despite an AUC ROC of 0.941, a vivid example of a threshold/imbalance collapse
that AUC hides and F1 exposes. Mean AUC ROC: RF 0.803, MOSAIKS 0.806, TIML 0.890, Presto 0.917.

**Missing data robustness (Figure 3).** With only a subset of the 12 input months, `Presto_R`
recovers the full-12-month performance of TIML and MOSAIKS-1D on CropHarvest. Fuel moisture was run
with only 3 timesteps. Dropping Dynamic World leaves mean F1 essentially unchanged.

**Regression tasks with geographically separated test sets (Table 3, Table 10).** RMSE (fuel
moisture / algae): linear regression 28.20 / 0.850; random forest 23.84 / 1.249; MOSAIKS-1D_RF
28.75 / 0.972; `Presto_FT` random init 26.07 / 0.955; `Presto_FT` 25.28 / 0.815; `Presto_RF`
25.98 / 0.884. Note that the plain random forest is actually the best single model on fuel moisture
(23.84 vs 25.28); Presto wins on the authors' "mean difference from linear regression" aggregate
because RF is poor on algae blooms. On the spatial-holdout tasks, therefore, a tree baseline on raw
inputs is competitive with or better than the foundation model on one of two tasks.

**TreeSatAI (Table 4, Table 15).** `Presto_RF` (frozen encoder plus default RF) beats the
benchmark's LightGBM and MLP on weighted and micro F1 for both S1 (38.34 vs 11.86 weighted F1) and
S2 (55.29 vs 48.17), though LightGBM/MLP keep higher precision and comparable or better mAP with S2
(MLP weighted mAP 64.19 vs Presto 61.53). On S1, a randomly initialized fine-tuned Presto slightly
outperforms the pre-trained one, which the authors attribute to the input (single timestep, single
channel group) being far from the pre-training distribution.

**EuroSAT (Figure 5, Tables 5, 11, 13, 14).** kNN on Presto embeddings is competitive with
SatMAE/ScaleMAE/ConvMAE on RGB and better with multispectral input at low resolution, at orders of
magnitude fewer FLOPs. Fine-tuned accuracy is 0.953 (MS) and 0.849 (RGB), versus SatMAE ViT-Large
0.990 (MS) and 0.955 (RGB), so the large image models still win at full resolution. Performance
plateaus as resolution increases because mean/std pooling loses the minority-pixel signal for
classes like Highway and River (Section A.6 failure mode).

**S2-Agri100 crop types (Table 6, Table 16).** Pre-trained Presto: OA 68.89, kappa 0.58, macro-F1
40.41. Pre-trained SITS-Former: OA 67.03, kappa 0.56, macro-F1 42.83. Presto wins on OA and kappa
but loses on macro-F1, which is the balanced metric our study uses as primary. Randomly initialized
Presto collapses to OA 45.98 / macro-F1 27.45, so pre-training contributes about 13 macro-F1 points
in this small-label crop task.

**Ablations.** Combining all four structured masking strategies gives the best CropHarvest validation
F1 (0.665 vs 0.646-0.664 for single strategies). Pre-training helps on most tasks (EuroSAT accuracy
up to +0.1, algae RMSE down about 15%, S2-Agri100 roughly +50% relative accuracy) but hurts on
TreeSatAI-S1. Larger variants help modestly (width 256: F1 0.687 vs 0.665).

## Relevance to Our Crop-Classification Study

**Not a new true-spatial-holdout crop comparator.** Presto's only explicitly geographically separated
test sets are the two U.S. regression tasks (fuel moisture, algae blooms). Its crop tasks are
binary CropHarvest tasks with benchmark-defined test sets whose spatial separation the paper does not
describe, and S2-Agri100, which is a parcel-disjoint split inside a single Sentinel-2 tile, i.e.
the in-region field-wise protocol that our paper argues can misrank models. Presto therefore does not
re-rank crop-type models across a spatially disjoint region and should not be cited alongside
`pankajakshan2025` or `cropformer2023` (TL1/TL2) as a transfer comparator. It belongs instead in the
"transfer through pre-training / foundation models" strand of the introduction, the strand our
manuscript explicitly distinguishes from our no-target-access domain-generalization setting.

**Pre-training sits on the other side of our framing.** Presto's global unlabeled pre-training
corpus (21.5M pixels, Dynamic World sampling, including southern Africa) could contain imagery from
the very region of our holdout tile. Using Presto as a baseline would therefore partially violate
the "no target access" premise of our protocol (unlabeled target-domain imagery, though from a
different year), and any Presto result on `34S_20E_259N` would need that caveat. This sharpens the
reconciliation already used for `cropformer2023`: dense temporal transformers may transfer when
given a strong global prior, but that prior is a form of target-domain exposure our supervised
from-scratch models (L-TAE, TempCNN, CNN-BiLSTM, Transformer encoder) did not receive.

**Supports our sparse/tree-like transfer argument, indirectly.** Three details align with our thesis.
First, the recommended cheap adaptation route is a frozen encoder plus a default random forest
(`Presto_RF`) or logistic regression (`Presto_R`), i.e. the authors themselves put a tree or linear
model on top of the transformer rather than relying on end-to-end fine-tuning, and `Presto_RF`
beats `Presto_FT` on TreeSatAI-S2. Second, on the fuel-moisture spatial holdout, a plain random
forest on raw inputs has the best RMSE of all models. Third, the paper notes random forests "remain
competitive with state-of-the-art methods" and are what large-scale operational maps actually deploy.
Conversely, the CropHarvest RF baseline (F1 0.441, including 0.000 on Brazil) is a weak,
untuned, raw-input RF, so its loss to Presto is not evidence against a properly tuned GBDT on
engineered features like our XGBoost on `xr_fresh` statistics.

**Aggregation parallels.** Presto is a pixel model that obtains parcel/image predictions by pooling
pixel embeddings (mean plus standard deviation), and the authors identify this crude aggregation as
the main failure mode when the label signal lives in a minority of pixels. This mirrors our finding
that field-level aggregation is a substantive lever and that how pixel outputs are pooled to `FID`
matters.

**Feasibility as a baseline on AI4FoodSecurity South Africa.** Technically possible but non-trivial,
and not a like-for-like comparison with our optical-only feature set. Our pipeline uses `B2`, `B6`,
`B11`, `B12`, `EVI`, and `hue` (config `BANDS`), with months `05` and `06` excluded. Mapping to
Presto's channel groups: `B11` and `B12` fill the S2 SWIR group completely; `B2` is one of three
bands in the RGB group and `B6` one of three in the Red Edge group, so those tokens would be partial;
`EVI` and `hue` have no Presto channel at all; Presto's NDVI channel needs `B4` and `B8`, which we do
not carry. S1, ERA5, SRTM, and Dynamic World would all be absent unless re-exported (Presto's
missing-sensor masking is designed to tolerate their absence, and Table 2 shows Dynamic World can be
dropped). The two missing months fit the structured-masking design (Figure 3 shows robustness to
subsets of months), but Presto expects monthly composites, which our per-band monthly parquets in
`DATA_DIR` approximately provide. A fair Presto baseline would most likely require re-extracting the
full 10-band Sentinel-2 stack (plus optionally S1 and SRTM) for the AI4FoodSecurity fields, encoding
each pixel with the frozen encoder, pooling to `FID`, and fitting a logistic regression or RF head
on the two training tiles before predicting the holdout. That would also break our "optical-only,
same six features for every model" controlled design, so it is best framed as a supplementary or
future-work comparison. Compute is not a barrier: the encoder is about 0.4M parameters and runs on a
CPU or a single GPU (our RTX 3080 Ti is ample).

**Metric discipline.** The S2-Agri100 result (Presto higher OA and kappa, lower macro-F1 than
SITS-Former) and the CropHarvest RF AUC-vs-F1 discrepancy are useful illustrations for our argument
that macro-F1 should be primary, since OA and threshold-free metrics can hide minority-class failure.

## Evaluation Caveats

The crop-relevant headline numbers are not transfer numbers. The CropHarvest F1 of 0.835 is the mean
of three binary tasks on benchmark-provided test sets whose spatial relationship to training data the
paper does not state, and the S2-Agri100 macro-F1 of 40.41 comes from a parcel-disjoint split inside
one Sentinel-2 tile. Neither is comparable to our spatially-disjoint-holdout macro-F1 on a
multi-class winter-crop problem; cite the protocol, not the score.

The only explicit geographic holdouts (fuel moisture, algae blooms) are regression tasks with
literature baselines the authors admit are not directly comparable (different inputs or private
test data), and on fuel moisture a raw-input random forest beats every Presto variant.

Baselines are uneven. RF and MOSAIKS heads use default scikit-learn hyperparameters, the CropHarvest
RF is run on raw inputs, and no tuned gradient-boosted tree (XGBoost, LightGBM with search) is
evaluated on any crop task. LightGBM appears only via the TreeSatAI benchmark's published numbers.
There is no comparison to temporal-specific supervised models like L-TAE, TempCNN, or TabNet.

Class imbalance is handled by `class_weight="balanced"` on CropHarvest and macro-F1 is reported for
S2-Agri100, but no per-class F1 is reported for any multi-class crop task, so minority-crop recall is
not visible.

Pre-training covers 2020-2021 globally; no cross-year evaluation is performed, and the paper does not
check whether downstream test regions overlap the pre-training sample locations, so leakage of
unlabeled target-domain imagery through pre-training is unexamined.

Sensor design differs sharply from ours: Presto assumes the full 10-band Sentinel-2 stack plus S1,
ERA5, SRTM, NDVI, Dynamic World, and coordinates, whereas our study is deliberately optical-only with
six bands/indices and engineered `xr_fresh` features. Gains attributed to Presto may partly reflect
the richer multi-sensor input rather than the architecture or pre-training per se.

Downstream heads use three seeds with standard errors, which is good practice, but the
pre-trained encoder itself is a single model, so variance from pre-training is not quantified.
The paper is an arXiv preprint (v4, Feb 2024) with no peer-reviewed venue or DOI identified.
