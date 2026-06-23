# Enhancing Crop Type Mapping in Data-Scarce Regions Through Transfer Learning: A Case Study of the Hexi Corridor

**Citation:** Mai, J., Feng, Q., Fu, S., Wang, R., Zhang, S., Zhang, R., & Liang, T. (2025). Enhancing Crop Type Mapping in Data-Scarce Regions Through Transfer Learning: A Case Study of the Hexi Corridor. *Remote Sensing*, 17(9), 1494. DOI: `10.3390/rs17091494` (verified against Crossref; title and authors match). Open access (MDPI, CC BY). Briefer written from the full PDF.
**BibTeX key:** `mai2025enhancing`

## Objectives

The paper tackles crop-type mapping in a **data-scarce target region** (the Hexi Corridor, an arid oasis-agriculture belt in Gansu, northwest China) where local ground-truth labels are sparse and expensive. The strategy is **instance-based transfer learning**: train on abundant, high-confidence labels from a *data-rich source domain* — the USDA **Cropland Data Layer (`CDL`)** over the US Midwest/Great Plains (Iowa, Missouri, Kansas, Minnesota, North/South Dakota) — and transfer that knowledge to the Chinese target domain.

Three explicit objectives: (1) assess the feasibility of **directly transferring `CDL`-trained models** to Hexi with no local labels; (2) measure how **incrementally adding target-domain labels** improves accuracy; (3) produce a 2022 crop map for the Hexi Corridor.

In our manuscript's terms this is a **cross-domain (indeed cross-continental) transfer** study for crop classification — one of the more on-point comparators in the corpus, and a concrete instance of the *instance-based transfer* technique catalogued in the Ma et al. (2024) review (`Transfer learning in environmental remote sensing.md`).

## Methods

**Source vs. target domains.**
- **Source:** `CDL` 2022, US Midwest/Great Plains. High-confidence pixels (confidence mask `>95%`), patches `<100` pixels removed; training points randomly generated and visually validated.
- **Target:** Hexi Corridor, China, 2022 — different continent, arid climate, high-elevation Qinghai–Tibet plateau margin, fragmented oasis farmland. This is a *very large* geographic domain shift, far larger than our single-province inter-tile holdout.
- **Domain-gap quantification:** Kolmogorov–Smirnov tests on the feature distributions are all `p < 0.05` (domains are statistically distinct). NDVI phenology similarity measured by Dynamic Time Warping: maize is most similar (DTW `1.09`), then spring wheat `2.21`, alfalfa `2.36`, canola `2.38`, oats `2.51`.

**Crops and class imbalance (Table 1).** Five classes: maize, alfalfa, oats, rapeseed (canola), spring wheat.
- Source (`CDL`) sample counts are large and fairly balanced: maize 89,550; alfalfa 172,959; oats 117,949; rapeseed 89,818; spring wheat 174,611.
- **Target (Hexi field survey) is severely imbalanced and tiny:** maize 893; alfalfa 1,272; oats 161; rapeseed/canola **74**; spring wheat 142. Alfalfa and maize dominate; canola is the rarest by far.

**Sensors and features.** Optical + SAR fusion: **Sentinel-1** (C-band SAR, `VV`+`VH`, IW mode, monthly max composites), **Sentinel-2** (all bands, weekly max composites), **Landsat-8** (monthly max composites), for the full 2022 year. Inputs = all Sentinel-2 and Landsat-8 bands + Sentinel-1 `VV`/`VH` + **16 vegetation indices** (NDVI, EVI, SAVI, NDRE, MSI, GNDVI, CVI, PSRI, kNDVI, VGCI, RVI, NDWI, etc.). Features are **engineered multi-temporal composites** — per-date band/index values across the year (feature naming `Satellite_Index/Band_TimeStep`, e.g. `s2_VGCI_17` = Sentinel-2 VGCI at week 17), yielding a high-dimensional *tabular* per-pixel vector. This is **xr_fresh-adjacent**: stacked per-period statistics fed to tree models, not raw sequences fed to a sequence encoder.

**Models / experiments (Table 2).** All classifiers are **tree-based / boosting**; no deep temporal or patch networks are tested.
- **Experiment 1 — naive transfer (source only, zero target labels):** `RF_naive`, `XGBoost_naive`, `DT_naive`.
- **Experiment 2 — transfer learning (source + incremental target):** `RF_transfer` (`warm_start`, add trees on target), `XGBoost_transfer` (`xgb_model` incremental boosting), `TrAdaBoost_RF`, `TrAdaBoost_DT`. **TrAdaBoost** is instance reweighting: down-weight misclassified source instances, up-weight target instances each boosting round (`adapt` library).
- **Experiment 3 — local training (target only):** `RF_local`, `DT_local`.

**Evaluation protocol (load-bearing).** Two things must be separated:

1. **The source→target transfer direction is a genuine, leakage-free, cross-continental spatial transfer** (US `CDL` → Hexi). This sits at the far "geographic domain shift" end of our spectrum — *stronger* spatial separation than our holdout tile.

2. **But the within-target accuracy is measured on a *random* split of target ground-truth points, not a spatially disjoint sub-region and not a field-wise/parcel split.** Specifically: 30% of the Hexi field-survey samples are held out as the "independent test set"; the remaining 70% are incrementally fed in as target training data; 20% of target is a validation set; 5-fold CV stabilises training. All accuracy numbers (Tables 4–5) are computed on that random 30% point holdout *within* the target domain. Once target labels are incorporated (Experiment 2/3), this is an **in-region random split** — the very protocol our manuscript argues overstates deployable accuracy. Mitigating factor: the target labels are **field-survey points, one per regularly-shaped plot** (collected to guarantee clean `10 m` pixel coverage), so it is closer to one-point-per-field than to dense pixel sampling — pixel-adjacency leakage is limited, but it is still not a spatially-held-out region. The clean cross-domain number is the **Experiment 1 source-only** result.

3. **Cross-year robustness is tested** (a strength): the `RF_transfer` model trained on 2022 is independently validated on a **2023** dataset (Supplementary Tables S1–S2).

**Metrics (a strength).** Confusion-matrix–based: overall accuracy (OA), recall, class-mean recall, precision, **Cohen's Kappa**, and **macro-F1, weighted-F1, and per-class F1** — i.e., balanced metrics are reported alongside OA, unlike many crop papers.

## Key Findings

All numbers from Table 4 (target 30% random test set).

**Naive transfer (Experiment 1, zero target labels) — the cross-domain "naked transfer" regime:**
- `XGBoost_naive`: OA **0.7833**, class-mean recall 0.4758, **macro-F1 0.4480**, weighted-F1 0.7629, **Kappa 0.6009** (best naive model).
- `RF_naive`: OA 0.6588, macro-F1 0.4325, weighted-F1 0.7166, Kappa 0.5103.
- `DT_naive`: OA 0.5079, macro-F1 0.3460, Kappa 0.3365 (worst).
- (Abstract/Conclusion headline the source-only OA as "up to 73.88%" with per-class accuracies maize 88.97%, alfalfa 85.23%; Table 4's clean per-model OA for `XGBoost_naive` is 0.7833.)

**The class-imbalance artifact is stark under naked transfer:** weighted-F1 sits at 0.72–0.76 while **macro-F1 collapses to 0.43–0.45**. Overall/weighted numbers look usable, but the average across classes is poor — staple crops (maize, alfalfa) transfer well while minority crops (oats, canola, spring wheat) are largely misclassified into the majority classes (confusion matrices, Figure 7).

**Transfer learning with full target data (Experiment 2):**
- `RF_transfer` is best: OA **0.9226**, class-mean recall 0.8027, **macro-F1 0.8431**, weighted-F1 0.9202, **Kappa 0.8723**.
- `TrAdaBoost_RF`: OA 0.9186, macro-F1 0.8271, Kappa 0.8651.
- `XGBoost_transfer`: OA 0.9029, macro-F1 0.7698, Kappa 0.8416.
- `TrAdaBoost_DT`: OA 0.9094, macro-F1 0.7874, Kappa 0.8484.
- All transfer models exceed 0.90 OA; macro-F1 jumps ~0.40 points once target labels are added.

**Local training (Experiment 3, target only):** `RF_local` OA 0.8990 / macro-F1 0.7641 / Kappa 0.8399; `DT_local` OA 0.7624 / macro-F1 0.5967. Every Experiment-2 transfer model **beats** the local model, supporting the transfer-learning claim. (Note: local training is hamstrung by the tiny minority-class counts.)

**Per-class weighted F1 (Table 5, Experiment 2, full target):** maize and alfalfa ~0.93–0.95 across all models; **canola is the worst** (best 0.6111 for `RF_transfer`, down to 0.4706 for `XGBoost_transfer`) — directly tied to its 74 target samples. Oats and spring wheat intermediate (~0.74–0.88).

**Progressive target incorporation (Figure 5):** accuracy rises monotonically as target fraction grows 0→100% (e.g. `RF_transfer` and `TrAdaBoost` curves climb into the low-0.90s); transfer models stay above the local models throughout.

**Feature importance (Figure 8, `RF_transfer`):** of the top 200 features, **191 are Sentinel-2, 7 Landsat-8, only 2 Sentinel-1**. Top features are VGCI, kNDVI, NDVI at week 17 (greening period). **Optical (especially red-edge `B5`–`B7`, SWIR `B11`/`B12`, and indices) dominates; SAR backscatter contributes almost nothing** to discriminating these crops.

**Tree models beat instance-transfer boosting.** `RF_transfer` and `XGBoost`-family are more robust to class imbalance and domain shift than `TrAdaBoost`, which the authors attribute to TrAdaBoost's sensitivity to hyperparameters, small target samples, and source intra-class heterogeneity.

## Relevance to Our Crop-Classification Study

This is a **directly relevant cross-domain transfer comparator**, joining `pankajakshan2026`, `rustowicz2019`, and `cropformer2023` as the corpus's true-spatial-transfer references, and pairing with the Ma et al. (2024) transfer-learning review as a concrete instance of instance-based transfer.

- **Validates our central evaluation argument — and illustrates the exact pitfall.** The source→target direction is a clean cross-continental transfer, but the headline 0.92 OA is computed on a *random in-target point split*. The honest cross-domain number is the Experiment-1 naked transfer (`XGBoost_naive` OA 0.78 / macro-F1 0.45), and the ~0.40 macro-F1 jump from naked transfer to target-augmented training quantifies how much "accuracy" is really in-region interpolation. We can cite this as a textbook example of why one must distinguish leakage-free transfer from in-region splits — the conflation our paper exists to correct.
- **The imbalance artifact is a quotable data point (rule 3).** Under naked transfer, weighted-F1 (0.72–0.76) looks fine while macro-F1 (0.43–0.45) collapses — exactly the majority-class masking we warn about (lucerne dominance in our data). Their reporting of macro-F1, per-class F1, and Kappa alongside OA is the balanced-metric standard we advocate; cite it approvingly.
- **Same model-class verdict, complementary angle.** Their winners are tree/boosting models (`RF`, `XGBoost`), with instance-transfer boosting (`TrAdaBoost`) *underperforming* plain RF — and they never test dense temporal/patch deep nets. This corroborates our "sparse, tree-like models transfer best" thesis, though it is not a head-to-head against deep sequence models the way ours is. Useful as supporting evidence that gradient-boosted/bagged trees are the transfer-robust workhorse.
- **Naked-transfer regime is the shared corner.** Our paper measures **zero-adaptation transfer** (train source, predict holdout, no target labels). Their Experiment 1 is the closest analogue and shows even a strong multi-sensor pipeline pays a large macro-F1 penalty before any target labeling — reinforcing that model-class/inductive-bias choice matters most precisely there.
- **Shared crop-signature insight.** Maize and alfalfa transfer best in both studies; **alfalfa = lucerne, our dominant and well-transferring class.** Minority crops fail under transfer in both — a usable point about which classes survive a holdout.
- **Feature-design overlap and contrast.** They confirm **optical Sentinel-2 (red-edge + SWIR + indices) carries the signal and SAR is near-useless** for these crops (191/200 vs 2/200 top features) — direct support for our optical-only Sentinel-2 design on bands `B2`,`B6`,`B11`,`B12`,`EVI`,`hue` (note overlap: their top features include `B5`/`B6` red-edge and `B11`/`B12` SWIR). Their feature scheme is xr_fresh-adjacent (per-date index composites), so it is a reasonable robustness reference for engineered-temporal-feature + tree pipelines. They fuse SAR; we do not — and their result says we lose little by skipping it.
- **They go one step further than us on cross-year.** They validate 2022→2023 (supplement); our design is single-year. A clean citation for the future-work paragraph.

## Evaluation Caveats

- **Do not compare absolute numbers to ours.** Their 0.92 OA / 0.84 macro-F1 are from an **in-target random point split** augmented by source transfer, not a spatially disjoint holdout like ours. Cite the *protocol* (cross-continental instance transfer; random in-target test) and prefer the Experiment-1 naked-transfer figures (OA 0.78 / macro-F1 0.45) when drawing the cross-domain analogy.
- **Within-target split is not field-wise or spatially partitioned.** The 30% test set is a random subset of field-survey points. Leakage is limited by one-point-per-plot sampling, but there is no spatially-held-out sub-region, so within-target metrics measure interpolation, not intra-region transfer.
- **No deep-learning comparator.** The study cannot adjudicate the tree-vs-deep transfer question that is our paper's core; it shows trees + instance transfer work, leaving the dense-deep-net failure mode untested. Corroborating, not decisive.
- **Severe minority-class fragility.** Canola has only 74 target samples; its F1 (0.47–0.61) and the unstable `TrAdaBoost` behaviour are small-sample artifacts, not general conclusions.
- **Spatial resolution.** Authors flag `10 m` Sentinel-2 as too coarse for Hexi's fragmented fields and suggest PlanetScope/WorldView — a resolution caveat that also bounds the comparability of their boundary-level results.
