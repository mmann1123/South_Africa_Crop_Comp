# Crop-Classification Literature: Project-Centric Findings Synthesis

Internal decision-support synthesis for the Western Cape Sentinel-2 crop-classification study. Organized around our modeling choices, evaluation bar, and concrete action items — **not** a general review (that is [general_literature_review.md](<general_literature_review.md>)). Every number below is tagged by the evaluation protocol that produced it; numbers from different tiers are not comparable.

**Our study, in one line:** nineteen classical / deep / patch models on multi-temporal Sentinel-2 optical-only imagery (`B2 B6 B11 B12 EVI hue`, 10 months, May/June excluded), benchmarked twice — in-region field-wise CV on the training tiles (`34S_19E_258N`, `34S_19E_259N`) and on a spatially disjoint holdout tile (`34S_20E_259N`). Primary metric macro-F1; also Kappa, weighted-F1, log-loss. Five crops, lucerne/medics-dominated (43% train, 56% holdout). Central claim: in-region validation misranks models; spatial transfer reorders them; sparse, feature-selecting/tree-like inductive bias (gradient-boosted trees, TabNet via sparsemax masks) transfers, dense temporal/patch nets overfit.

## Citation keys → briefers

- `pankajakshan` → [Deep Architectures Fail to Generalize](<Deep Architectures Fail to Generalize: A Lightweight Alternative for Agricultural Domain Transfer in Hyperspectral Images.md>)
- `cropformer2023` → [Cropformer (Wang et al. 2023)](<cropformer2023_Wang.md>)
- `jin_gee` → [Smallholder maize, national-scale GEE](<Smallholder maize area and yield mapping at national scales with Google Earth Engine.md>)
- `rustowicz2019` → [Crop-type segmentation, Ghana/South Sudan](<rustowicz2019_CropType_Africa.md>)
- `mann_ethiopia` → [Crop-loss forecasting, Ethiopia](<Predicting high-magnitude, low-frequency crop losses using machine learning: an application to cereal crops in Ethiopia.md>)
- `ltae2020` → [L-TAE (Garnot & Landrieu)](<ltae2020_Garnot_LTAE.md>)
- `tempcnn` → [TempCNN (Pelletier et al.)](<Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series.md>)
- `penabarragan2011` → [Object-based crop ID (phenology)](<Object-based crop identification using multiple vegetation indices, textural features and crop phenology.md>) / [typeset](<Object-based crop identification using multiple vegetation indices, textural features.md>)
- `teixeira` → [CNN-RNN + red-edge indices](<Deep Learning Applications for Crop Mapping Using Multi-Temporal Sentinel-2 Data and Red-Edge Vegetation Indices: Integrating Convolutional and Recurrent Neural Networks.md>)
- `maize_kenya` → [Characterising (inter)cropped maize signatures](<Characterising maize and intercropped maize spectral signatures for cropping pattern classification.md>)
- `russwurm` → [Sequential recurrent encoders](<Multi-Temporal Land Cover Classification with Sequential Recurrent Encoders.md>)
- `convstar_sar` → [3D-ConvSTAR optical-SAR fusion](<Enhanced crop classification through integrated optical and SAR data  a deep learning approach for multi-source image fusion.md>)
- `nowakowski` → [Crop type mapping by transfer learning](<Crop type mapping by using transfer learning.md>)
- `saini2018` → [RF vs SVM, single-date Sentinel-2](<saini2018_RF_SVM_Sentinel2.md>)
- `hcrnn` → [Hierarchical CNN-RNN (Laibin)](<forests-14-01881.md>)
- `ienco2017` → [RNN/LSTM for SITS](<ieee2017_Ienco_RNN.md>)
- `cnnrf_hybrid` → [Optimal feature selection + CNN-RF](<Crop Classification Method Based on Optimal Feature Selection and Hybrid CNN-RF Networks for Multi-Temporal Remote Sensing Imagery.md>)
- `multimodal_naip` → [Multimodal NAIP+MODIS](<Multimodal Deep Learning Based Crop Classification Using Multispectral and Multitemporal Satellite Imagery.md>)
- `unet_kenya` → [U-Net segmentation, CV4A Kenya](<Satellite Imagery Analysis for Crop Type Segmentation Using U-Net Architecture.md>)
- `optsar_iran` → [Optical+SAR phenology fusion](<Crop classification based on phenology information by using time series of optical and synthetic-aperture radar images.md>)
- `bandar2024` → [Attention-BiLSTM + TempCNN](<bandar2024_BiLSTM_TCN.md>)
- `cai_dl` → [DL multi-temporal (Landsat EVI)](<Deep learning based multi-temporal crop classification☆.md>)
- `wardlow` → [MODIS VI separability, Great Plains](<Analysis of time-series MODIS 250 m vegetation index data for crop classification in the U.S. Central Great Plains.md>)
- `breiman2001` → [Random Forests](<breiman2001_RandomForests.md>)
- `friedman2001` → [Greedy Function Approximation](<GREEDY FUNCTION APPROXIMATION:.md>)
- `chen2016` → [XGBoost](<xgboost_chen2016.md>)
- `ke2017` → [LightGBM](<ke2017_LightGBM_NeurIPS.md>)
- `tabnet_orig` → [TabNet](<tabnet_orig_Arik.md>)
- `sparsemax` → [Sparsemax (Martins & Astudillo)](<From Softmax to Sparsemax - A Sparse Model of Attention and Multi-Label Classification.md>)
- `belgiu2016` → [RF in remote sensing: a review](<Random forest in remote sensing: A review of applications and future.md>)
- `maxwell2019` → [ML classification in RS: applied review](<Implementation of machine-learning classification in remote sensing  an applied review.md>)
- `ball2017` → [DL in RS survey](<ball2017_DL_RemoteSensing_survey.md>)
- `dl_aerial_review` → [DL for crops in aerial imagery: review](<Deep Learning Models for the Classification of Crops in Aerial Imagery: A Review.md>)
- `hohl2024` → [XAI in remote sensing](<hohl2024_XAI_RemoteSensing.md>)
- `begue2020` → [RS & agriculture policy, Africa](<begue2020_RS_Africa_policy.md>)
- `cnnsvm_grape` / `crop_disease` / `weeds_tl` → off-domain close-range RGB ([grapevine](<A CNN-SVM study based on selected deep features for grapevine leaves classification.md>), [crop/disease](<Crop identification and disease classification using traditional machine learning and deep learning approaches.md>), [weeds](<Towards weeds identification assistance through transfer learning.md>))

## 1. Every paper sorted by evaluation protocol

**Tier 1 — true spatial transfer (different tile/region/zone, or cross-year/cross-sensor; no leakage of place):**
- `pankajakshan` — strongest in corpus: params fixed on one HSI dataset, applied *without retraining* to disjoint, cross-sensor, cross-region, open-set scenes (KSC, Botswana, PRISMA/Piedmont). 3D-CNN 97–99% in-region → "very poor" on transfer; sparse SVM+kernel transfers (AA/κ ~97–99%).
- `cropformer2023` — cross-region transfer + in-season cross-year. Cross-region OA ~62–64% (pre-trained) vs RF 25.2%. **Caveat: see §6 — no independent reproduction, no code, GaoFen-1 not Sentinel-2.**
- `jin_gee` — cross-landscape national holdout + external ground crop-cuts. Maize OA 79% Tanzania → 63% Kenya (visible transfer penalty). Binary-ish task, SAR fusion.
- `rustowicz2019` — cross-*difficulty* (Germany vs Ghana/South Sudan); deep-net edge over RF evaporates/reverses in smallholder regime (Ghana RF 62.4 ≥ best deep 57.3 macro-F1). Confounds region + data volume + field size; not a same-region disjoint tile.
- `mann_ethiopia` — spatial-autocorrelation-controlled CV + temporal rare-event holdout (forecasts 2015 drought). Loss forecasting, not crop-type classification; methodological ancestor (engineered temporal features + balanced metrics + anti-leakage).

**Tier 2 — field/parcel-disjoint or block-disjoint k-fold within a single scene/region (no spatial transfer):**
- `ltae2020` — field-wise 5-fold, one French region, one season. **Prime in-region exhibit:** crowns L-TAE (mIoU 51.7), buries RF (32.5) — the exact ordering our holdout reverses.
- `tempcnn` — parcel-disjoint k-fold, one 24×24 km scene. TempCNN > RF > RNN in-region.
- `penabarragan2011` — 650-field independent holdout, one county/year. Sparse decision tree (336→24 features); OA 79%, κ 0.75.
- `teixeira` — polygon 60/20/20, one N. Italy scene. 2D CNN-GRU 99.1% OA / macro-F1 99.1% (interleaved split).
- `maize_kenya` — field-level 65/35, one area, 87 fields. Flowering-phase OA 86%, κ 0.71.
- `russwurm` — 3.84 km spatial-block split, single AOI, pixel-scored. ConvGRU 89.7% OA. (Block-wise = stronger than pooled but still in-region.)
- `convstar_sar` — 10 km grid-disjoint 60/20/20, single Bei'an tile/year. 3D-ConvSTAR OA 91.7%, mean-F1 87.7%. (Best precedent for FID-disjoint as a *minimum* standard.)
- `nowakowski` — mosaic-disjoint but distribution-*matched* test, single season; authors concede geographic generalization "not fully addressed." ~83–90% weighted OA but macro recall ~0.5.

**Tier 3 — pooled / random k-fold; spatial-autocorrelation & FID leakage likely (numbers inflated):**
- `saini2018` (random pixel 70/30; RF 84.2% OA), `hcrnn` (random 10/90 pixel, contiguous patches — 97.6% OA, severe leakage), `ienco2017` (pooled 5-fold), `cnnrf_hybrid` (pooled pixel 25/25/50; Conv1D-RF 94.3% OA), `multimodal_naip` (random patch; 98.4%), `unet_kenya` (random patch 80/10/10; U-Net 95.3% OA but 73.6% F1, intercrop ~15–25%), `optsar_iran` (pooled point 50/50; fused 89% vs 77% single-sensor), `bandar2024` (pooled 80/20 on artificially rebalanced 7-class subset; hybrid F1 0.82), `cai_dl` (in-region, thin deep margin ~1.4 pts; protocol incompletely recoverable), off-domain RGB: `cnnsvm_grape` 97.6%, `crop_disease` ~90%, `weeds_tl` 99.3% micro-F.

**Tier 4 — reviews, theory, method papers, separability studies:**
- Tree/boosting foundations: `breiman2001`, `friedman2001`, `chen2016`, `ke2017`. Sparse-deep: `tabnet_orig`, `sparsemax`. RS reviews: `belgiu2016` (RF — and §6.1 documents RF spatial non-transferability), `maxwell2019`, `ball2017`, `dl_aerial_review` (DL beat classical in 35/36 in-region studies), `hohl2024` (XAI). Context/premise: `begue2020` (Africa policy), `wardlow` (MODIS separability — crops separable only in specific phenological windows; same-crop signature shifts ~1 month across sub-regions).

## 2. Directly comparable papers (define our realistic targets)

No paper is an exact match (Western Cape, Sentinel-2 optical-only, field-level, five crops, adjacent-tile holdout). The closest, in priority order:

1. **`pankajakshan` — the thesis-confirming comparator.** Independent demonstration that a high-capacity dense net at 97–99% in-region collapses under spatial/cross-sensor transfer while a sparse few-parameter model transfers. Different sensor (HSI) and task framing, but the *mechanism* and *direction* are exactly ours. Use as the primary external corroboration.
2. **`rustowicz2019` — the geography/regime comparator.** Global-South smallholder, optical(+SAR), RF-vs-deep, macro-F1 primary. Quantifies deep-net collapse toward/below RF in the hard regime. Our targets should sit near its smallholder-regime macro-F1s (best ~57–70). SAR-doesn't-help-small-fields validates our optical-only choice.
3. **`jin_gee` — the operational-transfer comparator.** Cross-landscape holdout with an honest transfer penalty (79→63% maize). Confirms that out-of-sample discipline exposes drops in-region CV hides; sparse RF on parsimonious composites transfers — aligns with our feature-reduction story.
4. **`cropformer2023` — the complicating comparator.** The one dense model reported to transfer well (cross-region OA ~62–64% vs RF ~25%). Must be cited *with* its confound (massive self-supervised pre-training our dense nets lack) — see §6/§7.
5. **`ltae2020` — the in-region foil and same-model-family anchor.** Field-level temporal-attention crowned by field-wise CV, RF buried; we run L-TAE ourselves and show it degrades/reorders under the holdout (and recovers with field aggregation).

## 3. Feature-design findings (discounted by tier)

- **Raw bands, especially red-edge & SWIR, carry most of the discriminating signal; derived indices add little once bands are present.** `cnnrf_hybrid` (Tier 3) flags B5/B12 most discriminative; `tempcnn` (Tier 2) — spectral indices add little over bands; `maize_kenya` (Tier 2) — full bands beat derived VIs. Supports keeping `B11`/`B12` (SWIR) in our set; we lack an explicit red-edge band — note as a possible gap.
- **Spectral indices transfer across regions more stably than raw bands** (`belgiu2016`, Tier 4, §6.1). Mild support for retaining `EVI`/`hue` as transfer-robust complements — a Tier-4 claim, not measured.
- **Phenological windows are decisive; separability is stage-dependent and shifts across sub-regions** (`wardlow`, `maize_kenya`, `penabarragan2011`). Same-crop signatures drift ~1 month across a single state (`wardlow`) — the seed mechanism of holdout reordering. Reinforces our May/June exclusion and multi-month design.
- **Automated time-series feature extraction is under-used and largely untested in crop classification.** `mann_ethiopia` is the lineage (41 phenological metrics + balanced metrics + anti-leakage CV). Our `xr_fresh` evaluation is genuinely first-of-kind — a contribution, not just a method choice.
- **Optical+SAR fusion helps in-scene (~12 OA pts, `optsar_iran`; minority-class rescue, `convstar_sar`) but the benefit is scale-dependent and may not survive to small fields** (`rustowicz2019`: SAR didn't reliably help smallholders). All fusion gains are Tier 2/3 (in-scene), so they are upper bounds, not transfer-validated. Supports our deliberate optical-only scope; flag SAR as future work, not a current omission.

## 4. Model-architecture findings (discounted by tier)

- **In-region, dense temporal/patch nets win — by thin and unstable margins.** `ltae2020` (Tier 2) L-TAE >> RF; `teixeira` (Tier 2) 2D CNN-GRU 99%; `tempcnn` (Tier 2) TempCNN > RF > RNN; `dl_aerial_review` (Tier 4) DL beat classical in 35/36 *in-region* studies. But margins are often ~1–4 pts (`cai_dl`, `convstar_sar`, `hcrnn`) and rankings among temporal nets flip across datasets (LSTM best in some, worst in `cai_dl`). Thin, unstable in-region margins are exactly what should not be trusted to survive transfer — our core point.
- **Under genuine transfer the ranking inverts: sparse/few-parameter models hold, dense nets collapse.** `pankajakshan` (Tier 1, decisive), `rustowicz2019` (Tier 1, RF ≥ deep in smallholder regime), `jin_gee` (Tier 1, sparse RF transfers). The lone counter is `cropformer2023` — but pre-training-confounded.
- **Swapping a dense softmax head for a sparse/tree-like head reduces overfitting** even in-region: `cnnrf_hybrid` (RF head over CNN features, +1.3–1.7 OA), and off-domain `cnnsvm_grape`/`weeds_tl` (CNN features → SVM/XGBoost head, small train/test gap). Mechanistic support for why TabNet groups with boosted trees.
- **Field/parcel aggregation is a real variance-reduction lever** (`ltae2020`'s Pixel-Set Encoder; `penabarragan2011`'s object level). Matches our finding that aggregation recovers transfer for dense temporal models (L-TAE) but is redundant for already-sparse ones.
- **The sparse inductive bias is shared by our transfer-robust set:** tree split criteria implicitly select features (`breiman2001`, `friedman2001`, `chen2016` sparsity-aware splits, `ke2017` EFB), and TabNet inherits it via `sparsemax` attentive masks (`tabnet_orig`, `sparsemax`). This is the §5 backbone of our mechanistic claim.

## 5. Conceptual / theoretical framing

- **Inductive bias governs transfer.** Sparse, axis-aligned, regularized feature selection (the two-decade RF/GBT workhorse property, `belgiu2016`/`breiman2001`/`friedman2001`/`chen2016`) is what degrades gracefully under covariate shift; dense end-to-end representation learning overfits region-specific signal (`ball2017` states transfer "typically fails" under sensor/region/time shift). `tabnet_orig`+`sparsemax` give the bridge: a deep net can keep the sparse bias and stay transfer-robust.
- **Sparse attention as interpretability + transfer lever.** `sparsemax` is the operator inside TabNet's masks and the basis for a sparse-attention L-TAE variant; `hohl2024` documents the field-wide XAI rigor deficit and the per-band-importance gap that sparse selection addresses.
- **Operational framing.** `begue2020` motivates why spatial/operational transfer (not in-distribution accuracy) is the right bar for the Global South; the Western Cape's large commercial fields make it comparatively tractable — our gaps are a *lower* bound on harder regions.

## 6. Project-specific caveats surfaced by the audit

- **Our holdout is an adjacent tile in the same agroecological zone** → a *local* spatial shift. Our reported gaps are a conservative lower bound on cross-region/cross-climate transfer (cf. `pankajakshan`/`jin_gee`, where shifts were larger and drops steeper). State this explicitly (already in `sn-article.tex` §Study Area / §Discussion).
- **`cropformer2023` is not independently verified.** As of mid-2026: no independent reproduction (the only transfer follow-ups are the same China Agricultural University group's self-citations), **no released code, no pre-trained weights, no released pre-training corpus, framework unspecified**, and it uses GaoFen-1 (RGB+NIR, 16 m) — not Sentinel-2 multispectral. Its cross-region numbers therefore are *not directly comparable* to ours and rest entirely on the original authors. Cite it as the motivating exception, and frame its transfer success as evidence that a strong learned prior (self-supervised pre-training), not architectural capacity, buys transfer.
- **Class imbalance is more severe in our holdout (56% lucerne/medics) than training (43%)** — distribution shift compounds covariate shift. Justifies macro-F1 as primary (cf. `belgiu2016`: RF "favors the most representative classes"; `unet_kenya`: 95% OA vs 74% F1).
- **We lack an explicit red-edge band**, which several Tier 2/3 studies found highly discriminative (`teixeira`, `cnnrf_hybrid`). Note as a feature-set limitation, not a fixable gap given the input data.
- **Several in-region SOTA numbers we might be tempted to compare against are leakage-inflated** (`hcrnn` 97.6%, `multimodal_naip` 98.4%, `teixeira` 99.1%). Never benchmark our holdout macro-F1 against these.

## 7. Prioritized action list

**High lift:**
1. **Frame the `cropformer2023` discussion precisely in `sn-article.tex`** (the related-work paragraph at ~line 120 already cites it). Add the §6 caveats: no independent reproduction, no code/weights/corpus, GaoFen-1 ≠ Sentinel-2, pre-training confound. This turns a potential reviewer objection ("but Cropformer shows dense nets transfer") into a supporting point (transfer came from a prior our nets lacked).
2. **Lean on `pankajakshan` as the primary external corroboration** in the discussion — it is the one Tier-1 study that independently confirms in-region misranking + sparse-model transfer + spatial regularization as a variance lever. Currently under-cited relative to its on-point-ness.
3. **Make the tier distinction explicit in the paper's evaluation-protocol argument.** The audit here (Tier 1 ≈ 5 papers; Tier 2/3 ≈ the rest; transfer measured in ~0 crop-type studies on a common holdout) is itself evidence for the gap we fill — quote the corpus-wide silence.

**Medium lift:**
4. **Optionally add Cropformer as a benchmarked model.** Integration surface is well-defined and small (per codebase audit): 4 new files (~1,100 lines) mirroring `ltae_model.py`/`ltae_field.py` + `out_of_sample/inference_*` patterns, plus 2 edits (`deep_learn/src/run_all_dl_models.py` STAGE_2 list; `out_of_sample/0_create_data_run_all_inference.py` STEPS dict). Input shape `(B, T=10, C=6)` fits directly; no config/report/registry changes. **But:** requires a clean-room re-implementation from the paper (no official code), and *without* the self-supervised pre-training corpus it would be a supervised-only Cropformer — which tests our thesis (does the architecture alone, minus the prior, transfer?) rather than reproducing their result. Frame any such run as "supervised Cropformer, no pre-training," and expect it to behave like the other dense nets. Decide based on reviewer pressure; not required for the core argument.
5. **Field-reduction experiment** (`experiments/field_reduction/`) directly supports the data-efficiency claim (`pankajakshan` transfers with 1–10% labels; `maxwell2019` RF <5% loss at 70% training cut). Ensure the 25% result is reported against the holdout, not in-region.

**Lower lift:**
6. Consider a sparse-attention (sparsemax/entmax) L-TAE ablation as future work (`sparsemax`) — interpretability + potential transfer gain. Mention, don't necessarily run.
7. Note the missing red-edge band as a limitation; note SAR fusion as future work (with the `rustowicz2019` caveat that it may not help our field sizes — though Western Cape fields are large, so it might).

## 8. Numerical targets the literature supports under genuine spatial transfer

- **Holdout macro-F1 in the ~0.55–0.65 band is a credible, defensible target for the best transferable model** — our TabNet ensemble at 0.60 (Kappa 0.54) sits squarely in the range set by the only comparable Tier-1 work: `rustowicz2019` smallholder best macro-F1 ~0.57–0.70; `jin_gee` cross-landscape maize OA 63–79%. Do **not** aim for or compare against the 0.90+ in-region/leaky figures (`teixeira`, `hcrnn`, `unet_kenya`); those are Tier 2/3 and measure in-distribution fit.
- **A large, architecture-dependent in-region→holdout gap is the expected, literature-consistent result**, not an anomaly: `pankajakshan` 97–99% → "very poor"; `jin_gee` −16 pts; our CNN-BiLSTM 0.72→0.46 (−0.26) is well within the precedented range of dense-net collapse.
- **Kappa under transfer ~0.45–0.55 for the best models** is consistent with `penabarragan2011` (0.75 in-region field holdout) and `russwurm` (0.87 in-region block) once discounted for the move to a truly disjoint tile and a more imbalanced target.
- **Logistic regression / GBT transferring within a few points of the best model (our LR ~0.56)** is consistent with `rustowicz2019`'s RF matching deep nets and `maxwell2019`'s "no universal winner; tree ensembles robust under reduced/noisy data." The story is not "classical beats deep" but "the gap between simple and complex *shrinks or reverses* under transfer."
