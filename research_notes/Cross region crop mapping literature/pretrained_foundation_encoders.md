# Pretrained / foundation encoders for pixel time-series crop classification (2023–2026)

Scope: whether to add a "Presto-style pretrained encoder" baseline (Reviewer 2) to the CEA revision. Protocol constraints: Sentinel-2 only, 2017 season, 5 winter crops, Western Cape; pixel monthly composites of B2, B6, B11, B12, EVI, hue (~10 steps, months 05/06 dropped); train on two tiles, test on adjacent holdout tile; field-level aggregation.

Full citations used below:
- Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., Kerner, H. (2023, v4 Feb 2024). "Lightweight, Pre-trained Transformers for Remote Sensing Timeseries" (Presto). arXiv:2304.14065. No peer-reviewed venue listed on the arXiv page. https://arxiv.org/abs/2304.14065
- Tseng, G., Fuller, A., Reil, M., Herzog, H., Beukema, P., Bastani, F., Green, J. R., Shelhamer, E., Kerner, H., Rolnick, D. (2025). "Galileo: Learning Global & Local Features of Many Remote Sensing Modalities." arXiv:2502.09356 (v3 Jun 2025; keywords list ICML, venue not explicitly confirmed on page). https://arxiv.org/abs/2502.09356
- Butsko, C., Van Tricht, K., Tseng, G., Milli, G., Rolnick, D., Cartuyvels, R., Becker Reshef, I., Szantoi, Z., Kerner, H. (2025). "Deploying Geospatial Foundation Models in the Real World: Lessons from WorldCereal." arXiv:2508.00858. https://arxiv.org/abs/2508.00858
- Astruc, G., Gonthier, N., Mallet, C., Landrieu, L. (2024/2025). "AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities." arXiv:2412.14123 (v3 May 2025; venue not shown on arXiv page, commonly cited as CVPR 2025, unverified here). https://arxiv.org/abs/2412.14123
- Marsocci, V., Audebert, N., et al. (2024/2025). "PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models." arXiv:2412.04204 (v2 Apr 2025). https://arxiv.org/abs/2412.04204
- Chang, Y.-C., Stewart, A. J., Bastani, F., Wolters, P., Kannan, S., Huber, G. R., Wang, J., Banerjee, A. (2025). "On the Generalizability of Foundation Models for Crop Type Mapping." IEEE IGARSS 2025; arXiv:2409.09451. https://arxiv.org/abs/2409.09451
- Feng, Z., Atzberger, C., Jaffer, S., Knezevic, J., Sormunen, S., Young, R., Lisaius, M. C., Immitzer, M., Jackson, T., Ball, J., Coomes, D. A., Madhavapeddy, A., Blake, A., Keshav, S. (2025, v7 Apr 2026). "TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis." arXiv:2506.20380. https://arxiv.org/abs/2506.20380
- Brown, C. F., Kazmierski, M. R., ..., Shelhamer, E., Wiles, O., Gorelick, N., ..., Kohli, P. (2025). "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data." arXiv:2507.22291. https://arxiv.org/abs/2507.22291
- Zvonkov, I., Tseng, G., Becker-Reshef, I., Kerner, H. (2025). "Cropland Mapping using Geospatial Embeddings." arXiv:2511.02923. https://arxiv.org/abs/2511.02923
- Shang, Z., Das, S., Eldawy, A. (2026). "Benchmarking Geospatial Foundation Models for Agriculture Applications." arXiv:2606.29664. https://arxiv.org/abs/2606.29664
- Mughees, M. A., Montefoschi, G., Brovelli, M. A., Chen, Z. (2026). "From Foundation Embeddings to Cropland Maps: Label Efficiency, Temporal Transferability and Independent Human Validation." arXiv:2609.17138. https://arxiv.org/abs/2609.17138
- Yuan, Y., Lin, L. (2021). "Self-Supervised Pretraining of Transformers for Satellite Image Time Series Classification" (SITS-BERT). IEEE JSTARS 14:474–487, doi:10.1109/JSTARS.2020.3036602 (citation from memory; code https://github.com/linlei1214/SITS-BERT). Pre-2023, included for context only.
- Yuan, Y., Lin, L., Liu, Q., Hang, R., Zhou, Z.-G. (2022). "SITS-Former: A pre-trained spatio-spectral-temporal representation model for Sentinel-2 time series classification." Int. J. Appl. Earth Obs. Geoinf. 106:102651 (citation from memory; see https://pure.nwpu.edu.cn/zh/publications/sits-former-a-pre-trained-spatio-spectral-temporal-representation/).
- Dumeur, I., Valero, S., Inglada, J. (2024). "Self-supervised spatio-temporal representation learning of Satellite Image Time Series" (U-BARN). IEEE JSTARS 17 (Jan 2024) (DOAJ listing https://doaj.org/article/78300210f8244fe1be47ded252e25742; HAL https://hal-agroparistech.archives-ouvertes.fr/RESEAU-TELEDETECTION-INRAE/hal-04084839v2).

## Q1. Presto: input format, partial bands / missing months, reported results, code, limitations

### Takeaway
Presto is a ~0.4M-parameter pixel-time-series transformer pretrained on monthly S1+S2+ERA5+SRTM+Dynamic World+lat/lon with structured channel/timestep masking, so it can technically be run with only 4 of its 10 S2 bands and with masked months; it reported large gains over random forest on CropHarvest (mean F1 0.835 vs 0.441) and, in WorldCereal, fine-tuned Presto beat raw-feature CatBoost most clearly under geographic shift. But our 6 channels map onto Presto's input space only partially (EVI and hue are not Presto inputs; NDVI needs B4/B8, which we do not currently have).

### Cited Findings
- Inputs: S1 VV/VH; 10 S2 bands (B2–B4 RGB, B5–B7 red edge, B8, B8A, B11–B12; 60 m bands removed) organized into channel groups; ERA5 total precipitation and 2 m temperature; NDVI from B4/B8; Dynamic World monthly mode class; static SRTM elevation/slope and lat/lon as 3D Cartesian coordinates — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- Temporal unit is monthly; pretraining used 2020–2021 data (12-month windows in main text, 24 one-month timesteps in Appendix A.1.1, an internal inconsistency) — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- Pretraining masks with four structured strategies (random, channel-group, contiguous timesteps, timesteps) at 0.75 ratio; combining structured + random masking gave best validation F1 (0.665 vs 0.646 random-only) — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- Size ~0.4M encoder parameters (402K); pretraining set 21.5M pixel samples sampled with Dynamic World's ecoregion-stratified strategy — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- Code: `presto.construct_single_presto_input` builds `x`, `mask`, `dynamic_world` from whatever bands you have (e.g., `s2_bands=["B2","B3","B4"]`); missing bands are masked and ignored; `dynamic_world` can be filled with 9 (ignored) if unavailable; `month` = start month (int or tensor); 1–24 timesteps supported; `mask=1` hides a token; encoder with `eval_task=True` returns pooled embeddings; `construct_finetuning_model(num_outputs=...)` adds a linear head; `Presto.load_pretrained()` loads weights; install via `pip install -e .` from repo (no PyPI route documented); MIT license — [nasaharvest/presto GitHub](https://github.com/nasaharvest/presto)
- CropHarvest mean F1: Random Forest 0.441, MOSAIKS-1D 0.738, TIML 0.802, Presto (frozen + RF/LR head, "PrestoR") 0.835; per region Kenya 0.816, Brazil 0.891, Togo 0.798; still beat TIML/MOSAIKS when given only a subset of months — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- TreeSatAI weighted F1 (S2): MLP 51.97, LightGBM 48.17, PrestoRF 55.29; EuroSAT MS fine-tune accuracy 0.953 vs SatMAE 0.990 (303M params) — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- S2-Agri100 (crop): Presto OA 68.89 / F1 40.41 vs SITS-Former 67.03 / 42.83; randomly initialized Presto only 45.98 OA, so pretraining matters strongly — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- Mixed regression results: fuel moisture RMSE RF 23.84 better than PrestoFT 25.28 and PrestoRF 25.98; pretraining did not help on TreeSatAI S1 — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- Stated limitations: built for 10 m pixels; pixel-based so image-level tasks rely on mean/std pooling; plateaus on EuroSAT classes with few labelled pixels — [Presto arXiv v4 HTML](https://arxiv.org/html/2304.14065v4)
- WorldCereal (18-month pixel series of S1, S2, DEM, AgERA5): crop-type macro F1 random/geographic/temporal splits: raw CatBoost 0.728/0.563/0.649; fine-tuned Presto 0.809/0.650/0.686; Presto random-init 0.782/0.620/0.646; extra in-domain SSL gave no consistent gain — [Butsko et al. 2025 HTML](https://arxiv.org/html/2508.00858v1)
- WorldCereal per-country crop-type F1 (CatBoost vs fine-tuned Presto): Brazil 0.563 vs 0.745, Spain 0.400 vs 0.516, Morocco 0.228 vs 0.384, Mozambique 0.397 vs 0.437, Italy 0.614 vs 0.623; cropland Ethiopia Presto 0.683 < deployed expert-feature baseline 0.692 — [Butsko et al. 2025 HTML](https://arxiv.org/html/2508.00858v1)
- WorldCereal caveats: Presto pretrained on S2 L1C least-cloudy composites vs operational L2A pipeline (preprocessing mismatch); gains vanish without pretraining; averages hide weak countries; Presto cheap (~38M MACs per 12-step series vs 89M Galileo-Nano, 890M AnySat). WorldCereal paper reports only fine-tuning, not frozen-embedding + CatBoost — [Butsko et al. 2025 HTML](https://arxiv.org/html/2508.00858v1)

### Inferences
- Presto can accept our data with masks: supply B2, B6, B11, B12 as S2 inputs, mask the other S2 groups, S1, ERA5, NDVI; mask timesteps for May/June. But B6 sits alone in the red-edge group (B5–B7) and B2 alone in RGB, so most channel groups would be partially populated; EVI and hue have no Presto slot and would be dropped (or used only in the head). This is much weaker input than Presto was evaluated with; expect degraded embeddings.
- AI4FoodSecurity provides full S2 band stacks in the original challenge data, so re-extracting B3, B4, B5, B7, B8, B8A (and NDVI) from the raw inputs would make a Presto baseline much fairer. Whether the project's `DATA_DIR` per-band parquets contain those bands needs to be checked locally (not verified here).
- Presto's month embedding and 2020–2021 pretraining are not a hard barrier for 2017 (pixel-level reflectances, not calendar-specific), but L1C vs our product level is a known mismatch.
- The WorldCereal numbers are the closest analogue to our question (crop type, geographic split): pretrained + fine-tuned Presto beat a strong tree baseline by ~+0.09 macro F1 on geographic holdout, though benefits varied by country.

### Gaps
- Peer-reviewed venue for Presto not confirmed on the pages fetched (arXiv only).
- No source found that runs Presto with only 4 S2 bands and no S1/ERA5; the partial-band robustness is untested in literature I could find.
- No result found for Presto (or any foundation model) on AI4FoodSecurity South Africa specifically.

## Q2. Other candidates: pixel-time-series capable vs patch-only

### Takeaway
Pixel-time-series-native options are Presto, Galileo (which also handles space), TESSERA, AlphaEarth (precomputed annual embeddings), SITS-BERT, and U-BARN's temporal branch; AnySat can produce pixel features but is heavy. SatMAE, Prithvi, SSL4EO, CROMA, DOFA, Copernicus-FM and Clay are image/patch encoders (Prithvi has limited multi-temporal input) and fit our pixel protocol poorly. AlphaEarth is the only candidate with confirmed 2017 global coverage.

### Cited Findings
- Galileo sizes: ViT-Nano 0.8M, Tiny 5.3M, Base 85M params; inputs S1 VV/VH, S2 (all but B1/B9/B10), NDVI, SRTM, Dynamic World, WorldCereal, ERA5, TerraClimate, VIIRS, LandScan, lat/lon; 24 monthly steps at 96×96 px — [Galileo arXiv HTML](https://arxiv.org/html/2502.09356)
- Galileo linear-probing pixel-timeseries results (Togo/Brazil/Kenya/Breizhcrops): Presto 75.5/98.8/84.0/63.0; AnySat-Base 73.4/76.7/75.5/66.1; Galileo-Nano 73.5/76.4/84.5/67.3; Galileo-Tiny 74.7/97.2/85.4/69.0; Galileo-Base 74.8/99.3/84.2/73.0 (metric not stated in the extracted table; CropHarvest done with default sklearn settings because no validation set) — [Galileo arXiv HTML](https://arxiv.org/html/2502.09356)
- Galileo code/weights: MIT license, nano weights in repo, other sizes on Hugging Face (`nasaharvest/galileo`); mask values 0/1/2 control what encoder sees — [Galileo GitHub](https://github.com/nasaharvest/galileo)
- AnySat: JEPA with scale-adaptive spatial encoders, trained on GeoPlex (5 datasets, 11 sensors); code at github.com/gastruc/AnySat — [AnySat arXiv](https://arxiv.org/abs/2412.14123); ~890M MACs per 12-step series, ~23x Presto — [Butsko et al. 2025 HTML](https://arxiv.org/html/2508.00858v1)
- TESSERA: pixel-wise foundation model on S1 (VV/VH asc/desc) + S2 L2A 10 bands; 128-d embeddings at 10 m (Matryoshka in v2); precomputed global embeddings for 2024 and 2017–2025 for regions such as US and Europe, other regions being backfilled toward 2017; on-request generation or self-run (≥1 TB storage per 100×100 km, 128 GB RAM, GPU); weights/embeddings CC0, code MIT — [TESSERA GitHub](https://github.com/ucam-eo/tessera); [TESSERA arXiv](https://arxiv.org/abs/2506.20380)
- AlphaEarth Satellite Embedding: 64-d unit-length vectors per 10 m pixel, annual 2017–2024 (2017 layer regenerated in v1.1, Nov 2025), multi-source (Sentinel-1/2, Landsat, etc.), CC-BY 4.0, in Earth Engine as `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`; GCS bucket became provider-pays July 2026 — [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL); [AlphaEarth arXiv](https://arxiv.org/abs/2507.22291)
- PANGAEA: only SatlasNet and Prithvi natively process time series; other GFMs need linear temporal mapping or L-TAE head (L-TAE usually better; Prithvi on PASTIS-R 27.40→33.93 mIoU); authors conclude dedicated multi-temporal GFMs are needed. Models covered include CROMA, DOFA, SSL4EO variants (S12-DINO, S12-Data2Vec), SpectralGPT, Prithvi, SatlasNet — [PANGAEA HTML](https://arxiv.org/html/2412.04204)
- SITS-BERT: BERT-style self-supervised pretraining of a pixel-time-series transformer — [SITS-BERT GitHub](https://github.com/linlei1214/SITS-BERT). SITS-Former: patch-based, pretrained by masked missing-data imputation on S2 series, +2.64–3.30% OA over supervised on two crop tasks; U-BARN: U-Net spatial encoder + temporal transformer, pretrained on large unlabeled S2 set, frozen features beat supervised linear on land cover and crop classification and don't need cloud masks — [CESBIO summary](https://www.cesbio.cnrs.fr/multitemp/training-deep-neural-networks-for-satellite-image-time-series-with-no-labeled-data); [SITS-Former NWPU record](https://pure.nwpu.edu.cn/zh/publications/sits-former-a-pre-trained-spatio-spectral-temporal-representation/); [U-BARN HAL](https://hal-agroparistech.archives-ouvertes.fr/RESEAU-TELEDETECTION-INRAE/hal-04084839v2)

### Inferences
- Classification for our protocol: pixel-native and plug-in friendly: Presto, Galileo-Nano/Tiny (time-only pixel mode), TESSERA, AlphaEarth, SITS-BERT. Patch-oriented and poor fit: SatMAE, SSL4EO-S12, CROMA, DOFA, Copernicus-FM, Clay, Prithvi (few timesteps), SITS-Former (patch), OmniSat (patch/multimodal; not fetched). AnySat possible but costly.
- AlphaEarth/TESSERA are "embedding products" (annual vectors from their own full-year imagery), not encoders fed our 6-channel series. They bypass band mismatch and missing months entirely but also bypass our input representation, so a comparison would test "different data + pretrained features" rather than "pretraining on same inputs." Their training data may include our holdout tile region and year (no labels involved, but worth noting).
- Galileo ≈ Presto-successor from the same group; Galileo-Tiny/Base modestly beat Presto on Kenya/Breizhcrops but not uniformly (Togo, Brazil roughly equal).

### Gaps
- Did not verify details for Copernicus-FM, Clay, DOFA, CROMA, OmniSat, SSL4EO, Prithvi-EO-2.0 individually beyond PANGAEA coverage; their patch-image orientation is from general knowledge and PANGAEA's statement on temporal handling.
- TESSERA 2017 coverage for South Africa not confirmed (README says backfill ongoing; request form exists).
- No dedicated crop-specific pretrained SITS encoder with public weights covering South Africa found.
- AgriFM: one search hit (Frontiers 2026) concerned crop stress/yield, not verified as a crop-type SITS encoder.

## Q3. Benchmark evidence: do pretrained encoders beat supervised baselines (geographic shift, low labels, Africa)?

### Takeaway
Evidence is mixed. Pixel-time-series pretrained encoders (Presto) show consistent gains over tree baselines on CropHarvest and under WorldCereal geographic splits, including in African countries, but patch-based GFMs frequently fail to beat UNet/ViT and all models degrade heavily under regional shift, especially for minority crops. Gains concentrate in low-label regimes.

### Cited Findings
- PANGAEA crop datasets (mIoU, 100% labels): PASTIS-R ViT 38.53 > best GFM (S12-DINO 36.18); South Sudan CropTypeMapping-SS: GFMs beat UNet 47.57 (S12-Data2Vec 54.03); AI4SmallFarms UNet 46.34 vs best GFM ~27.2. At 10% labels, GFMs help more (South Sudan: UNet 13.88 vs CROMA 36.77, S12-DINO 38.44). Overall: GFMs do not consistently beat supervised models; advantage clearest with scarce labels; regional domain-shift test (non-crop) shows large drops for all — [PANGAEA HTML](https://arxiv.org/html/2412.04204)
- Chang et al. (IGARSS 2025): S2-specific pretrained weights (SSL4EO-S12) beat ImageNet weights across five crop datasets on five continents; ~100 labelled images enough for high OA, ~900 needed to handle class imbalance — [Chang et al. arXiv](https://arxiv.org/abs/2409.09451)
- Shang et al. 2026: Prithvi, SpectralGPT, SatMAE on multi-temporal CDL crop segmentation with region-held-out splits in four US states: all degrade sharply under regional shift, predicting dominant crops and near-zero for minority crops (e.g., SatMAE 45.68 mIoU Iowa vs 3.20 Minnesota); no supervised baseline included — [Shang et al. HTML](https://arxiv.org/html/2606.29664v1)
- WorldCereal: fine-tuned Presto's advantage over raw CatBoost largest on geographic split (crop type +0.087 macro F1, cropland +0.052) vs temporal split (+0.037, +0.012); Tanzania cropland still poor for all (0.258–0.442) — [Butsko et al. 2025 HTML](https://arxiv.org/html/2508.00858v1)
- Togo cropland: Presto and AlphaEarth embeddings + random forest; Presto reported higher (OA ~0.897 vs ~0.859) but AlphaEarth's earliest year then (2021) did not match label season (2019–2020), so not an equal comparison — [Zvonkov et al. 2025 arXiv](https://arxiv.org/abs/2511.02923) (numbers via search summary of the PDF, not directly verified in fetched abstract; also cited secondhand in [Mughees et al. 2026](https://arxiv.org/html/2609.17138v1))
- AlphaEarth + random forest cropland in Maine: 93.75% OA, robust 2018–2023 temporal transfer (92.5–93.8%), and agreement with human validation 95.3% vs CDL 91.7%; no raw-feature baseline, so does not show embeddings beat raw S2 — [Mughees et al. 2026 HTML](https://arxiv.org/html/2609.17138v1)
- Search results also flagged a 2026 Eastern Africa analysis reporting limited separability of frozen embeddings in smallholder/mixed cropping, and a leave-one-country-out SSA yield study (arXiv:2605.08113) asking whether foundation embeddings improve cross-country generalisation — [search hits; not fetched](https://arxiv.org/pdf/2605.08113)

### Inferences
- For a reviewer-facing argument: the literature supports the expectation that a pretrained pixel encoder can help under geographic shift, but the effect size is dataset-dependent and smallest when the supervised baseline is already strong and labels are not scarce. Our setting (thousands of labelled fields, adjacent tile, same season) is a mild shift and moderately label-rich, where PANGAEA/WorldCereal patterns predict small or no gains.
- Head-to-head evidence vs L-TAE/TempCNN (rather than RF/CatBoost) on pixel series is thin; Presto's S2-Agri result is roughly at parity with SITS-Former.

### Gaps
- No study found directly comparing Presto/Galileo frozen embeddings vs L-TAE or TempCNN trained from scratch under cross-region crop-type transfer.
- Did not fetch "Harvesting AlphaEarth" (Ma et al., arXiv:2601.00857) or the Eastern Africa embedding study; findings unverified.
- No South Africa winter-crop (wheat/barley/canola) foundation-model result found.

## Q4. Practical recommendation for this protocol

### Takeaway
Presto is the most defensible single addition: it is exactly what the reviewer named, pixel-time-series native, tiny (runs on CPU/GPU in minutes), MIT-licensed, has documented masking for missing bands/months, and has published crop-type geographic-shift evidence. Run it two ways: (1) frozen pooled embeddings + the paper's standard head (logistic regression/RF/XGBoost) and (2) full fine-tuning with a linear head, both on identical FID-wise splits, then field-aggregate like other pixel models. AlphaEarth 2017 embeddings are a cheap optional second baseline but test a different data input.

### Cited Findings
- Presto API supports both modes: `encoder(..., eval_task=True)` for frozen pooled embeddings and `construct_finetuning_model(num_outputs=...)` for fine-tuning; missing bands masked via `construct_single_presto_input` — [nasaharvest/presto GitHub](https://github.com/nasaharvest/presto)
- Frozen Presto + simple head was the original CropHarvest protocol (PrestoR/PrestoRF); fine-tuned Presto was used in WorldCereal and outperformed CatBoost — [Presto arXiv v4](https://arxiv.org/html/2304.14065v4); [Butsko et al. 2025](https://arxiv.org/html/2508.00858v1)
- Pretraining preprocessing (L1C, least-cloudy monthly composite) differs from operational L2A inputs, a known pitfall — [Butsko et al. 2025](https://arxiv.org/html/2508.00858v1)
- Galileo (same group, MIT, HF weights) is a reasonable second if compute allows, with modest gains over Presto on some pixel tasks — [Galileo arXiv HTML](https://arxiv.org/html/2502.09356); [Galileo GitHub](https://github.com/nasaharvest/galileo)
- AlphaEarth provides 2017 annual 64-d embeddings via Earth Engine (CC-BY 4.0) — [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)

### Inferences
- Recommended implementation details:
  - Inputs: build monthly series with Presto's normalization (`NORMED_BANDS`); provide B2, B6, B11, B12 in Presto's S2 slots; mask all absent channels (B3, B4, B5, B7, B8, B8A, S1, ERA5, NDVI, DW=9); include real lat/lon; set `month` to the true first month and mask the May/June timesteps rather than deleting them so month positions stay correct. Drop EVI/hue from the encoder (optionally concatenate them to the embedding for the head; report both).
  - Strongly preferred: re-extract the missing S2 bands (at least B3, B4, B8, B8A, B5, B7) from the raw AI4FoodSecurity S2 data so Presto gets near-native input; report the 4-band masked variant as a sensitivity check. Otherwise reviewers can argue the baseline was crippled.
  - Use the same 5-seed / FID-wise split / field majority-vote aggregation as other pixel models; report F1 macro, kappa, and the cross-entropy "Xent" column on the holdout.
  - Expect pitfalls: (a) band mismatch and partially filled channel groups; (b) our monthly composites differ from Presto's least-cloudy L1C monthly composites, and reflectance scale/normalization must match; (c) 2017 predates Presto's 2020–2021 pretraining (low risk for reflectance-based pixel model, but mention); (d) frozen 128-d mean-pooled embedding can lose crop-phenology detail separating wheat/barley, so fine-tuning likely needed; (e) Dynamic World absent for 2017 Western Cape (DW starts mid-2015, but simply set to masked).
- Framing in the paper: present Presto as the "pretrained pixel-SITS encoder" baseline; cite PANGAEA and Shang et al. that GFMs do not reliably beat supervised models under regional shift, and WorldCereal for the opposite case, so either outcome is interpretable.
- If only one extra item is affordable, avoid patch-image GFMs (SatMAE, Prithvi, CROMA, DOFA, Clay, Copernicus-FM): they require spatial chips and full band sets and do not match the pixel protocol.

### Gaps
- Actual runtime / accuracy of Presto on this dataset unknown until run.
- Whether project raw parquets include the bands needed for a full Presto input was not checked in this research.
- Presto pip-installable package beyond `pip install -e .` not verified (WorldCereal has its own wrappers, not checked).
