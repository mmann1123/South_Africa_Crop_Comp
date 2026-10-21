# Label-efficiency / learning-curve literature for crop-type classification from SITS (2019–2026)

Scope: papers that report accuracy as a function of the number of labelled fields/pixels (or k-shot per class) for crop mapping, with attention to (i) deep temporal models vs classical models, (ii) spatially separate evaluation, and (iii) the few-shot/transfer literature's classical baselines at a few hundred labels. Our unpublished result for reference: Western Cape S2 benchmark, 10 draws per fraction, adjacent-tile holdout; at 10% of fields (~270) every model sits at ~0.55 macro-F1 (L-TAE, TempCNN, CNN-BiLSTM, logistic regression) except XGBoost on engineered features (0.51); at 100% deep temporal models reach ~0.65, LR plateaus 0.61, XGBoost 0.61, TabNet 0.57.

Access flags: all MDPI, Scientific Data, arXiv and PMC items below are open access. Paywalled or access-restricted items are flagged inline with DOI.

Method note: ~20 searches/fetches; where a publisher blocked WebFetch (MDPI, rcaap) the PDF was downloaded and text-extracted locally, so quoted numbers are from the papers' own tables, not from search snippets, unless flagged "snippet only".

---

## Key question 1: Do deep SITS models beat RF / logistic regression at very small label counts (<500 fields), or only at scale? Any explicit crossover estimates?

### Takeaway
No paper gives a numeric crossover point. The only paper that states the qualitative result directly (Portalés-Julià et al. 2021) says RF matched or beat LSTM/BiLSTM under reduced training data and deep models won only with large training sets; the one head-to-head at ~1,300 labels (Kerner et al. 2020, Togo) has RF beating an LSTM trained on the same local labels (F1 0.76 vs 0.72). Few-shot benchmarks show deep models from scratch collapsing at ≤20 shots per class, with non-parametric baselines (DTW, compressors) catching or passing them at 5–20 shots.

### Cited Findings
- Portalés-Julià, Campos-Taberner, García-Haro & Gilabert (2021), Agronomy 11(4):654, Valencia (Spain), Sentinel-2 multitemporal (67 images × 13 predictors), classifiers k-NN, DT, RF, LSTM, BiLSTM, 3 class scenarios: "The considered deep learning techniques outperformed the machine learning algorithms when large datasets were used. In the case of reduced data for the training process, the random forest is able to perform at the same level, and even better than the deep learning techniques." Best result BiLSTM OA 98.2% (8 active crop classes + abandoned). The discussion adds that DL won "especially ... in the third scenario, where training datasets disposed the abundant samples for the majority of classes" and "random forest performed even better than deep learning architectures in scenarios with limited training data". Same-region random split, not spatial transfer. — [Portalés-Julià et al. 2021, DOI 10.3390/agronomy11040654](https://www.mdpi.com/2073-4395/11/4/654)
- A 2026 systematic review (ScienceDirect, S235293852600412X; access status not verified, may be paywalled) cites Portalés-Julià et al. (2021) as showing that deep learning's advantage over RF appears as training set size increases (snippet only). — [Review](https://www.sciencedirect.com/science/article/pii/S235293852600412X)
- Kerner et al. (2020), "Rapid Response Crop Maps in Data Sparse Regions" (KDD 2020 Humanitarian Mapping workshop), Togo cropland (binary), Sentinel-2 monthly 12-step series, 1,319 hand-labelled points, test set 350 points photointerpreted by 4 labellers, consensus test set 106 crop + non-crop. Table 1: Single-headed LSTM (hand-labelled only) Acc 0.71 / AUC 0.90 / P 0.59 / R 0.93 / F1 0.72; SVM 0.79 / 0.88 / 0.65 / 0.84 / 0.73; Random Forest 0.81 / 0.90 / 0.66 / 0.88 / 0.76; Multi-headed LSTM (hand-labelled + global GeoWiki "See" data) 0.83 / 0.91 / 0.81 / 0.68 / 0.74. So with ~1.3k local labels, RF beats the locally trained LSTM on accuracy and F1; the LSTM only pulls ahead on accuracy/precision when augmented with a global labelled dataset. Figure 5 plots AUC/accuracy of the multi-headed model vs number of hand-labelled examples and reports performance "increased as the size of the hand-labeled dataset increased" (full curve values not extractable from text). — [Kerner et al. 2020, arXiv 2006.16866](https://arxiv.org/abs/2006.16866)
- SustainBench's summary of the same Togo work: a multi-headed LSTM outperformed SVM and RF on cropland prediction in Togo (true for accuracy/AUC, not for F1, per the table above). — [SustainBench, arXiv 2111.04724](https://arxiv.org/pdf/2111.04724)
- Tseng et al. (2023), Presto, CropHarvest few-label tasks (binary crop/non-crop, pixel time series): training labels Kenya 1,345, Brazil 203, Togo 1,319. F1: Random Forest on raw inputs Kenya 0.559 / Brazil 0.000 / Togo 0.756 (mean 0.441); MOSAIKS-1DR 0.790 / 0.746 / 0.679 (0.738); TIML (meta-learned LSTM) 0.838 / 0.835 / 0.732 (0.802); PrestoR (frozen pretrained transformer + linear/RF head) 0.816 / 0.891 / 0.798 (0.835). Presto fine-tuning was not run on CropHarvest "because the dataset is designed for few-shot learning". The RF collapse on Brazil (203 labels, highly imbalanced) is the clearest published example of a classical model failing at a few hundred labels, but the winner is a pretrained/meta-learned model, not a from-scratch deep net. No label-count sweep. — [Tseng et al. 2023, arXiv 2304.14065](https://arxiv.org/abs/2304.14065)
- Mohammadi, Belgiu & Stein (2024), Remote Sensing 16(6):1026, pixel-level few-shot crop mapping. Scenario 1 Ghana (source 7.25M px, target 1.16M px, 24 target crop types); Scenario 2 France/PASTIS infrequent classes (7 target classes). Five-way macro-F1 (%) under realistic Dirichlet query sampling, 30,000 tasks, Ghana: "Baseline" deep model (pretrained on source, nearest-centroid on target) 42.5 / 57.3 / 61.4 / 64.7 at 1/5/10/20 shots; DTW nearest-neighbour (non-deep) 40.3 / 56.9 / 64.2 / 70.9; α-TIM (transductive) 43.4 / 60.4 / 68.8 / 75.2; MAML 38.9 / 57.8 / 63.9 / 66.9. France five-way: Baseline 46.2 / 68.4 / 73.3 / 76.1; SimpleShot 48.8 / 68.4 / 73.2 / 75.9. 24-way Ghana: Baseline 29.4 / 39.6 / 42.1 / 44.5; α-TIM 59.6 at 20-shot. DTW, a classical distance method, overtakes the deep Baseline at 10 and 20 shots in Ghana. — [Mohammadi et al. 2024, DOI 10.3390/rs16061026](https://www.mdpi.com/2072-4292/16/6/1026)
- Cheng et al. (2024), "Low-Resource Crop Classification from Multi-Spectral Time Series Using Lossless Compressors", pixel-level, 3 datasets (German, T31TFM-1618, PASTIS), n-shot per class with 5 seeds. PASTIS mIoU (%) at 50/20/10/5-shot: TempCNN 75.59 / 57.51 / 42.53 / 29.18; Transformer 74.34 / 58.70 / 41.10 / 28.74; LSTM 70.14 / 54.97 / 45.75 / 34.50; DCM (attention BiLSTM) 59.96 / 39.32 / 36.78 / 16.82; MLP 34.45 / 10.15 / 2.51 / 3.17; compressor method 62.63 / 49.31 / 41.13 / 33.93. German dataset: TempCNN 70.66 / 56.43 / 43.72 / 31.40; Transformer 54.14 / 39.34 / 23.83 / 16.69; LSTM 52.91 / 41.08 / 29.62 / 19.78; compressor 50.25 / 39.47 / 33.78 / 28.52. Full-data OA on PASTIS: TempCNN 98.67, Transformer 98.32, LSTM 97.37, MLP 92.06. Authors: at 5-shot the non-parametric method "outperforms almost all deep learning models", and "CNN-based models perform better in the few-shot setting". Same-region random split; no RF/LR. — [Cheng et al. 2024, arXiv 2405.18119](https://arxiv.org/abs/2405.18119)
- Resilient in-season classification (growth-stage normalization, arXiv 2009.10189): RF had lower OA than CNN-LSTM but higher OA than the other neural networks tested; authors suggest RF "can achieve comparable performance to deep learning methods for this problem" (snippet only). — [arXiv 2009.10189](https://arxiv.org/pdf/2009.10189)
- Smallholder street-view study (arXiv 2309.05930): RF trained on ~81,000 CNN-labelled (noisy) samples reached 93% accuracy vs 69% with ~984 expert-labelled samples; authors attribute this to sample volume outweighing label noise (snippet only). — [arXiv 2309.05930](https://arxiv.org/pdf/2309.05930)

### Inferences
- Across the found literature, from-scratch deep SITS models never clearly beat RF/kNN/DTW-type baselines below ~20 labels per class (Mohammadi; Cheng), and at ~1,300 binary labels RF still edged a locally trained LSTM (Kerner). The deep wins at a few hundred labels come from pretraining or meta-learning (Presto, TIML, α-TIM), not architecture alone.
- Portalés-Julià is the only paper that states the "RF ≥ DL at low data, DL > RF at high data" pattern in crop/land-use SITS, and it gives no sample sizes or crossover; our explicit ~15%-of-fields (≈400 fields, ≈45/class) crossover under spatial holdout appears to be new.
- Our 10% point (~270 fields over 9 classes ≈ 30 fields/class) sits at exactly the regime where RF itself becomes unstable (Schulthess, below) and where few-shot papers show deep models far below their asymptote, which is consistent with all models converging to a common floor.

### Gaps
- No paper found that reports a deep-vs-classical crossover point in number of fields or fraction of data. Portalés-Julià gives the direction only; the extracted text contains no training-set sizes.
- Rustowicz et al. (2019, CVPRW) Ghana/South Sudan: confirmed F1 57.3/OA 60.9% (Ghana) and 69.7/85.3% (South Sudan) for the best DL model, but no RF baseline or label sweep was found. — [Rustowicz 2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.html)
- Weikmann/TimeSen2Crop, BreizhCrops (Rußwurm 2020), PASTIS (Garnot 2021), DeepCropMapping (Xu 2020): no label-fraction sweep was found for any of them; BreizhCrops benchmarks RF against 6 DL models on spatially separated splits at full data only (snippet). — [BreizhCrops](https://arxiv.org/abs/1905.11893)

---

## Key question 2: Any paper showing linear models equal deep models at ~10% data under spatial holdout?

### Takeaway
None found. The only label-sweeps under genuine spatial transfer (EuroCropsML Latvia→Estonia) include no linear or tree baseline at all; the only deep-vs-classical small-label comparisons (Kerner Togo, Portalés-Julià, Mohammadi, Cheng) are not fraction sweeps and/or are same-region splits. Logistic regression appears nowhere as a crop-SITS baseline in the found label-sweep literature.

### Cited Findings
- EuroCropsML dataset paper (Reuss et al. 2025, Scientific Data), 706,683 parcels, 176 classes, Sentinel-2 L1C per-parcel medians 2021, Estonia/Latvia/Portugal. Only model: Transformer encoder, from scratch vs pretrained, fine-tuned on Estonia with k parcels per class. Test accuracy on Estonia (Table 2; units ambiguous, likely fractions): from scratch 0.04 / 0.10 / 0.28 / 0.41 / 0.45 / 0.50 / 0.57 at 1/5/10/20/100/200/500-shot; pretrained on Latvia 0.17 / 0.29 / 0.33 / 0.44 / 0.53 / 0.59 / 0.66; pretrained Latvia+Portugal 0.06 / 0.12 / 0.12 / 0.18 / 0.48 / 0.51 / 0.58 (worse than scratch at 10 and 20 shots). No classical baseline. — [Reuss et al. 2025, DOI 10.1038/s41597-025-04952-7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12009327/)
- Reuss et al. (2025, arXiv 2504.11022v2, "Benchmarking for Practice"), same benchmark, 5 runs, Estonia fine-tuning pool 105,543 parcels (127 classes), test 35,182 parcels. Table 4a accuracy, Latvia→Estonia: no pre-training 0.17±0.14 / 0.33±0.11 / 0.438±0.027 / 0.490±0.020 / 0.518±0.033 / 0.573±0.019 / 0.6317±0.0062 at 1/5/10/20/100/200/500-shot; standard transfer learning (LV) 0.143 / 0.203 / 0.254 / 0.367 / 0.465 / 0.503 / 0.575 (below scratch at every k); ANIL 0.189 / 0.250 / 0.419 / 0.490 / 0.561 / 0.589 / 0.652; MAML 0.176 / 0.259 / 0.390 / 0.507 / 0.552 / 0.611 / 0.648; PRESTO (pretrained weights) 0.157 / 0.24 / 0.236 / 0.322 / 0.458 / 0.491 / 0.553; PRESTO-XTS 0.050 / 0.18 / 0.276 / 0.443 / 0.501 / 0.486 / 0.622. Text: "For LV→EE, the standard transfer learning approach did not lead to any improvement ... compared to training from scratch"; the no-pretraining baseline "outperforms all supervised LV+PT→EE algorithms for the 1-, 5-, 10- and 20-shot benchmark tasks". No RF/LR baseline (RF is mentioned only in related work). — [Reuss et al. 2025, arXiv 2504.11022](https://arxiv.org/abs/2504.11022)
- Spatial transferability of RF with S1/S2 (Remote Sensing 2022, 14:1493): RF models using only SAR features showed the lowest accuracy losses when transferred to unseen regions; losses attributed to regional differences and lack of representative training samples (snippet only; no label sweep, no DL). — [DOI 10.3390/rs14061493](https://doi.org/10.3390/RS14061493)
- Nigeria LSTM vs RF with local vs global training data (2023): adding local training data improved accuracy by up to 0.246 (RF) and 0.178 (LSTM); Nigeria-only or regional training beat global training (snippet only, no fraction sweep). — [search result](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=31863)

### Inferences
- The EuroCropsML from-scratch transformer curve (0.44 at 10 shots/class → 0.63 at 500 shots/class, still rising) is the best published "deep model under spatial transfer vs label count" curve; it shows no plateau, matching the rising deep curves in our result, but cannot speak to classical parity because no classical model was run.
- Logistic regression on flattened monthly composites as a competitive crop-SITS baseline is essentially absent from this literature; our LR result (0.61 at full data, 0.55 at 10%) fills that gap.

### Gaps
- No paper found combining (a) multiple deep architectures, (b) linear/GBM baselines, (c) a training-fraction sweep with repeated draws, and (d) an out-of-region test tile. That exact configuration appears to be novel.
- Could not verify whether EuroCropsML accuracy units are fractions or percent (Scientific Data table caption says percent, values are ≤0.66); the arXiv benchmark tables are clearly fractions.

---

## Key question 3: Is "deep models are data-efficient" or "deep models need more data" the consensus, and what numbers support it?

### Takeaway
"Deep models need more data" is the stated consensus in the SITS literature (motivation sections of SITS-BERT, Mohammadi, Cheng, Portalés-Julià), supported by few-shot collapses (deep PASTIS models fall from ~98% OA at full data to ~29% mIoU at 5 shots/class) and by the fact that every low-label deep "win" uses pretraining or meta-learning. The counter-claim (data-efficient deep models) appears only for CNN-type temporal models relative to RNN/Transformer, not relative to RF.

### Cited Findings
- Cheng et al. 2024: deep models' few-shot performance "is not satisfactory" (citing prior work); PASTIS TempCNN goes from OA 98.67 (full) to mIoU 29.18 at 5-shot; MLP collapses to 3.17 mIoU; CNN-based models are the most few-shot-robust of the deep families. — [arXiv 2405.18119](https://arxiv.org/abs/2405.18119)
- Mohammadi et al. 2024: deep models "require a large volume of labeled samples for training", motivating FSL; their deep Baseline gains +22 macro-F1 from 1 to 20 shots (Ghana 5-way) and is passed by DTW at 10–20 shots. — [DOI 10.3390/rs16061026](https://www.mdpi.com/2072-4292/16/6/1026)
- SITS-BERT (Yuan & Lin 2021, IEEE JSTARS, DOI 10.1109/JSTARS.2020.3036602, open access): "deep learning methods suffer from overfitting when labeled data are scarce"; self-supervised pretraining gives "substantial improvements" for Transformer, 1D-CNN and BiLSTM on three benchmarks (label-fraction numbers not retrieved). — [DOAJ record](https://doaj.org/article/4fb58140e0d948d39692eecad8be09a1)
- U-BARN (Dumeur et al. 2024, JSTARS): "in scenarios with scarce reference data, fine-tuning brings a significant performance gain compared to fully-supervised approaches" on PASTIS (snippet only). — [DOAJ record](https://doaj.org/article/78300210f8244fe1be47ded252e25742)
- ICML 2025 "Shaping Fine-Tuning of Geospatial Foundation Models": PASTIS-HD crop segmentation at 1%, 10%, 50%, 100% labels vs U-TAE; full fine-tuning beats partial fine-tuning at 1–10% (numbers not retrieved). — [ICML 2025](https://icml.cc/virtual/2025/47930)
- Mask-PSTIN (ISPRS JPRS 2024, S0924271624003897; paywalled, DOI via ScienceDirect): argues that "insufficient and noisy training data" lead LSTM-type models to overlook generalizable features; proposed model beat RF, SVM, Transformer and CNN-LSTM across three regions (snippet only). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003897)
- Portalés-Julià 2021: "deep learning architectures were able to extract the most information from the training data" when samples were abundant; RF equal or better with reduced data (see Q1). — [DOI 10.3390/agronomy11040654](https://www.mdpi.com/2073-4395/11/4/654)
- EuroCropsML benchmark 2025: transformer from scratch needs 500 shots/class (tens of thousands of parcels) to reach 0.63 accuracy under Latvia→Estonia transfer; meta-learning adds only +0.02 at 500 shots (ANIL 0.652) and nothing consistent below 20 shots. — [arXiv 2504.11022](https://arxiv.org/abs/2504.11022)
- Tree-species U-Net on Sentinel-2 (Lee et al. 2025, Forests 16:1718): 90% of maximum accuracy reached with 2.4–2.7% of the training sample and 95% with 4.4–5.9%, i.e. a strongly concave deep learning-curve on a very large pixel-labelled dataset (not crops). — [DOI 10.3390/f16111718](https://doi.org/10.3390/f16111718)

### Inferences
- The field's "deep is data-hungry" claim is mostly motivational; the quantitative support is few-shot (≤50/class) rather than fraction sweeps at the hundreds-to-thousands-of-fields scale that practitioners face. Our sweep (270 → 2,700 fields) covers exactly the under-documented middle range.
- The one data-efficiency claim with numbers (Lee 2025) concerns a segmentation U-Net on a dataset large enough that 2.5% is still thousands of pixels; it does not contradict the few-shot collapses.

### Gaps
- Could not retrieve label-fraction tables for SITS-BERT, SITS-Former (Yuan et al. 2022) or U-BARN; those papers would give deep-only curves at 10–20% labels but not classical comparisons.
- Miller et al. (2024) GRSM review "Deep Learning for Satellite Image Time Series" (DOI 10.1109/MGRS.2024.3393010) likely discusses label requirements, but only the abstract was retrievable. — [arXiv 2404.03936](https://arxiv.org/abs/2404.03936)

---

## Key question 4: Do reported learning curves plateau for classical models while deep models keep rising?

### Takeaway
Yes on both halves, but in different papers: RF-only studies show saturation at roughly 40–60 fields per class / ~500 fields total (Schulthess 2023) or near-insensitivity to sample size (Moraes 2021), while deep-only sweeps (EuroCropsML, Cheng) are still rising at the largest budgets tested. No single paper shows both curves on the same axes.

### Cited Findings
- Schulthess et al. (2023), Remote Sensing 15:608, Yaqui Valley Mexico, Sen2-Agri RF, >6,000 labelled fields, 9 crops, 6 random training draws per scenario. Equal designs 10–80 fields/crop; Ratio designs 0.08–0.64 of fields (876, 1,509, 3,018 fields for the three largest). OA "gradually increased with the number of fields ... to about 0.9 when the impact of adding more training fields leveled off"; for Equal "the increase started to level off after 540 fields in total, or 60 fields per crop"; for Ratio "only a slight improvement ... above 876"; Ratio-0.18 (876 fields) and Equal-80 (720 fields) both OA 0.92. "Using less than 30 fields yielded unstable results." F-score breakpoint (<2.5% gain per +10 fields) "around 40 to 45 fields" per crop, "resulting in a total of around 500 fields". Proportional (Ratio) sampling gave higher per-crop F-scores than balanced. Validation fields drawn from the same region (no spatial holdout stated). — [Schulthess et al. 2023, DOI 10.3390/rs15030608](https://doi.org/10.3390/rs15030608)
- Kaijage, Belgiu & Bijker (2024), Sensors 24(7):2108, RF only, monthly NDVI S2. Netherlands (7 classes): 350 samples OA 84%; 97 spatially-explicit AL samples 80%; 97 random samples 70%. Belgium (8 classes): 560 samples OA 65%; 223 AL samples 60%; 223 random 54%. RF loses 11–14 OA points when randomly cut to ~28–40% of labels. — [Kaijage et al. 2024, DOI 10.3390/s24072108](https://pmc.ncbi.nlm.nih.gov/articles/PMC11014375/)
- Moraes et al. (2021, IGARSS; IEEE, paywalled), Portugal, RF on multitemporal S2: RF "is considered to be of low sensitivity to variations in sample size" and results "revealed similar classification accuracies after major reductions in sample size" (snippet only; rcaap record is bot-protected, numbers not retrieved). — [RCAAP record](https://www.rcaap.pt/vufind/PublicationRecord/run_c50b3aafa4d22a5e333ef3cd79040cc2)
- Noi & Kappas (2017, pre-window): RF/SVM/kNN over 14 sample sizes on S2; SVM least sensitive, then RF, then kNN (snippet). — [PMC5796274](https://pmc.ncbi.nlm.nih.gov/articles/PMC5796274)
- Deep-only rising curves: EuroCropsML from-scratch transformer 0.438 → 0.490 → 0.518 → 0.573 → 0.632 at 10/20/100/200/500 shots/class, no plateau (Table 4a). — [arXiv 2504.11022](https://arxiv.org/abs/2504.11022)
- Cheng et al. 2024: TempCNN PASTIS mIoU 29 → 43 → 58 → 76 at 5/10/20/50 shots and 96.5 at full data, still steeply rising at 50 shots. — [arXiv 2405.18119](https://arxiv.org/abs/2405.18119)
- Representation-learning paper (arXiv 2409.19439): best MLP and RF lost only ~1–2 points on crop classification when training examples dropped from 10,000 to 1,000 (snippet only; table of LR/MLP/RF at 1k vs 10k). — [arXiv 2409.19439](https://arxiv.org/pdf/2409.19439)

### Inferences
- Schulthess's RF saturation at ~45 fields/class and ~500 fields total maps closely onto our LR/XGBoost plateau beginning around 15% of fields (~400 fields, ~45/class for 9 classes), and his "<30 fields unstable" threshold matches our 10% point (~30/class) where all models converge and RF/XGBoost dip.
- The literature therefore supports our curve shapes piecewise (classical saturates early; deep keeps rising) but has not previously put them on one plot with a shared out-of-region test.

### Gaps
- No paper found with both model families on the same learning curve beyond Portalés-Julià's qualitative statement.
- Schulthess's actual per-size OA values are in Figure 4 (box plots) and not in text; only the levelling points and the 0.92 endpoints are quotable.

---

## Key question 5: Any paper reporting that learning-curve shape depends on preprocessing / imputation?

### Takeaway
No paper was found that reports learning-curve shape changing with imputation or gap-filling. The closest evidence is that deep models trained directly on cloud-gapped (unfilled) series can match or beat gap-filled inputs, that learned reconstruction beats linear/Whittaker gap filling as a front end to Bi-LSTM/TempCNN, and that RF is indifferent to the fill method; none of these vary label count.

### Cited Findings
- EGU 2026 abstract: a temporal masked autoencoder treating cloud occlusion as a mask reconstructs growth curves "even under extreme conditions with only 20% valid observations", and its classifier "achieved higher accuracy compared to Bi-LSTM and Temp-CNN models" that used linear interpolation and Whittaker smoothing for gaps (conference abstract, no numbers, no label sweep). — [EGU26-8646](https://meetingorganizer.copernicus.org/EGU26/EGU26-8646.html)
- MDPI Remote Sensing 2021, 13:2790: CNN/LSTM/GRU/hybrid models on S2 series with random cloud gaps; concludes crops "could be identified by deep learning features in Sentinel-2 dense time series images with missing information due to clouds"; some models reached higher accuracy on unfilled than filled data (snippet, hedged). — [Remote Sensing 13:2790](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-02790/article_deploy/remotesensing-13-02790.pdf)
- Bertellini et al. 2023 (binary crop RF): linear interpolation and out-of-scale fill values gave equivalent RF accuracy (snippet). — [Annals CSIS](https://annals-csis.org/Volume_35/drp/1703.html)
- SITS-Former (Yuan et al. 2022) and SITS-BERT pretrain by regressing masked/contaminated observations, i.e. imputation is the pretext task that later yields low-label gains, which links imputation quality to label efficiency only indirectly. — [SITS-Former record](https://pure.nwpu.edu.cn/zh/publications/sits-former-a-pre-trained-spatio-spectral-temporal-representation/)

### Inferences
- Our finding that an imputation choice produced an artefactual "flat" deep-model learning curve has no published precedent that I could find; the literature treats gap filling and label efficiency as separate questions.

### Gaps
- No study found that crosses imputation method × training fraction × model family. This is a plausible second novelty claim but should be stated as "to our knowledge".

---

## Quick reference table of quantitative points (for the writer)

| Paper | Region / data | Labels | Metric | Classical | Deep | Spatial holdout? |
|---|---|---|---|---|---|---|
| Kerner 2020 Togo | S2 monthly, binary cropland | 1,319 pts | F1 (Acc) | RF 0.76 (0.81), SVM 0.73 (0.79) | LSTM local-only 0.72 (0.71); multi-head +global 0.74 (0.83) | Same country, random test pts |
| Tseng 2023 Presto / CropHarvest | Kenya/Brazil/Togo, binary | 1,345 / 203 / 1,319 | F1 | RF 0.559 / 0.000 / 0.756 | TIML 0.838/0.835/0.732; PrestoR 0.816/0.891/0.798 | Within-country |
| Reuss 2025 EuroCropsML (Sci Data) | S2 parcels, Latvia→Estonia, 127 cls | k/class, k=1…500 | Acc | none | scratch 0.04→0.57; LV-pretrained 0.17→0.66 | Yes (country) |
| Reuss 2025 arXiv 2504.11022 | same | k=1…500 | Acc | none | scratch 0.17→0.632; transfer 0.143→0.575; ANIL →0.652 | Yes |
| Mohammadi 2024 | Ghana pixels, 5-way | 1/5/10/20 shots | macro-F1 % | DTW 40.3/56.9/64.2/70.9 | Baseline 42.5/57.3/61.4/64.7; α-TIM 43.4/60.4/68.8/75.2 | Source→target domains |
| Cheng 2024 | PASTIS pixels | 5/10/20/50 shots | mIoU % | compressor 33.9/41.1/49.3/62.6 | TempCNN 29.2/42.5/57.5/75.6; Transformer 28.7/41.1/58.7/74.3 | No |
| Schulthess 2023 | Yaqui Valley MX, 9 crops | 10–80/crop; 876–3,018 total | OA | RF plateaus ~0.9 after ~540 fields (60/crop); 0.92 at 720–876 | none | No |
| Kaijage 2024 | NL 7 cls / BE 8 cls | 350→97 ; 560→223 | OA | RF 84→70 (random) ; 65→54 | none | No |
| Portalés-Julià 2021 | Valencia ES, S2, 8–9 cls | not stated | OA | RF ≥ DL when data reduced | BiLSTM 98.2% with abundant data | No |
| Ours (unpublished) | W. Cape, 10 comps × 6 bands, 9 cls | 270→2,700 fields, 10 draws | macro-F1 | LR 0.55→0.61; XGB 0.51→0.61 | L-TAE/TempCNN/CNN-BiLSTM 0.55→0.65; TabNet →0.57 | Yes (adjacent tile) |
