# Improving crop type mapping by integrating LSTM with temporal random masking and pixel-set spatial information

**Citation:** Xinyu Zhang, Zhiwen Cai, Qiong Hu, Jingya Yang, Haodong Wei, Liangzhi You, and Baodong Xu (2024). *ISPRS Journal of Photogrammetry and Remote Sensing*, 218, 87-101. DOI: `10.1016/j.isprsjprs.2024.10.013` (verified against Crossref on 2026-10-21 and again on 2026-11-02: the title matches). Code and the region A data are at [ZhiwenCai-RS/Mask-PSTIN](https://github.com/ZhiwenCai-RS/Mask-PSTIN) (commit 4a03467; the paper's `BruceKai/Mask-PSTIN` URL redirects there). The supplement is open as `1-s2.0-S0924271624003897-mmc1.docx` on ars.els-cdn.com.

**Source note:** this briefer was rewritten on 2026-11-02 from the full text, an interlibrary-loan copy kept locally as `... (ILL full text, do not commit).pdf` because its cover sheet forbids redistribution. It also uses the supplement and the code. The tracked PDF in this folder is still the ScienceDirect landing page.

## Objectives

The paper proposes `Mask-PSTIN`, which pairs a training-time *temporal random masking* augmentation with `PSTIN`. `PSTIN` is a pixel-set aggregation encoder (PSAE) over a 5 by 5 patch, followed by a two-layer attention BiLSTM (AtBiLSTM). The authors argue that clouds and rain leave "gaps and noise" in satellite time series, that small noisy training sets make deep models "overfit a specific temporal pattern", and that interpolation "may not reliably reconstruct time series" (citing Che et al. 2024). They choose augmentation over gap filling. They also aim to improve early-season mapping and to explain the model with gradient-based feature importance.

## Methods

**Data.** The data are Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`), using `B2`, `B3`, `B4`, `B8` and `B11` at 10 m. Pixels with Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) below 0.65 are removed, and the rest go into 20-day composites. The reducer is not stated, and neither is what an empty composite receives. There are three regions, all in 2021:

- **Region A, Auvergne, France:** 2020-11-01 to 2021-11-01, 19 steps. RPG labels in 7 classes give 7,403 points, split 7:3, with 1,919 of the training points held out for early stopping.
- **Region B, Otog Front Banner, Inner Mongolia:** 2021-05-01 to 2021-10-27. A roadside field survey gave 510 points, split 5:5.
- **Region C, Jianghan Plain, Hubei:** 2020-10-01 to 2021-10-01. A roadside survey gave 886 points in rotation classes, split 6:4.

**Gap handling (from code).** `dataset/mydataset.py` does `feat[np.isnan(feat)] = 0` for every sample at training and test time, so empty composites are zero-filled. The released arrays are `uint16`, so empty composites are stored directly as zero. Augmentation draws `torch.rand(T) < mask_probability` and sets those whole time steps to zero (`feat[mask] = 0`) for a random half of the training samples. The masking threshold `τm` is 0.40 for A, 0.10 for B and 0.25 for C, chosen per region from accuracy curves whose evaluation split is not stated (Supplement Part III). No mask indicator is passed to the LSTM, so a natural gap and an augmented gap look identical.

**Model and training.** PSAE uses the centre pixel as the query and its neighbours as keys and values. Its output feeds a BiLSTM with temporal attention, trained with focal loss (`γ = 1`) and Adam. Baselines are RF (500 trees), RBF SVM, a Transformer and a CNN-LSTM, plus the ablation ladder LSTM, Mask-LSTM, PSTIN and Mask-PSTIN.

**Evaluation protocol:** a random point split within one region and one year, evaluated separately for each region. This is in-region validation reported as test accuracy, so the main tables measure no transfer. Points are labelled pixels, and the paper does not say whether one parcel can contribute points to both train and test. The 5 by 5 patches around nearby points can overlap, so spatial leakage is likely in region A, where 7,403 points are drawn from RPG parcels. The only transfer tests are in Supplement Part VI, for region A only, and are shown as per-class bar charts (Figs. S4 and S6) with no numbers in the text. One is temporal: train on 2019 and 2020 (3,316 points each) and test on 2021. The other is spatial: train on Allier (T1) and test on Puy-de-Dôme (T2). The metrics are OA, per-class F1 and mF1 (macro). Early-season analysis truncates the series and treats a crop as identified once its F1 exceeds 0.8.

## Key Findings

- **Ablation, as mean OA and mF1 across regions:**
  - LSTM: 80.9 % and 0.781.
  - Mask-LSTM: 84.5 % and 0.820.
  - PSTIN: 83.9 % and 0.818.
  - Mask-PSTIN: 87.4 % and 0.865.
  - Masking alone adds about 3.6 OA points, and every non-zero `τm` beat `τm = 0` in every region.
- **Against baselines:** Mask-PSTIN beats RF, SVM, the Transformer and the CNN-LSTM in every region, with mF1 of 0.855 for A, 0.863 for B and 0.878 for C. RF reaches 0.785, 0.813 and 0.818. The largest gains are for minority classes, such as rapeseed in A, which rises from 0.829 to 0.907 with masking.
- **Early season:** Mask-PSTIN reaches F1 > 0.8 earlier than RF and SVM. In region B, corn is identified at DOY 200 against 220 for RF.
- **Feature importance:** models trained without masking concentrate their gradients on a few steps, while masked models spread attention across the phenological cycle.
- **Supplement Part VI:** Mask-PSTIN trained on 2019 and 2020 transfers well to 2021 for corn, sunflower and rapeseed. For wheat, rye and barley it shows "notable performance degradation", "attributed to the year-to-year phenological shifts". The spatial model trained on Allier matches or beats the model trained on Puy-de-Dôme. RF transfers worse in both tests.
- **Adding SWIR1:** adding `B11` to RGB+NIR raises mF1 by about 0.03.

## Relevance to Our Crop-Classification Study

**Closest published analogue to our remedy.** Temporal random masking zeroes whole time steps without an indicator, which is in effect observation dropout on a zero-filled input. The paper is the closest published analogue to the remedy our missingness analysis points to: zero-fill a missing month, then train the model to expect zeros. It shows the remedy helps even in-region, at +3.6 OA points.

**Measured zero fill in the released data** (`experiments/external_audits/mask_pstin/`). In the released region A 2021 arrays, 9.9 % of composites are zero-filled, and 87.6 % of points have at least one zero composite. The zeros cluster in midwinter, at 44 % for 31 December to 19 January and 63 % for 20 January to 8 February. Training at `τm = 0.40` on half the samples zeroes about 20 % of steps, roughly twice the natural rate.

**Missingness audit.** This is rank 2 in `reports/Missingness confound audit of transfer papers.md`. Gap handling is identical across domains, but the paper never reports how often the fill fired in each year. The cereal loss in the 2019-2020 to 2021 transfer falls on the classes whose winter composites are most often empty, but the 2019-2020 data are not released.

**Design overlaps.** It shares our band choice of `B2`, `B11` and visible/NIR bands, a winter-cereal problem (wheat, barley, rye and rapeseed against our wheat, barley and canola), and patch-level spatial context. The result that a small spatial encoder plus regularisation beats RF is in-region only, so it says nothing about our finding that sparse models transfer better. The supplement's RF comparisons are bar charts with RF's own gap handling unstated.

## Evaluation Caveats

- **In-region results reported as test accuracy.** Every headline number comes from random within-region, within-year splits, which our holdout-tile evaluation is built to correct. `τm` was chosen per region, and the split used to choose it is not stated.
- **Transfer tests are thin.** They sit in the supplement, cover one region, are shown as bars without numbers or confidence intervals, and come from a single run.
- **Minority classes.** Regions B and C have tiny classes (Rape-Upland has 21 training points), so per-class F1 is noisy. There are no seeds or variance estimates anywhere.
- **Unreported gap statistics.** No valid-observation count is reported for any region, year or department. The zero-fill rule is in the code but not in the paper.
- **Release.** Only region A 2021 is released. The 2019-2020 transfer data and regions B and C are not.
