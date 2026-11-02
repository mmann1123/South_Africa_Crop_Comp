# Improving crop type mapping by integrating LSTM with temporal random masking and pixel-set spatial information

**Citation:** Xinyu Zhang, Zhiwen Cai, Qiong Hu, Jingya Yang, Haodong Wei, Liangzhi You, and
Baodong Xu (2024). "Improving crop type mapping by integrating LSTM with temporal random masking and
pixel-set spatial information." *ISPRS Journal of Photogrammetry and Remote Sensing*, 218, 87-101.
DOI: `10.1016/j.isprsjprs.2024.10.013` (extracted from the PDF first page; verified against Crossref
on 2026-11-02: title, journal, volume 218, pages 87-101, and author list Zhang, Cai, Hu, Yang, Wei,
You, Xu all match). Zhang and Cai contributed equally; Hu and Xu are corresponding authors.

Sources for this briefer: the full text (a Gelman Library interlibrary-loan scan, kept locally as
`... (ILL full text, do not commit).pdf` because its cover sheet forbids further distribution; the
tracked PDF of the same base name in this folder is still the ScienceDirect landing page), the open
supplement (`1-s2.0-S0924271624003897-mmc1.docx` on `ars.els-cdn.com`), and the authors' code at
[ZhiwenCai-RS/Mask-PSTIN](https://github.com/ZhiwenCai-RS/Mask-PSTIN), commit `4a03467` (the
paper's `github.com/BruceKai/Mask-PSTIN` URL redirects there). This briefer replaces the
partial-extraction briefer written on 2026-10-21 from the landing page alone.

## Objectives

The paper proposes `Mask-PSTIN` (Mask Pixel-set SpatioTemporal Integration Network) for
post-harvest and early-season crop type mapping from Sentinel-2 time series. It has two parts:

1. **Temporal random masking**, a training-only data augmentation that sets the inputs at randomly
   chosen time steps to zero. The stated motivation is that "gaps and noise in satellite
   observations caused by clouds and rain" combined with "insufficient and noisy training data" lead
   deep models to "overfit a specific temporal pattern or noise present in the training data."
   Masking is meant to force "more generalized and invariant features."
2. **PSTIN**, a patch model in which a pixel-set aggregation encoder (`PSAE`) pools spectral
   features from a 5 by 5 neighbourhood under the guidance of the centre pixel, followed by a
   two-layer bidirectional LSTM with temporal attention (`AtBiLSTM`, after Xu et al. 2020).

The authors explicitly position augmentation as an alternative to gap reconstruction: interpolation
"may not reliably reconstruct time series, offering limited improvement in crop type mapping and
increasing computational costs" (citing Che et al. 2024, "Linearly interpolating missing values in
time series helps little ...", and Zhao et al. 2021). They also cite the closest prior augmentation
work: Garnot et al. (2022) temporal dropout across modalities, and Pham et al. (2024a, 2024b)
random observation selection for temporal transferability. They note that these "were mainly
tested in single agricultural landscapes" and set out to test generalisability "across diverse crop
types and landscapes".

The PSAE is motivated by smallholder settings without parcel boundaries. Pixel-set encoders
(Garnot et al. 2020) need parcel polygons, and naive pooling over an image patch mixes neighbouring
crops. PSAE weights neighbours by their similarity to the centre pixel, as a "spatial attention
module."

Stated contributions: (1) the masking augmentation; (2) PSAE plus AtBiLSTM; (3) a gradient-based
feature-importance analysis; and (4) evaluation in three regions "characterized by diverse
agricultural landscapes and cropping patterns."

## Methods

### Study regions and labels

- **Region A, Auvergne, central France** (44.6 to 46.8 N, 2 to 4.5 E; Allier, Puy-de-Dôme, Cantal,
  Haute-Loire). Temperate, cropland 74 percent of area, mean parcel 2.37 ha. Labels come from the
  2021 Registre Parcellaire Graphique (RPG), 177 categories consolidated into seven classes: wheat,
  corn, rye, barley, sunflower, rapeseed, others. There are 7,403 point samples split 7:3 into
  5,235 training and 2,168 test points; 1,919 of the training points are held out for early
  stopping, leaving 3,316 for fitting (Table 1).
- **Region B, Otog Front Banner, Inner Mongolia, China.** Temperate continental, 276 mm rain, mean
  parcel 1.16 ha. Classes: alfalfa, chili, corn, others. One field survey in June 2021, along roads,
  with handheld GPS (under 5 m error); non-crop points interpreted on Google Earth. There are 510
  points split 5:5.
- **Region C, five counties of the Jianghan Plain, Hubei, China.** Humid, 1,300 mm rain, frequent
  cloud, mean parcel 0.51 ha. Classes are rotation systems: fallow-rice, rape-rice, rice crayfish,
  wheat-rice, rape-upland, wheat-upland, others. Two roadside surveys (April and September 2021)
  gave 886 points split 6:4.
- Five-fold cross-validation is used to tune hyperparameters in regions B and C "due to the limited
  number of samples."

Class balance is far from even. In region A the test set holds 685 others, 614 wheat and 446 corn
against 46 rapeseed. In region C, rape-rice has 16 training points and rape-upland 21.

### Satellite data and preprocessing

- **Sensor:** Sentinel-2 L2A surface reflectance from Google Earth Engine
  (`COPERNICUS/S2_SR_HARMONIZED`). Optical only, no SAR.
- **Bands:** five, `B2`, `B3`, `B4`, `B8` and `B11` (blue, green, red, NIR, SWIR1), resampled to
  10 m. No red-edge bands and no indices. The authors suggest red-edge for future work.
- **Cloud masking:** pixels with a Cloud Score+ quality score below 0.65
  (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) are removed.
- **Compositing:** 20-day composites (citing Lin et al. 2022 and You and Dong 2020). The reducer
  (median, mean, or quality mosaic) is not stated.
- **Windows:** region A from 2020-11-01 to 2021-11-01 (19 steps), region B from 2021-05-01 to
  2021-10-27, and region C from 2020-10-01 to 2021-10-01.
- **Empty composites:** the paper does not say what a composite with no valid observation receives.
  The released loader (`dataset/mydataset.py`) answers it: `feat[np.isnan(feat)] = 0`, applied to
  every sample at training and test time, followed by division by 10,000. Empty composites are
  therefore zero-filled, with no mask channel. The released region A arrays are `uint16`, so empty
  composites are already stored as zero.

### Temporal random masking (Eq. 1 and code)

For a sequence of length `T`, a random vector `M_t` is drawn and `X_t` is set to zero wherever
`M_t < τm`. In the released code this happens for a random half of training samples
(`p = np.random.random(); if p < 0.5: feat[mask] = 0`), so the expected share of zeroed steps is
`τm / 2`. The thresholds are `τm` = 0.40 (region A), 0.10 (B) and 0.25 (C), chosen from accuracy
curves over `τm` in Supplement Part III. The supplement does not say which split those curves were
computed on. Masked steps are indistinguishable from naturally empty composites: both are all-zero
vectors with no indicator. No masking is applied at inference.

### PSTIN architecture (Supplement Table S1)

- `PSAE`: a shared MLP (`C0 → 32 → 64`) over each of the 25 patch pixels at each time step; the
  centre pixel yields the query (`64 → 16`) and every pixel a key (`64 → 16`) and value
  (`64 → 64`). Softmax-normalised similarities weight the values, which are combined with the
  centre pixel's features. Output `(B, T, 64)`.
- `AtBiLSTM`: two-layer BiLSTM (`64 → 512`), a shared fully connected attention layer with softmax
  over time steps (Eqs. 2-3), and a linear output layer.
- Loss: focal loss with `γ = 1` (Eq. 4), to soften class imbalance. Optimiser: Adam.
- The 5 by 5 patch size follows Ienco et al. 2019, Interdonato et al. 2019 and Yuan et al. 2022.

### Baselines and ablations

- Ablation ladder: LSTM (the `AtBiLSTM` on the centre pixel only), Mask-LSTM, PSTIN, Mask-PSTIN.
- Classical: random forest (`n_estimators = 500`, `max_features = sqrt`) and RBF-kernel SVM. Their
  inputs and gap handling are not described, and neither appears in the released code.
- Deep: a Transformer (hidden size 512, 8 heads, 2 layers, feed-forward 2048) on the centre pixel,
  and a CNN-LSTM that replaces PSAE with two 3 by 3 convolutions (32 and 64 kernels). Both use focal
  loss and Adam.

### Metrics

Producer's and user's accuracy, overall accuracy (OA), per-class F1 and mean F1 (`mF1`, i.e. macro
F1) from confusion matrices. For early-season mapping, a crop counts as "identified" once its F1
exceeds 0.8 as the series is truncated at successive dates.

**Evaluation protocol:** In-region, single-year, random point split, reported as "testing" accuracy.
Each region is trained and tested on a random 7:3, 5:5 or 6:4 split of points from the same region
and the same 2021 season, so every headline number in Tables 2 and 3 sits at the first rung of the
spectrum this manuscript is built around, or at best the second. It is in-region validation
presented as test accuracy, and it measures no spatial or temporal transfer.

Spatial leakage is not ruled out. The unit of sampling is a labelled pixel with its 5 by 5 patch,
and the paper does not say whether points were drawn one per parcel or whether a parcel can supply
points to both train and test. The roadside survey design in regions B and C clusters points along
routes, and in region A 7,403 points are drawn from RPG parcels across four departments with no
spatial blocking. Neighbouring points' patches can overlap.

The only transfer experiments are in Supplement Part VI, for region A only, reported as per-class
F1 bar charts (Figs. S4, S6) with no numbers in the text, no confidence intervals and one run each:

- **Temporal:** train on RPG 2019 and 2020 (3,316 points per year), test on 2021. Compared with RF
  trained the same way and with Mask-PSTIN trained on 2021.
- **Spatial:** train on Allier (T1), test on Puy-de-Dôme (T2). Compared with RF trained on T1 and
  with Mask-PSTIN trained on T2.

These are the only results in the paper at the third and fourth rungs: a disjoint department, and a
new year.

## Key Findings

### Ablation (Table 2)

| Model | OA A | OA B | OA C | Mean OA | mF1 A | mF1 B | mF1 C | Mean mF1 |
|---|---|---|---|---|---|---|---|---|
| LSTM | 80.9 | 80.1 | 81.7 | 80.9 | 0.767 | 0.801 | 0.776 | 0.781 |
| Mask-LSTM | 83.3 | 83.2 | 87.0 | 84.5 | 0.804 | 0.838 | 0.817 | 0.820 |
| PSTIN | 81.7 | 83.2 | 86.8 | 83.9 | 0.797 | 0.832 | 0.826 | 0.818 |
| Mask-PSTIN | 86.2 | 85.9 | 90.1 | 87.4 | 0.855 | 0.863 | 0.878 | 0.865 |

- Masking alone adds 3.6 OA points and 0.038 mF1 on average. PSAE alone adds 3.0 and 0.037. The two
  together add 6.5 and 0.084.
- Supplement Part III: every non-zero `τm` beats `τm = 0` in all three regions, with accuracy rising
  to a peak and then falling as masking removes too much phenology.
- The masking gain is largest for small classes. For region A rapeseed (116 training points),
  F1 rises from 0.829 (LSTM) to 0.907 (Mask-LSTM) and from 0.867 (PSTIN) to 0.933 (Mask-PSTIN).
- PSAE helps most with phenologically similar cereals: barley F1 rises by 0.118 from LSTM to PSTIN
  in region A. It also reduces salt-and-pepper noise in the maps (Fig. 10).

### Comparison with baselines (Table 3)

| Model | OA A | OA B | OA C | mF1 A | mF1 B | mF1 C |
|---|---|---|---|---|---|---|
| RF | 82.4 | 82.0 | 85.9 | 0.785 | 0.813 | 0.818 |
| SVM | 78.0 | 79.3 | 85.1 | 0.693 | 0.790 | 0.814 |
| Transformer | 81.3 | 85.0 | 86.8 | 0.763 | 0.844 | 0.839 |
| CNN-LSTM | 84.6 | 84.8 | 88.7 | 0.835 | 0.848 | 0.871 |
| Mask-PSTIN | 86.2 | 85.9 | 90.1 | 0.855 | 0.863 | 0.878 |

- In region A, RF (0.785) beats the pixel Transformer (0.763) and the unmasked pixel LSTM (0.767)
  in-region. The authors attribute the Transformer's weakness there to wheat, rye and barley sharing
  similar phenology.
- Note that the conclusion quotes region A OA as 86.5 percent while Tables 2 and 3 give 86.2.

### Early season (Fig. 12)

Mask-PSTIN reaches the F1 > 0.8 threshold earlier than RF and SVM for most crops. Wheat, rapeseed
and sunflower in region A are identified mid-development, and corn in region B at DOY 200, 20 days
before RF. RF and SVM are the only early-season baselines.

### Feature importance (Fig. 13)

Absolute input gradients show that unmasked models concentrate on narrow periods. LSTM uses the
fallow stage (DOY 365 to 160) for fallow-rice in region C, and PSTIN uses only the rice stage for
rape-rice. Masked models spread importance across the phenological cycle, and PSAE makes the
important steps sparser.

### Training-set size and bands (Supplement Parts IV and V)

- Mask-PSTIN beats RF at every training size tested in region A and matches RF with half the
  samples (1,328 against 2,656). Both are weak at 332 samples for 7 classes.
- Adding `B11` (SWIR1) to RGB+NIR raises mF1 by about 0.03 on average. Mask-PSTIN on four bands
  still beats RF on five (region A mF1 0.806 against 0.785).

### Transfer (Supplement Part VI)

- **Temporal (2019 and 2020 to 2021):** Mask-PSTIN transfers better than RF and is close to the
  2021-trained model for corn, sunflower and rapeseed. For wheat, rye and barley it shows "notable
  performance degradation", which "is attributed to the year-to-year phenological shifts". No
  per-year cloud, valid-observation or zero-fill statistic is reported.
- **Spatial (Allier to Puy-de-Dôme):** Mask-PSTIN trained on Allier matches or beats Mask-PSTIN
  trained on Puy-de-Dôme, which the authors attribute to Allier's larger samples (for example
  rapeseed). RF trained on Allier is worse for almost every class.
- The main-text Discussion summarises this as "good spatial transferability across almost all crop
  types" and temporal limits "for crops with similar features due to the phenological shifts."

## Relevance to Our Crop-Classification Study

### The protocol is the conflation our paper corrects

All headline comparisons, including "Mask-PSTIN significantly outperformed traditional machine
learning and deep learning methods", come from random within-region, within-season point splits.
Their 0.855 to 0.878 mF1 is therefore comparable to our in-region field-wise k-fold numbers, not to
our holdout-tile (`34S_20E_259N`) macro-F1, and probably optimistic even against those because the
split is by point, not by field. Our central finding is that in-region rankings of dense temporal
and patch deep nets against tree models do not survive a spatially disjoint holdout. This paper
offers no main-text evidence either way. Its ranking of a patch BiLSTM over RF is exactly the kind
of in-region ranking we show can reverse.

The supplement's spatial test (Allier to Puy-de-Dôme) is the one result at our rung, and in it the
deep patch model transfers well and RF poorly. That runs against our finding, but it is a single
run, shown as bars, between two adjacent departments of one region and one year, with RF's inputs
and gap handling undescribed. It is a weak counter-example, not a contradiction, and worth citing as
such alongside `pankajakshan2026`, which finds the opposite with a proper cross-domain design.

### Masking is the remedy our missingness analysis points to

Temporal random masking writes zeros over whole time steps with no mask indicator, which is exactly
what the loader does to naturally empty composites. It is in effect observation dropout on a
zero-filled input: training the model to expect the zeros it will meet at test time. That is the
control our own holdout analysis calls for. In our data, 40.6 percent of holdout pixels lack a
September observation while the training tiles have none missing, and zero-filling that month
rather than the region change drove most of the CNN-BiLSTM's apparent transfer drop (macro-F1 0.51
to 0.66 once September is filled from training-tile medians). Mask-PSTIN shows the remedy helping
even in-region (+3.6 OA, +0.038 mF1), with the largest gains for small classes. Together with Pham
et al. 2024 (random observation selection), it is citable prior art that training on simulated
gaps improves robustness, and it is direct support for adding month dropout to our deep models
before crediting tree models' transfer advantage to architecture alone.

### Measured zero fill in the released data

`experiments/external_audits/mask_pstin/` counts zero-filled composites in the released region A
2021 arrays (`data/{train,val,test}/data.npy`, exactly the paper's 3,316 / 1,919 / 2,168 split):

- 9.9 percent of centre-pixel composites in the test split are all zero (9.5 percent over the whole
  patch). 87.6 percent of points have at least one, with a mean of 1.9 and a maximum of 7 of 19.
- The zeros are seasonal: 44 percent of points are empty for 31 December to 19 January and 63
  percent for 20 January to 8 February, with smaller peaks of 12 to 17 percent in early March,
  early May and early July. The final window covers only five days and is 31 percent empty.
- Classes differ little (9.1 percent sunflower to 11.3 percent rye), and train, validation and test
  match, as a random within-year split must.
- Training at `τm = 0.40` on half the samples zeroes about 20 percent of steps, twice the natural
  rate, so zero composites are in distribution for the trained model.

In `reports/Missingness confound audit of transfer papers.md` the paper is **rank 2, confidence
medium**. The gap handling is identical across domains, but the paper never reports how often it
fired in each year. The supplement's cereal losses fall on the classes whose midwinter composites
are most often empty, and the authors attribute them to phenology without a missingness test. The
2019-2020 arrays that would test this are not released.

### Feature and sensor design

- Optical only, five bands (`B2`, `B3`, `B4`, `B8`, `B11`), close to our `B2`, `B6`, `B11`, `B12`
  plus `EVI` and `hue`. Their SWIR1 ablation (+0.03 mF1) is consistent with the value of SWIR in our
  band set.
- Raw 20-day composite sequences, with no engineered time-series features. There is no comparison
  to an xr_fresh-style feature set, and RF receives the same raw sequence, so the paper cannot
  speak to our finding that engineered features and raw reflectance transfer about equally.
- Patch granularity (5 by 5 at 10 m, so 50 by 50 m) with similarity-weighted pooling, against our
  pixel, field and patch models. PSAE is a lighter-weight alternative to our patch CNNs and could be
  tested on our holdout tile.
- Crop mix: winter cereals (wheat, barley, rye) and rapeseed overlap with our wheat, barley and
  canola, and the confusion between wheat, rye and barley mirrors our wheat and barley confusion.

### What the paper does not measure

No spatial transfer in the main text, no cross-year transfer in the main text, no field-wise
splitting, no seeds or variance, no per-domain valid-observation counts, no test-time gap
experiment (for example scoring models on series with additional dropped steps), no computational
cost, and no comparison against gradient-boosted trees, which is the model family that transfers
best in our study.

## Evaluation Caveats

- **In-region results reported as test accuracy.** Every number in Tables 2 and 3 and Figs. 9 to 13
  comes from a random within-region, within-year point split. These are in-region accuracies and
  should not be compared with our holdout-tile macro-F1.
- **Spatial leakage likely.** Point-level splitting with overlapping 5 by 5 patches, no stated
  one-point-per-parcel rule, and roadside-clustered sampling in regions B and C.
- **Hyperparameters chosen per region.** `τm` was selected per region from accuracy curves whose
  evaluation split is not stated, and the masking gain is reported at each region's best `τm`.
- **Transfer evidence is thin.** It sits in the supplement, covers region A only, is shown as bar
  charts without numbers or intervals, and comes from one run per model. The spatial test is
  between adjacent departments in the same year.
- **Class imbalance.** Focal loss and mF1 are used, which is good practice, but several classes have
  10 to 37 test points (region C rape-upland 10, region B alfalfa 26), so per-class F1 swings are
  large and no variance is given. OA in region A is dominated by others, wheat and corn.
- **Undocumented gap handling.** The zero-fill rule for empty composites appears only in the code,
  the compositing reducer is not stated, and no per-region or per-year valid-observation statistic
  is reported. The RF and SVM baselines' handling of empty composites is unknown.
- **Minor inconsistencies.** The conclusion's region A OA (86.5 percent) differs from the tables
  (86.2 percent).
- **Partial release.** The code and region A 2021 arrays are public. The 2019-2020 transfer data,
  the Allier and Puy-de-Dôme split, and regions B and C are not released, so none of the transfer
  results can be re-run.
