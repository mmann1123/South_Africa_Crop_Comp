# Linearly interpolating missing values in time series helps little for land cover classification using recurrent or attention networks

**Citation:** Xianghong Che, Hankui K. Zhang, Zhongbin B. Li, Yong Wang, Qing Sun, Dong Luo, and
Hao Wang (2024). *ISPRS Journal of Photogrammetry and Remote Sensing*, 212, 73-95.
DOI: `10.1016/j.isprsjprs.2024.04.021` (verified against Crossref on 2026-10-21: title, journal,
volume, pages, and author list match). Open access (CC BY). Training data:
`https://doi.org/10.5281/zenodo.7944466`; code:
`https://github.com/chexianghong/Bi-LSTM-and-Transformer-classification-codes`.

## Objectives

The paper tests a single, sharply posed question: when the classifier is a recurrent or attention
network that can mask out missing time steps, does linearly interpolating cloud-induced gaps in a
satellite time series improve land-cover classification at all? The authors note that almost every
deep-learning SITS classification study (they list Rußwurm and Körner 2017, 2018; Ienco et al.
2019; Interdonato et al. 2019; Tang et al. 2022; Xu et al. 2020, among others) still linearly
interpolates before training, a habit inherited from random-forest pipelines where "traditional
machine learning classifiers cannot mask out missing values," while a few studies (Metzger et al.
2021; Zhao et al. 2021; Rußwurm et al. 2023) feed gapped sequences directly, and none has compared
the two.

Specific objectives:

1. Build two parallel inputs from the same 2021 Harmonized Landsat Sentinel-2 (HLS) time series over
   the Amur River Basin: 16-day composites *with* gaps (missing periods flagged) and the same
   composites *after* linear interpolation (no gaps).
2. Classify both with a masking Bi-LSTM and a masking Transformer, and compare accuracy against the
   run-to-run randomness of model training.
3. Compare wall-clock cost of the two pipelines.
4. Check sensitivity to (a) legend (eight land-cover classes vs five vegetated classes vs 17-class
   and 10-class CDL crop legends in California and Oklahoma), and (b) gap frequency, by randomly
   masking 50 percent of observations and by sweeping the fraction of good-quality composites from
   10 to 100 percent.

## Methods

### Data

- HLS v2.0 NBAR surface reflectance (Landsat 8/9 OLI and Sentinel-2A/B MSI), 30 m, 144 MGRS tiles
  covering 890,308 km2 of north-east China, 6 March to 15 October 2021, 17,963 images (1,940 GB).
  Per tile 94 to 171 acquisition days.
- Temporal compositing to 14 fixed 16-day periods using a median-NDVI best-observation rule after
  QA cloud masking. A period with no clear observation is a missing value (coded `-9999`).
- Fourteen predictors per period: seven reflectance bands (coastal, blue, green, red, NIR, SWIR1,
  SWIR2) and seven indices (`NDVI`, `EVI`, `SAVI`, `MSAVI`, `NDMI`, `NDWI`, `NDIIb7`). Input tensor
  per pixel is 14 periods x 14 variables; standardized per variable using training-sample mean and
  standard deviation.
- 7,735 (text also says 7,755) visually interpreted 30 m pixel samples in eight classes (woodland
  1,010; grassland 1,265; rainfed farmland 1,087; paddy field 933; wetland 1,157; water 737;
  built-up 860; barren 706), stratified by GlobeLand30. Mean nearest-neighbour sample distance
  10.71 km, minimum 0.10 km; the authors argue this makes samples spatially independent.
- On average 15.35 percent of the 14 periods are missing per sample (water 29.6 percent, woodland
  20.1 percent, grassland 9.3 percent). The first period (snow) and periods 8 to 11 (summer rain)
  have the most gaps.
- Crop-type sensitivity datasets: 2021 HLS over CDL tiles T11SKA (California, 17 classes with more
  than 100 samples, 18,539 pixels combined across both tiles) and T14SPF (Oklahoma, 10 classes),
  CDL sampled every 20 pixels (600 m) and filtered for 3 x 3 label homogeneity. Roughly 40 percent of
  samples have at least one missing period.

### Missing-value handling

Two inputs are built from identical observations:

- *Without interpolation*: the composite with `-9999` at missing periods plus a binary mask vector
  `m_t` per period.
- *With interpolation*: Eq. (1), linear interpolation between the nearest valid composites before
  and after; one-sided copy when only one neighbour exists.

Models that ingest the gapped input:

- Bi-LSTM with a masking layer: when `m_t = 0` the LSTM cell is skipped and the cell and hidden
  state pass directly from `t-1` to `t+1` in each direction; four Bi-LSTM layers (32, 64, 128, 256
  units), dropout 0.5, dense ReLU layer with batch norm, softmax output.
- Transformer encoder with learnable positional encoding, attention weights at masked positions set
  to very small values, and a masked average over unmasked periods before the classifier; 4 heads,
  dropout 0.1.

The authors explicitly exclude 1D CNNs because a CNN "usually assigns missing values as zero rather
than masking out them in classification (known as zero-padding)," citing Zhao et al. (2021). The
zero-fill representation is therefore acknowledged as a distinct and inferior option but is not
tested.

Hyperparameters were grid-searched (batch size, learning rate, dropout, epochs, hidden width,
dense units; about 15 GPU-days for Bi-LSTM, 2 for Transformer). Adam, learning rate 1e-4, 80
epochs, batch 32.

**Evaluation protocol:** Five-fold cross-validation over the 7,735 pixel samples, with each fold's
training portion further split 24:1 into train and validation; every fold is run five times with
different random initialization, so each model is trained 25 times. Overall accuracy and per-class
F1 are reported per fold and per run, and the key statistic is the difference between the
with-interpolation and without-interpolation means compared against the run-to-run standard
deviation within a fold (0.1 to 0.4 percent) and the across-fold standard deviation (0.6 to 1.0
percent). Folds are random over samples, not spatially blocked, but the large mean sample spacing
(10.71 km, roughly 350 pixels) means neighbouring pixels of the same field do not straddle folds; it
is an in-region, pooled-sample k-fold over a very large region rather than a spatial holdout. The
train and test partitions have the same missingness distribution because they are random folds of
the same 2021 composites, so the missingness regime is *matched*. Imputation strategies are compared
at inference only in the sense that each model is trained *and* tested on the same representation;
there is no cross condition (train on interpolated, test on gapped, or vice versa).

## Key Findings

### Main comparison (Tables 4, 5, 6; eight classes, Amur basin)

| Model | Without interpolation OA | With interpolation OA | Difference |
|---|---|---|---|
| Bi-LSTM (5 runs, mean of 5 folds) | 90.0 to 90.4 percent | 90.2 to 90.4 percent | under 0.2 points |
| Transformer | 89.8 to 90.3 percent | 89.7 to 90.2 percent | under 0.5 points |

Both differences are smaller than the within-fold run-to-run standard deviation. Per-class F1
differences are below 0.007 (Bi-LSTM) and 0.008 (Transformer) for all eight classes and below the
run-to-run standard deviation for seven of eight. Rainfed farmland shows the largest and most
consistent (still tiny) benefit from interpolation (0.887 to 0.894 Bi-LSTM; 0.876 to 0.884
Transformer).

### Map agreement and cost (Section 4.4, 4.5)

Pixel-level agreement between the two pipelines on full tiles is 91.8 and 91.4 percent (Bi-LSTM)
and 89.1 and 87.4 percent (Transformer) for tiles 52UDV and 51TXL, so the two inputs yield
*different* maps for 8 to 13 percent of pixels at *equal* accuracy. Dropping interpolation cuts
end-to-end processing time by a factor of about 13 because interpolation dominates compute.

### Legend sensitivity (Tables 7 to 13)

Five vegetated classes only: OA difference under 0.5 points. CDL crop types, Transformer only:
California OA 88.7 to 88.9 without vs 88.8 to 89.0 with (difference under 0.3); Oklahoma 95.5 to
95.7 vs 95.4 to 95.6 (under 0.2). Per-class F1 differences under 0.006 and 0.008 except for the two
rarest classes (Double-crop triticale/corn, 120 samples, 0.147 vs 0.170; Other hay, 100 samples,
0.484 vs 0.461), where the sign is inconsistent.

### Gap-frequency sensitivity (Tables 14, 15; Fig. 19, 20)

With 50 percent of each series randomly masked in both train and test, Transformer OA is 89.1 to
89.3 without vs 89.0 to 89.5 with interpolation (difference under 0.3), and all accuracies drop by
about one point relative to the original series. Sweeping the number of good composites from 10 to
100 percent on the CDL tiles: accuracy rises then plateaus; the two pipelines are indistinguishable
once more than about 5 of 14 composites (30 percent) are present; below that, the *non-interpolated*
pipeline is 1 to 5 OA points *better*, which the authors read as linear interpolation producing
unreliable series when anchors are sparse.

### Interpretation (Section 5.1)

Linear interpolation adds no information the network does not already have: where reflectance
changes roughly linearly across a short gap, the skip-connected Bi-LSTM (carrying state across the
gap in both directions and concatenating) and the attention re-weighting over neighbouring periods
reproduce the interpolation implicitly; where reflectance changes abruptly (harvest, double
cropping, flood), the interpolated value is wrong and neither pipeline can recover the missing
dynamics. The authors conclude that historical use of interpolation "is more about mitigating the
inability of traditional classifiers to handle missing values rather than improving
classifications."

## Relevance to Our Crop-Classification Study

Our revised findings are that the holdout tile's September gap (41 percent of pixels, never missing
in training) was zero-filled before standardization in the old pipeline, that this cost every
raw-sequence deep model 0.08 to 0.14 macro-F1 and inverted the apparent in-region vs out-of-region
ranking, and that filling the month with training-tile medians restores dense temporal models to
the top (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63; TabNet 0.62;
tree, linear, MiniRocket 0.59 to 0.61). Che et al. is the cleanest published test of whether gap
*filling* matters for exactly our two dense model families (a bidirectional LSTM and a
transformer), and it bears on the revision in the following ways.

**It supports the claim that the gap itself is not the problem; the representation is.** Under a
matched regime where the network is allowed to *skip* missing periods, the presence of 15 percent
missing composites (and even 50 percent) costs essentially nothing relative to a filled series:
under 0.2 OA points for Bi-LSTM, under 0.5 for Transformer, per-class F1 within 0.008. In our
pipeline the same two architectures lost 8 to 14 macro-F1 points when the gap was represented as a
standardized zero. The combination says the loss we observed cannot be attributed to "a dense
network needs every time step"; both of their networks are indifferent to the step being absent
when told it is absent. What they never saw was a plausible-looking but wrong value at the missing
position, which is what post-standardization zero is. Che et al. therefore support revised finding
(1) that the inversion was an imputation artifact, and their Section 3.2 rationale for excluding
CNNs ("CNN usually assigns missing values as zero ... known as zero-padding") is a published
acknowledgement that zero-filling is a different and worse thing than masking.

**It does not test our regime, and that limit should be stated when citing it.** Their train and
test folds share the same missingness distribution; every model is trained on gapped series when
tested on gapped series. The question of a network trained on *complete* series meeting a gap only
at inference (our situation, and the Baseline arm of Wang et al. 2025) is not asked. So the paper
shows that gap handling is cheap when the training distribution contains gaps; it does not show that
gaps are harmless when the model has never seen one. Our finding (2), that a sparsemax channel gate
adds +0.06 macro-F1 of robustness to the zero-filled month, is a result about an *unseen* corruption
and is outside Che et al.'s design.

**It qualifies our median-fill choice without contradicting it.** The paper's thesis is that
interpolation adds no information, not that it harms, so a training-statistic fill is neither
endorsed nor rejected. Two of their sub-results are directly useful. First, when fewer than 30
percent of composites are good, interpolation *hurt* by 1 to 5 OA points, consistent with Wang et
al.'s interpolation losses and a warning that a constant fill is safe only when the gap is small and
well anchored (ours is one month of roughly ten, with the training tiles fully observed in that
month). Second, equal accuracy coincided with 8 to 13 percent of pixels changing class between the
two pipelines, which means a filled-vs-masked comparison can look identical in aggregate while
redistributing errors; our per-class and per-field agreement analyses should be reported alongside
macro-F1 for the zero-fill vs median-fill comparison.

**It bears on the relative ranking of masking-capable models.** Che et al. find the Bi-LSTM
slightly above the Transformer (median OA) at 30 m with 14 steps, consistent with our corrected
holdout order (CNN-BiLSTM 0.66, Transformer 0.64), but the gap is within noise in both studies and
should not be cited as a ranking result.

**Architectural implication for the revision.** The cheapest way to make our raw-sequence models
immune to this class of artifact is the one they implement: pass a per-step mask and let the LSTM
skip and the attention ignore the step. That removes the need to choose a fill value at all. Che et
al. give a working recipe (masking layer, masked average pooling) and open code for both of our
dense families. The L-TAE already has attention; a masked-attention variant would be a direct
replication of their Transformer arm on our tile. The 1D CNN family (TempCNN) cannot skip steps and
would still need a fill value or an indicator channel.

**Learning curves.** Not studied; the paper is silent on revised finding (3).

**What to cite it for.** (a) Direct evidence that, for a Bi-LSTM and a Transformer trained and
tested on gapped series, linear interpolation changes OA by under 0.5 points and per-class F1 by
under 0.008, below training noise (Tables 4 to 6). (b) Evidence that this holds for crop legends of
10 to 17 CDL classes (Tables 10 to 13). (c) Evidence that interpolation can hurt when anchors are
sparse (Fig. 19). (d) A published statement that zero-padding gaps is the known weak point of CNN
pipelines and the reason masking networks were preferred.

**Matched or mismatched regime.** Matched: train and test are random folds of the same composites
with the same gap statistics; the 50 percent masking experiment masks both train and test.

**Imputation strategies compared at inference.** One strategy (linear interpolation, with one-sided
copy at the series ends) versus masking. No zero fill, no mean or median fill, no cross-condition
evaluation.

## Evaluation Caveats

1. **In-region k-fold, not spatial transfer.** The accuracy figures are five-fold cross-validation
   over samples drawn from one basin in one year. The authors' argument that 10.71 km mean spacing
   makes samples independent is reasonable for pixel-level autocorrelation but does not make the
   protocol a spatially disjoint holdout: all folds share the same tiles, the same 2021 phenology,
   and the same cloud climatology. Their 90 percent OA is not comparable to a cross-tile macro-F1.
2. **Matched missingness.** The study cannot speak to train/test missingness shift. It shows that
   networks trained with gaps tolerate gaps.
3. **Zero-fill is excluded, not measured.** The most common fill value in practice is dismissed by
   citation rather than tested, so the paper gives no number for how much worse zero-padding is.
4. **Labels are visual interpretations** of Landsat composites for the Amur basin and CDL
   (85 to 95 percent accurate) for the crop tiles; label noise is likely non-trivial for
   rainfed vs paddy and for wetland vs grassland, the two lowest-F1 pairs.
5. **Class imbalance is mild and masked by OA.** Classes range from 706 to 1,265 samples, so OA is
   not grossly misleading, but the headline result is phrased in OA; the per-class F1 tables are
   where the actual equivalence is demonstrated, and the two rarest CDL classes (100 and 120
   samples) show inconsistent signs with F1 of 0.15 to 0.48.
6. **Single sensor family, single year.** HLS 30 m, 2021. No cross-year, no cross-sensor, no
   10 m Sentinel-2-only test, and no SAR.
7. **Gap representation vs gap frequency.** The 50 percent random-masking experiment confirms
   indifference to interpolation but reduces accuracy by about one point in both arms, so the
   paper does quantify a cost of missing data; it is just the same cost in both arms.
8. **Compute claim is pipeline-specific.** The 13x speed-up reflects their interpolation
   implementation over 1,940 GB of imagery; it is not a general property of interpolation.
9. **Hyperparameter search is asymmetric.** Bi-LSTM received about 15 days of tuning and
   Transformer about 2 days; model-to-model comparisons in the paper are not on equal footing.
10. **Sample-count inconsistency.** The text reports both 7,735 and 7,755 samples; the discrepancy
    is small and does not affect conclusions but should not be propagated.
