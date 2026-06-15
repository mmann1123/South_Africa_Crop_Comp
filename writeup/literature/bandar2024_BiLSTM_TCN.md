# Crop Classification with Attention Based BI-LSTM and Temporal Convolution Neural Network Combination for Remote Sensing Breizhcrop Time Series Data

**Citation:** Bandar, A. N. A. A., & Coşkunçay, A. (2024). Crop Classification with Attention Based BI-LSTM and Temporal Convolution Neural Network Combination for Remote Sensing Breizhcrop Time Series Data. *Yüzüncü Yıl University Journal of the Institute of Natural & Applied Sciences*, 29(1), 173–188. DOI: 10.53433/yyufbed.1335866 (verified via Crossref).

## Objectives

The paper proposes a hybrid deep network for crop-type classification from
multi-temporal Sentinel-2 time series, evaluated on the public **BreizhCrops**
dataset (Brittany, France). The architecture fuses two parallel branches: a
**bidirectional LSTM with an attention mechanism** (intended to extract salient
*local* temporal features and weight informative time steps) and a **Temporal CNN
(TCN/Temporal-CNN)** branch (intended to extract *general/global* sequence features).
The two feature sets are concatenated and passed to a softmax classifier. The stated
goal is to outperform single-paradigm temporal baselines (Temporal CNN, vanilla
LSTM, Star RNN) by combining local-attention and global-convolution features, and to
handle the strong class imbalance typical of agricultural landscapes.

## Methods

Input is the BreizhCrops Sentinel-2 time series: 13 spectral features per time step
(`B1`–`B12` plus an index), 50 time steps in sequence order, across 13 original crop
classes and four Brittany regions (Frh01–Frh04). Preprocessing: per-feature
normalization, **class balancing by undersampling** the majority classes, and
**removal of six small/ill-defined classes**, leaving 7 classes (Barley, Wheat,
Rapeseed, Corn, Misc., Permanent meadow, Temporary meadow) and 37,348 samples
(Table 1). The authors explicitly reject oversampling/synthetic generation as
unsuitable for time-series data and instead undersample.

Architecture (Figs. 4, 6): Branch 1 = Bidirectional-LSTM attention layer ->
dropout -> LSTM -> flatten. Branch 2 = Conv1D -> ReLU -> dropout -> Conv1D -> ReLU
-> MaxPooling1D -> flatten. Concatenate -> softmax. Baselines: Temporal CNN
(Lea et al. 2016), Star RNN (Tran et al. 2023), and Vanilla LSTM. Metrics: precision,
recall, F1-score, and Cohen's Kappa, plus per-class confusion matrices. Training on
CPU (i7-8550, 12 GB RAM); code on GitHub. Training duration and parameter counts are
reported (Table 5).

**Evaluation protocol.** Validation is a **single random 80/20 train/test split
pooled across all four Brittany regions** ("using 20% of the data... the validation
of our method has been done"). This is **pooled random hold-out within one
agro-climatic zone**, not a spatially disjoint holdout. The four regions Frh01–Frh04
are *mixed together* before splitting, so samples from the same region — and
plausibly spatially adjacent/autocorrelated fields — appear in both train and test.
On our generalization spectrum (pooled-pixel k-fold -> field-wise k-fold ->
spatial holdout -> cross-year/cross-sensor), this sits at the **pooled random
hold-out** rung — the weak end. The split is by sample, with **no stated field-wise
(FID) disjointness** and **no spatial separation of train and test**. The reported
0.82 precision/F1 is therefore an *in-region* number; by our manuscript's thesis it
is exactly the kind of in-region validation that systematically misranks dense
temporal nets relative to how they would transfer to a disjoint tile. The paper does
**not** test spatial transfer across the four regions (e.g., train Frh01–03, test
Frh04), even though the dataset's regional structure makes such a test trivial — a
notable silence.

## Key Findings

- The proposed **BI-LSTM-attention + Temporal-CNN hybrid** achieved the best scores
  among the four models: precision 0.82, recall 0.82, F1 0.82, Kappa 0.76 (Table 3),
  versus Temporal CNN (0.80/0.80/0.80/0.73), Vanilla LSTM (0.73/0.71/0.79/0.73), and
  Star RNN (0.71/0.72/0.71/0.69).
- Per-class confusion matrices (Fig. 6) show the hybrid improves the hardest
  meadow classes; vanilla LSTM and Temporal CNN are competitive, while Star RNN
  trails. Vanilla LSTM "alone does not work properly" but improves with attention.
- The authors report **macro-level metrics on a balanced (undersampled) 7-class
  problem**; the headline F1 of 0.82 is on this reduced, rebalanced label set, not
  the full 13-class problem.
- Compute: the hybrid has fewer parameters (255,965) than Star RNN (567,103) and
  trains in ~49 s, comparable to other models; the authors frame this as a
  favorable accuracy/parameter tradeoff.
- A stated limitation: the model requires every sample to have the *same fixed-length
  time series* and substantial sequence length to reach high accuracy.

## Relevance to Our Crop-Classification Study

This is a close methodological sibling of several of our deep-learning baselines —
it combines exactly the dense temporal paradigms (BI-LSTM, Temporal-CNN, attention)
that our manuscript shows overfit the training region and collapse out-of-sample.
Its value to us is as a **contrast case and a cautionary citation**:

- **Same model family, weaker protocol.** It uses CNN-BiLSTM/TempCNN-style
  architectures and reports them as winners — but only under pooled in-region
  hold-out. This is precisely the validation design our central finding indicts. We
  can cite it as representative of the literature norm (dense temporal net + in-region
  random split = high reported F1) against which our spatial-holdout reordering is
  the corrective.
- **Imbalance handling by undersampling + class deletion.** The authors discard six
  minority classes and undersample the majority. Our study instead keeps the
  lucerne/medics-dominated full label set and reports macro-F1/Kappa/per-class F1.
  Their approach inflates apparent balance and sidesteps the minority-recall problem
  rather than solving it — a contrast worth drawing for our imbalance discussion.
- **Attention is not sparsemax.** Their soft attention reweights time steps but does
  not impose hard sparsity. This contrasts with TabNet's attentive *sparsemax* masks,
  which our results show transfer better. Useful for the inductive-bias argument that
  *sparse* selection (not merely *soft* attention) is what aids transfer.
- **Optical-only Sentinel-2, raw temporal sequences.** Like us, optical-only, no SAR;
  but they feed raw `B1`–`B12` sequences to the network rather than engineered
  time-series features. This is the raw-sequence-vs-engineered-feature axis our paper
  examines.
- **Field granularity ambiguous.** Samples appear to be field-tracked but the model
  classifies sequences without an explicit field-level aggregation step like ours;
  field-level variance reduction is not exploited.

## Evaluation Caveats

- **Cite the protocol, not the 0.82.** The 0.82 macro-F1 is on a balanced,
  undersampled, 7-of-13-class problem under a pooled random in-region 80/20 split —
  *not* comparable to our spatially disjoint macro-F1. Presenting it as a transfer
  result would be exactly the misranking our paper warns against.
- **Spatial leakage not controlled.** Four regions are pooled before splitting; no
  field-wise (FID) disjointness or spatial train/test separation is stated, so
  spatially autocorrelated fields can straddle the split and inflate the numbers.
- **Class imbalance engineered away.** Six classes deleted and majorities
  undersampled means the metrics describe an artificially balanced subset; the
  reported F1 does not reflect minority-crop recall on the native distribution, the
  regime our lucerne-dominated problem lives in.
- **Silences.** No cross-region spatial transfer (despite the obvious Frh01–04
  holdout), no cross-year robustness, no full-13-class results, and per-class numbers
  only via confusion-matrix heatmaps rather than reported per-class F1. Compute is
  benchmarked, which is a plus.
- **Fixed-length-sequence dependence.** Acknowledged requirement for long,
  equal-length series limits applicability where months are missing — relevant to our
  exclusion of months `05` and `06`.
