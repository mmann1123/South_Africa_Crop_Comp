# Self-attention for raw optical Satellite Time Series Classification

**Citation:** Marc Rußwurm and Marco Körner (2020). *ISPRS Journal of Photogrammetry and Remote
Sensing* 169, 421–435. DOI: `10.1016/j.isprsjprs.2020.06.006` (from Crossref title match; the PDF in
the folder is arXiv:1910.10536v3, 23 Oct 2020, whose header states "Accepted at Journal of
Photogrammetry and Remote Sensing Volume 169, November 2020, Pages 421-435"; Crossref returns the
identical title, ISPRS JPRS, issued 2020-11). Code: `github.com/marccoru/crop-type-mapping`. Dataset
is the "BavarianCrops" parcel time-series set reused by Obadić et al. (arXiv:2210.13167).

## Objectives

The paper asks whether end-to-end deep models can replace the expert-designed preprocessing
(atmospheric correction, cloud masking, temporal selection of clear observations) that remote-sensing
pipelines usually apply before classification, and which of the three dominant sequence mechanisms,
1D convolution, recurrence, and self-attention, copes best when that preprocessing is absent. The
authors motivate this with the low exploitation ratio of Sentinel-2 (7.6 % of 2018 products ever
downloaded) and argue that "lightweight problem-agnostic models that do not require region- or
problem-specific expert knowledge" are needed to use the daily data volume.

Two stated contributions: (i) a large-scale hyperparameter-tuned comparison of LSTM-RNN, Transformer,
multi-scale 1D ResNet (`MS-ResNet`), `TempCNN`, `DuPLO`, and a Random Forest baseline on the same
field parcels represented once by raw top-of-atmosphere (L1C) time series and once by an
industry-standard preprocessed product; (ii) a quantitative (gradient backpropagation) and qualitative
(attention-score) analysis of how self-attention and recurrence treat cloudy observations.

## Methods

**Study area and labels.** Three spatially separate regions in Bavaria, Germany (Hollfeld in Upper
Franconia, Krumbach in Swabia, and part of the Bavarian Forest), roughly 100 km apart, with similar
climate but different elevation and soils. Labels come from the Bavarian State Ministry of Food,
Agriculture and Forestry (farmer-declared CAP subsidy data), aggregated by the ministry and GAF AG into
a 23-class land-use nomenclature (several grassland sub-types, cereals, maize, rapeseed, etc.) and a
coarser 12-class land-cover nomenclature. The class distribution is long-tailed: of 269 original
categories, the top 15 cover 90 % of parcels.

**Two representations of the same parcels, year 2018.**
- *Raw dataset*: Sentinel-2 L1C top-of-atmosphere reflectance pulled from Google Earth Engine, all 13
  bands, every acquisition (about 70 observations per year on swath centres, up to 140 in overlap
  zones), pixel values mean-aggregated inside the parcel polygon. No cloud masking; cloud appears as
  positive reflectance spikes across all bands. Sequence lengths vary.
- *Preprocessed dataset*: the same parcels passed through GAF AG's operational engine (atmospheric
  correction, cloud classification and masking, restriction to the vegetative period, temporal
  interpolation), yielding 23 time steps of 10 bands with repeated values where gaps were filled.

**Models.** All operate on a parcel-level sequence `X in R^{T x D}`.
- `LSTM-RNN`: stacked bidirectional LSTM, concatenated final forward/backward states, softmax head.
  Tuned hidden size `2^4..2^8`, layers 1–7, dropout.
- `Transformer`: encoder-only Vaswani architecture with positional encoding, global max-pool over
  time, dense softmax head. Tuned hidden size, layers 1–8, heads 1–8, warm-up steps.
- `MS-ResNet`: three parallel residual 1D-conv streams with kernels 3/5/7, fixed `T = 512` input
  obtained by nearest-neighbour interpolation of the sequence; only kernel count tuned.
- `TempCNN` (Pelletier et al. 2019): three 1D conv + BN + ReLU + dropout layers, flatten, dense.
  Tuned kernel count, kernel size 3/5/7, dropout.
- `DuPLO` (Interdonato et al. 2019): two-stream CNN + GRU with simplified soft-attention, used with
  the authors' fixed hyperparameters, auxiliary losses, treating the parcel series as a 1-pixel image.
- `RF`: flattened vector of all time steps times (13 bands + NDVI, NDWI, BI, IRECI, EVI), i.e. 1260
  features on raw data and 345 on preprocessed data. Because RF did not train acceptably on the full
  imbalanced set, it was trained and tested on a more uniform subset of at most 500 series per class,
  which the authors say should advantage RF on class-wise accuracy.

**Training and tuning.** Adam, log-uniform learning rate and weight decay, Transformer warm-up
schedule, early stopping on validation loss. Hyperparameters were searched with Hyperopt (34 random
draws, then kernel-density-guided sampling) combined with asynchronous successive halving over 60
epochs in four brackets, on Ray Tune on a DGX-1: 300 configurations per model per dataset, 2400 model
evaluations total, tuning on at most 500 samples per class. Final numbers are the mean and standard
deviation over the three best configurations, each with its own seed.

**Evaluation protocol.** The three regions were tiled into 4500 m x 4500 m blocks with a 500 m
margin between blocks; blocks were randomly assigned to train/validation/test in a 4:1:1 ratio (94,
21, 29 blocks). All three regions contribute blocks to all three partitions. This is a *block-wise
spatial cross-validation within the training regions*: it removes the within-field and
near-neighbour autocorrelation leak (the authors cite their earlier observation of implicit
overfitting without it), but test blocks remain embedded in, and climatically identical to, the
training landscape. It is not a disjoint-region or disjoint-tile holdout, and no cross-year test is
run. On our manuscript's spectrum it sits one notch above field-wise k-fold (blocks are 4.5 km, so
test parcels are at least 500 m from any training parcel) and well below a spatially disjoint
holdout tile. Metrics: overall accuracy, Cohen's kappa, and class-mean F1, reported separately for
raw/preprocessed and for 23/12 classes (Table 1).

**Feature-importance analysis (Sec. 8.3).** For a trained model they backpropagate the gradient of
the top predicted score `y*` to the input tensor, `dy*/dX`, per time step and band, with no labels
required, and plot it against the raw reflectance series for a summer barley and a maize parcel.
**Attention analysis (Sec. 8.4–8.5).** First-layer attention matrices of the Transformer are drawn as
heat maps and bipartite graphs, and the column-averaged attention is compared to DuPLO's soft-attention
scores. **Embedding analysis (Sec. 8.6).** t-SNE, UMAP, and PCA of hidden features after each
Transformer layer.

## Key Findings

**Preprocessing helps every model; architecture matters little once data are clean.** On the
preprocessed product all six models are within noise of each other. Kappa, 23 classes: RF 0.76,
LSTM-RNN 0.78 ± 0.01, Transformer 0.79 ± 0.03, DuPLO 0.81, MS-ResNet 0.76 ± 0.03, TempCNN 0.80 ± 0.00.
Kappa, 12 classes: RF 0.86, LSTM 0.87, Transformer 0.87, DuPLO 0.89, MS-ResNet 0.85, TempCNN 0.88.
Overall accuracy is 0.83–0.86 (23 classes) and 0.91–0.92 (12 classes) for everything including RF.
The authors conclude that "the choice of model architecture was less critical when extensive data
preprocessing was utilized."

**On raw data the ranking changes, and the change is the paper's main empirical result.** With
clouds left in:
- RF collapses: kappa 0.53 (23-class) and 0.62 (12-class), i.e. 0.23–0.24 below its preprocessed
  score; class-mean F1 falls to 0.18 and 0.34. Flattened raw reflectances plus indices are not a
  usable feature set for trees without cloud handling.
- TempCNN falls to the bottom of the deep models: kappa 0.67 ± 0.02 (23-class) and 0.71 ± 0.07
  (12-class), the latter 0.17 below its preprocessed value and with by far the largest variance in
  the table; class-mean F1 0.36 and 0.41.
- MS-ResNet: 0.69 and 0.78 kappa; DuPLO 0.68 and 0.78.
- LSTM-RNN and Transformer hold up best: kappa 0.71 ± 0.01 / 0.83 ± 0.01 and 0.71 ± 0.03 / 0.82 ± 0.02;
  class-mean F1 0.43 / 0.63 and 0.45 / 0.64. In the 12-class setting the LSTM's class-mean F1 on raw
  data (0.63) is *higher* than on preprocessed data (0.60).
The raw-vs-preprocessed gap is therefore smallest for recurrence and self-attention (about 0.04–0.08
kappa), larger for convolutions (0.07–0.17), and largest for RF (0.23–0.24). In the 23-class task the
gap between deep models is only 1–2 % accuracy and 0.01 kappa ("not significant considering their
reported variances"); the 12-class task separates them by 0.03–0.07 kappa and 5–11 % accuracy.

**Gradient analysis: recurrence and attention suppress cloudy dates; convolutions do not fully.**
For the LSTM-RNN and Transformer, `dy*/dX` is non-zero at only a handful of time steps and is zero at
the reflectance spikes caused by clouds, "learned purely from data." Both can "transition from a time
observation with high gradients to zero gradients without any time delay." MS-ResNet and especially
the shallow TempCNN keep non-zero gradients on cloudy observations. The convolutional models seem to
need at least two or three consecutive cloudy observations to suppress them: MS-ResNet zeroed its
gradient at `t = 30` (three consecutive cloudy acquisitions) but not at `t = 34` (a single cloudy
acquisition). MS-ResNet also attends to winter observations that should carry no crop information. The
authors attribute this to the local receptive field of convolution versus the gating of LSTM cells and
the per-observation weighting of attention.

**Attention scores do not land on cloud peaks.** Average first-layer attention concentrates on a few
clear-sky dates with distinct heads focusing on early-middle versus early-and-late season; the DuPLO
soft-attention, computed after a monodirectional GRU, is smeared over neighbouring dates and biased to
the late season. Deeper Transformer layers produce increasingly separable embeddings (maize and
rapeseed form distinct manifolds).

**What is not measured.** No transfer to a region, tile, or year outside the three training
landscapes; no experiment in which the training data are clean and the test data are corrupted (the
cloud robustness is learned because clouds are present in training); no per-class results beyond the
class-mean F1; no reporting of how much of the raw-data advantage comes from the extra observations
(70–140 versus 23) rather than from the lack of preprocessing; RF trained on a different, balanced
subset, so its row is not strictly comparable.

### Numbers to cite (Table 1, mean of three best configurations)

| Metric | Data | RF | LSTM-RNN | Transformer | DuPLO | MS-ResNet | TempCNN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kappa, 23 cl. | pre | 0.76 | 0.78 | 0.79 | 0.81 | 0.76 | 0.80 |
| Kappa, 23 cl. | raw | 0.53 | 0.71 | 0.71 | 0.68 | 0.69 | 0.67 |
| Kappa, 12 cl. | pre | 0.86 | 0.87 | 0.87 | 0.89 | 0.85 | 0.88 |
| Kappa, 12 cl. | raw | 0.62 | 0.83 | 0.82 | 0.78 | 0.78 | 0.71 |
| Mean F1, 12 cl. | pre | 0.55 | 0.60 | 0.66 | 0.66 | 0.64 | 0.60 |
| Mean F1, 12 cl. | raw | 0.34 | 0.63 | 0.64 | 0.51 | 0.55 | 0.41 |

Raw-minus-preprocessed kappa, 12 classes: RF -0.24, TempCNN -0.17, MS-ResNet -0.07, DuPLO -0.11,
Transformer -0.05, LSTM-RNN -0.04. The paper's own summary sentences, usable verbatim: "data
preprocessing still increased the overall classification performance for all models while the choice
of model was less crucial"; "self-attention and recurrent neural networks, by their architecture,
outperformed convolutional neural networks on raw satellite time series"; and, from the gradient
analysis, the LSTM-RNN and Transformer "were able to suppress these cloudy observations even if they
appeared in a single time instance," whereas the convolutional models "require at least two
subsequent observations to reduce the influence of the particular time instance."

## Relevance to Our Crop-Classification Study

Our manuscript's revised findings are: (1) the holdout tile (`34S_20E_259N`) is missing September for
41 % of pixels while the training tiles never are; the old pipeline zero-filled that month before
standardization for every raw-sequence deep model, a train/test missingness shift that cost those
models 0.08–0.14 holdout macro-F1 and manufactured an apparent in-region/out-of-region ranking
inversion. With September filled from training-tile medians, dense temporal pixel models lead on the
holdout (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63), TabNet is 0.62,
tree/linear/MiniRocket are 0.59–0.61, and there is no inversion. (2) A sparsemax channel gate
(L-TAE-S, and the same gate on CNN-BiLSTM and TempCNN with softmax controls) gives no transfer gain on
complete inputs and no consistent low-data gain, but about +0.06 robustness to the zero-filled month;
TabNet itself is not robust to zero fill (0.61 to 0.51). (3) On the corrected holdout every model is
near 0.55 macro-F1 at 10 % of training fields (XGBoost 0.51); deep models rise to about 0.65 at 100 %
and logistic regression plateaus at 0.61.

**This paper is the closest published analogue to our imputation artifact, with one decisive
difference.** Rußwurm and Körner show that a single anomalous observation (a cloud spike) inside an
otherwise informative sequence is handled very differently by different inductive biases: recurrence
and attention learn to zero its influence, a shallow TempCNN does not, and RF on flattened raw inputs
collapses. Our zero-filled September is exactly such a single-slot anomaly in a 10-step monthly
sequence, and the ordering of damage we observed before the fix is broadly consistent with theirs. The
difference is that their models *saw clouds during training* and so had the opportunity to learn
suppression; ours never saw a zero-filled month in training. Their paper therefore explains why
suppression is learnable, and our result shows what happens when the corruption is absent at training
time: no architecture, however good at gating, can learn to ignore a value it has never encountered.
That is the distinction between noise robustness (their setting) and missingness shift (ours), and it
is the reason the fix had to be at the data level (median fill, or in future a mask or temporal
dropout at training) rather than at the model level.

**The raw-vs-clean ranking change is a template for how we should describe our own ranking change.**
Their Table 1 is a controlled demonstration that the same models on the same parcels reorder when the
input distribution changes: TempCNN goes from joint-best (kappa 0.80/0.88 clean) to worst deep model
(0.67/0.71 raw), and RF from competitive to unusable. They do not interpret this as "TempCNN overfits"
or "RF is fragile"; they interpret it as a property of the input, and locate the mechanism with a
gradient analysis. Our revised manuscript should make the same move: the old inversion was a property
of the zero-filled input, not of dense models, and we can show it the same way. A `dy/dx`
backpropagation to the September slot (their Sec. 8.3 recipe, a few lines of autograd, no labels
needed) on the old zero-filled holdout versus the median-filled holdout would give a direct,
model-agnostic picture of how much of each model's prediction was driven by the artificial zeros. This
would be a stronger figure than the before/after F1 table alone, and it is cheap.

**On the sparsemax channel gate (finding 2).** Their attention analysis shows that softmax attention
over *time* already becomes effectively sparse on clear-sky dates without any sparsity prior, which is
one reason to expect that forcing sparsity (sparsemax) adds little on complete inputs, exactly what we
found. The +0.06 robustness of the gated models to zero fill is also interpretable in their terms: a
gate that can drive a channel's weight to exactly zero is a hard version of the suppression they
observe in LSTM/Transformer gradients, so it is plausible that a gated model partially suppresses an
all-zero slot even without having been trained on one. But their results also warn against reading
this as a general transfer property: their LSTM and Transformer suppressed clouds because clouds were
in the training distribution, and our gate's benefit appears only under the artifact we have now
removed. We should present it as a robustness-to-corruption result, not a transfer result.

**TabNet's collapse under zero fill (0.61 to 0.51) is consistent with their RF row.** Their RF on raw
flattened reflectances lost 0.23–0.24 kappa because trees have no mechanism to recognise that a
feature value is "not a measurement"; TabNet's sequential attentive feature selection operates on the
same flattened representation and so inherits the same blindness. Sparsity in feature *selection*
(trees, TabNet) is not the same thing as the ability to *gate a corrupted observation* (LSTM cells,
attention over time, our channel gate). The old manuscript's inductive-bias argument conflated these
two kinds of sparsity; this paper separates them cleanly.

**Dense models leading on the corrected holdout is unsurprising in their frame.** With clean,
consistently represented inputs they find all deep models within noise of each other and
only slightly above a well-fed RF; our corrected holdout shows the same compression (0.59–0.66 across
all families) with the dense temporal models on top. Their data are parcel-mean aggregates; ours are
pixels aggregated to fields by majority vote, which may explain why our field-level aggregation also
matters (the paper reports no pixel-level results).

**Learning curves (finding 3).** No label-fraction experiment is reported here, so this paper does
not bear on finding 3 beyond the observation that RF needed a class-balanced subset to train at all,
a hint that tree models are sensitive to the composition of small training sets.

**Caution on citing this paper for "attention or recurrence transfers better."** Its evaluation never
leaves the three Bavarian training landscapes; its robustness claim is to in-distribution noise, not
to spatial or temporal shift. It should be cited for (a) the raw-vs-clean ranking change, (b) the
mechanism (gating/attention zeroing single anomalous time steps while shallow convolutions cannot),
and (c) the gradient-attribution recipe, not for cross-region transfer.

## Evaluation Caveats

- **Spatial protocol.** Block-wise (4.5 km blocks, 500 m margins) random assignment within three
  regions that all appear in train, validation, and test. This controls field-level and near-neighbour
  leakage but is not a disjoint-region holdout; climate, soil, and crop calendar are shared between
  train and test. Treat all numbers as in-region spatial-CV, not out-of-region transfer.
- **No cross-year or cross-sensor test.** Single year (2018), single sensor (Sentinel-2).
- **Cloud robustness is learned in-distribution.** Clouds are present in the raw training data. The
  paper does not test a clean-train/corrupted-test or corrupted-train/clean-test configuration, so it
  says nothing directly about missingness *shift*.
- **Class imbalance.** Long-tailed labels; overall accuracy is dominated by frequent classes (the
  authors say so). Class-mean F1 is reported but is low everywhere (0.18–0.66), and no per-class
  tables are given, so minority-class behaviour under raw data is invisible beyond the mean.
- **RF is not on equal footing.** Trained and tested on a balanced subset of at most 500 series per
  class because full-data training "did not obtain satisfactory results," and given hand-crafted
  indices the deep models were not given. Its collapse on raw data is still informative but its row
  cannot be read as a like-for-like baseline.
- **Raw and preprocessed inputs differ in more than cloudiness.** Raw sequences have 70–140 steps and
  13 bands; preprocessed have 23 steps and 10 bands, so sequence length and temporal density are
  confounded with the presence of clouds.
- **DuPLO** could not be tuned (fixed authors' configuration); MS-ResNet requires interpolation to a
  fixed `T = 512`, which itself is a form of preprocessing.
- **Variance.** Reported standard deviations come from three hyperparameter configurations with
  different seeds, not from repeated runs of one configuration, so they mix tuning variance with
  initialization variance.
- **Metrics.** Kappa, OA, and class-mean F1 are all reported; this is better than most of the corpus,
  and kappa is the metric the authors tune on.
- **Computational cost** of training (2400 tuned runs on a DGX-1) is noted but inference cost is not
  compared across models.
