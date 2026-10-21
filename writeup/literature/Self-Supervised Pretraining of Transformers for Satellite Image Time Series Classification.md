# Self-Supervised Pretraining of Transformers for Satellite Image Time Series Classification

**Citation:** Yuan Yuan and Lei Lin (2021). *IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing* 14, 474–487. DOI: `10.1109/JSTARS.2020.3036602` (verified: extracted
from the PDF's first-page footer; Crossref returns the identical title, JSTARS, issued 2021).
Published online 9 Nov 2020; CC BY 4.0. Model name: SITS-BERT. Code and pretrained weights:
`github.com/linlei1214/SITS-BERT`.

## Objectives

The paper proposes the first self-supervised pretraining scheme for pixel-level satellite image time
series (SITS) classification, to counter the overfitting of deep models when labeled samples are
scarce. Three stated contributions: (i) a BERT-style pretext task in which randomly chosen
observations in an annual Sentinel-2 pixel series are corrupted with noise and the transformer must
reconstruct them from the surrounding clear observations and their acquisition dates; (ii) a
transformer-encoder architecture (SITS-BERT) as an alternative to 1D-CNN and LSTM for SITS; (iii)
experiments on three datasets, one inside the pretraining area and two outside it, measuring the gain
from pretraining, the gain over classical and deep baselines, and the dependence on the number of
labeled samples per class.

## Methods

**Architecture (Sec. IV).** Input is an annual series of tuples `(O_i, t_i)` with `O_i` a
10-band bottom-of-atmosphere reflectance vector and `t_i` the day of year. The observation embedding
concatenates a linear projection of `O_i` with a sinusoidal positional encoding of `t_i` (not added as
in BERT, to keep observation and time distinguishable; concatenation also converged faster). Embedding
size `d_m = 256`; standard transformer encoder blocks with 8 heads of dimension 32, feed-forward width
1024, residual connections and layer normalization. Using DOY rather than sequence index is meant to
make the model "trainable and transferable across different years" and to handle irregular sampling
"so no smoothing, gap filling, or compositing method is needed." Sequence-level representation by a
`[CLS]` token, average pooling, or max pooling (three variants); max pooling is adopted as default.

**Pretext task (Sec. V-A).** 15 % of the observations in each series are selected at random. Each
selected observation has a 50 % chance of having a positive noise added to all bands (simulating
reflectance spikes from clouds and snow) and a 50 % chance of a positive noise subtracted (simulating
shadows); noise is uniform in `[0, 0.5]`; the positional encoding is left intact. The model predicts
the original values of the corrupted observations through a linear head, with mean-squared error over
the corrupted positions only. No `[MASK]` token is introduced, which the authors note avoids BERT's
pretrain/fine-tune discrepancy. Pretraining: 100 epochs, batch 256, Adam at 1e-4 with 30-epoch warm-up
and exponential decay, dropout 0.1.

**Fine-tuning (Sec. V-B).** A softmax output layer is added and all parameters are trained end to end
with cross-entropy, 100 epochs, Adam at 2e-4, batch 128.

**Pretraining data.** Three Sentinel-2 tiles in California's Central Valley (`T10SEJ`, `T10SFH`,
`T11SKA`), all images with under 10 % cloud from January 2018 to December 2019 (219 images), pixels
sampled every 10 rows and columns, sequences shorter than 10 discarded, about 6.5 million unlabeled
series. California was chosen for biome and crop diversity and a Mediterranean climate with dry
summers, so that "the spectral trajectory of major crops throughout the entire growth cycle can be
characterized."

**Fine-tuning datasets (Table I).**
1. *California crop*, tile `T11SKA`, 45 images in 2019, 13 classes (corn, cotton, winter wheat,
   alfalfa, tomatoes, grapes, citrus, almonds, walnuts, pistachios, developed, evergreen forest,
   grass/pasture). Same area as pretraining but the fine-tuning pixels were not used for pretraining.
2. *Missouri/Arkansas crop*, tile `15SYA`, 26 images in 2019, 10 classes (corn, cotton, rice,
   soybeans, fallow, water, developed, forest, grass/pasture, woody wetlands). Outside the pretraining
   area.
3. *Beijing land cover*, tile `50TMK`, 20 images in 2017, 5 classes (cropland, forest, grassland,
   water, impervious) from FROM_GLC10. Far from the pretraining area in climate and vegetation.
Labels for 1 and 2 come from the USDA Cropland Data Layer (30 m, upsampled to 10 m) filtered to pixels
whose 7x7 neighbourhood is homogeneous with CDL confidence above 80 %. All Sentinel-2 L1C scenes were
processed to L2A with Sen2Cor; the scene classification map was used to discard cloud, cirrus, shadow,
and snow observations, and each pixel series must contain at least three clear observations.

**Baselines.** SVM (RBF, grid over `C` and `gamma`), RF (100–500 trees), three-layer CNN-1D (128
filters, kernel 5, dropout 0.5), three-layer LSTM (256 units), three-layer Bi-LSTM (128 units). SVM and
RF receive the series as a flat vector *zero-padded to the maximum length* (missing observations
filled with zeros, consistently in training and test). The deep baselines share SITS-BERT's observation
embedding. Baselines are trained from scratch for 300 epochs.

**Evaluation protocol.** For each dataset, 100 samples per class are drawn at random for training and
another 100 per class for validation; *all remaining labeled pixels in the same tile* form the test
set. The authors explicitly reject a proportional train/validation/test split: "the data distributions
of training samples and test samples may be different," with the aim of "verifying the effectiveness
of the proposed pretraining scheme under small-sample conditions." Pixels are sampled on a regular
10-pixel grid with no field or parcel information, so training and test pixels can come from the same
field. On our manuscript's spectrum this is **pooled random pixel sampling within a single scene**,
below field-wise k-fold; it does not measure spatial transfer of the *classifier*. The one transfer
element is that for datasets 2 and 3 the *pretrained representation* comes from a different region,
but the labels used for fine-tuning and testing are always from the target tile. Metrics: overall
accuracy, kappa, and average (per-class mean producer's) accuracy, reported in Tables II–IV, which are
embedded as images in the PDF; the numbers quoted below are those stated in the text.

**Label-fraction experiment (Sec. VII-E).** Pretrained CNN-1D, Bi-LSTM, and SITS-BERT are fine-tuned
with 50 to 500 labeled samples per class and compared with RF (300 trees) and with the non-pretrained
SITS-BERT at each size (Fig. 12).

## Key Findings

**Pretraining helps every variant and every dataset.** Averaged over the `[CLS]`, average-pool, and
max-pool variants, pretraining raises overall accuracy by 2.97, 1.88, and 2.58 points and kappa by
0.034, 0.023, and 0.045 on the California, Missouri, and Beijing datasets respectively, and average
accuracy by 0.030, 0.029, and 0.068. The authors read the gains on datasets 2 and 3 as evidence of
"positive transfer" of the representation to regions outside the pretraining area; the largest average
accuracy gain is on the most distant dataset (Beijing). Max pooling gives the best OA and kappa;
`[CLS]` gives higher average accuracy on datasets 2 and 3.

**Against baselines (Table III).** Relative to RF, pretrained SITS-BERT improves OA by 5.27, 2.38, and
3.35 points, kappa by 0.061, 0.029, and 0.053, and average accuracy by 0.056, 0.018, and 0.026 on the
three datasets. The ordering of the other models is the finding most relevant to us: "both CNN and RNN
networks yield comparable or even worse results than traditional methods (SVM and RF), while the
non-pre-trained SITS-BERT performs slightly better than traditional methods on the latter two
datasets. Apart from the proposed model, RF performs best overall, ranking first on the latter two
datasets." The authors attribute this to overfitting at 100 samples per class and note that the
transformer and CNN-1D "seem to be more robust to overfitting compared to LSTM-based networks." The
non-pretrained SITS-BERT is "much worse" than RF on the Beijing dataset. Pretraining also reduces
confusion for heterogeneous classes (pistachios vs evergreen forest: +20 points producer's accuracy
over the next best model; cropland in Beijing: +9 points).

**Pretraining transfers to other architectures (Table IV).** Applying the same pretext task to
CNN-1D and Bi-LSTM yields average OA gains of 2.58 and 5.42 points; the gain is larger for the
recurrent model, which the authors link to LSTM's and the transformer's ability to model long-range
dependencies versus a CNN's local receptive field.

**Label-fraction results (Fig. 12).** With 50 to 500 labeled samples per class, "all the pretrained
deep learning models significantly outperform RF and the non-pre-trained model, among which SITS-BERT
surpasses all other competitors in all datasets." SITS-BERT fine-tuned on only 50 samples per class is
comparable to RF trained on ten times as many on datasets 1 and 2: the best RF OA at 500 per class is
93.33 % (California) and 97.58 % (Missouri), while the worst SITS-BERT OA (at 50 per class) is 93.26 %
and 98.24 %. The non-pretrained SITS-BERT is slightly above RF on datasets 1 and 2 but far below it on
dataset 3, and "it is not guaranteed that the accuracy of non-pre-trained SITS-BERT will improve as the
number of training samples increases," which the authors attribute to unstable training from random
initialization.

**Compute (Table V).** CNN-1D is about 19 % faster than SITS-BERT at inference; Bi-LSTM about 5 %
faster; SITS-BERT parallelizes across GPUs.

**What is not measured.** Spatial transfer of a fine-tuned classifier to a tile with no labels;
cross-year application of a fine-tuned model (DOY encoding is argued to permit it but it is not
tested); field-level or parcel-level accuracy; macro-F1 (average accuracy is a recall-only balanced
metric); the effect of corrupted or missing observations at *test* time on a model that was or was not
pretrained with the noise pretext; the role of the zero padding in SVM/RF.

### Numbers to cite (differences stated in the text; absolute table values are images in the PDF)

| Comparison | California (13 cl.) | Missouri (10 cl.) | Beijing (5 cl.) |
| --- | --- | --- | --- |
| Pretrained minus non-pretrained SITS-BERT, OA (points) | +2.97 | +1.88 | +2.58 |
| same, kappa | +0.034 | +0.023 | +0.045 |
| same, average accuracy | +0.030 | +0.029 | +0.068 |
| Pretrained SITS-BERT minus RF, OA (points) | +5.27 | +2.38 | +3.35 |
| same, kappa | +0.061 | +0.029 | +0.053 |
| same, average accuracy | +0.056 | +0.018 | +0.026 |
| Pretraining gain for CNN-1D / Bi-LSTM, OA (mean over datasets) | +2.58 / +5.42 | | |

Label-fraction anchors (Fig. 12, OA): RF at 500 samples per class, 93.33 % (California) and 97.58 %
(Missouri); SITS-BERT at 50 samples per class, 93.26 % and 98.24 %. Protocol: 100 training and 100
validation pixels per class drawn at random from the tile, all remaining labeled pixels used for
test, no field or spatial blocking.

Citable sentences: "both CNN and RNN networks yield comparable or even worse results than traditional
methods (SVM and RF)"; "apart from the proposed model, RF performs best overall"; "for small datasets,
sophisticated deep learning models have no substantial advantages over traditional machine learning
methods in SITS classification tasks"; "all the pretrained deep learning models significantly
outperform RF and the non-pre-trained model."

## Relevance to Our Crop-Classification Study

Our manuscript's revised findings are: (1) the holdout tile is missing September for 41 % of pixels,
never missing in training; zero-filling that month before standardization in every raw-sequence deep
model imposed a train/test missingness shift that cost those models 0.08–0.14 holdout macro-F1 and
produced an apparent ranking inversion; with September filled from training-tile medians, dense deep
temporal pixel models lead (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet
0.63), TabNet 0.62, tree/linear/MiniRocket 0.59–0.61, no inversion. (2) A sparsemax channel gate gives
no transfer gain on complete inputs and no consistent low-data gain, but about +0.06 robustness to the
zero-filled month; TabNet is not robust to zero fill (0.61 to 0.51). (3) At 10 % of training fields
every model is near 0.55 macro-F1 (XGBoost 0.51); deep models reach about 0.65 at 100 %, logistic
regression plateaus at 0.61.

**The label-fraction result is the closest published analogue to our learning curves, and it agrees
with them in shape but not in the tree baseline.** At their smallest size (50–100 samples per class)
the from-scratch CNN-1D and LSTMs are at or below RF, and the from-scratch transformer is only
slightly above it; this is the same compression we see at 10 % of training fields, where every family
sits near 0.55 macro-F1. Where the two studies differ is the bottom of the curve: their RF is a strong
small-sample baseline (first or second on every dataset), whereas our XGBoost is the *worst* model at
10 % (0.51). Two differences in protocol plausibly explain this. Their test pixels are drawn from the
same scene and the same fields as the training pixels, so RF's tendency to memorize local spectral
detail is rewarded rather than penalized; ours are evaluated on a disjoint tile where that detail does
not transfer. And their RF is fed per-date reflectances, while ours is fed xr_fresh summary statistics,
which at 10 % of fields may be too few samples to estimate the gradient-boosting split structure
over hundreds of engineered features. Their curves cannot, therefore, be cited as showing that trees
are the right low-data choice *for transfer*; they show it for in-scene pixel sampling only. The
useful citable fact is the convergence of all from-scratch models at small sample sizes, which matches
finding 3, and the claim that pretraining is what separates the deep models from RF at 50 per class.

**Pretraining is the lever we have not pulled.** Their deep models beat RF at every label budget only
*after* self-supervised pretraining on 6.5 million unlabeled series; without it, the transformer is
marginal and the CNN and LSTMs lose. Our deep models are all trained from scratch on two tiles, and
their advantage on the corrected holdout (0.63–0.66 versus 0.59–0.61) emerges only at 100 % of fields.
SITS-BERT's recipe is directly applicable to us: the holdout tile and the two training tiles together
contain far more unlabeled pixels than labeled fields, the pretext task needs no labels, and the
authors show that representations pretrained on one region improve fine-tuning on a distant one. A
pretrained Transformer or L-TAE fine-tuned on 10 % of fields is the obvious experiment to test whether
the flat low-data region of our learning curves is a label-scarcity ceiling that pretraining lifts, as
in their Fig. 12, or a transfer ceiling that it does not. Presto (already briefed in this folder) is
the modern version of the same idea.

**The pretext task is a missingness-robustness training signal, which bears on finding 1.** Their
corruption model injects an anomalous reflectance vector at a random date and asks the model to see
through it using the dates on either side. A model trained this way has seen, millions of times,
inputs where one date is wrong and has learned to reconstruct it from context. That is precisely the
capability our from-scratch deep models lacked when they met a zero-filled September they had never
seen in training; it is why the damage (0.08–0.14) was so large and why the sparsemax gate's +0.06 is
the only robustness we have. SITS-BERT-style pretraining, or even its pretext task used purely as a
training-time augmentation without the pretraining stage, is a principled alternative to median fill:
it makes the model robust to the corruption rather than making the corruption disappear. Note the
important asymmetry with our case: they corrupt observations *during training* and test on clean
data; we trained on clean data and tested on corrupted data. The paper does not run the second
configuration, so it demonstrates the mechanism but not the cure for our exact failure.

**Zero padding for trees, done consistently, is harmless; done inconsistently, it is our finding.**
Their SVM and RF inputs are zero-padded to the maximum series length, i.e. missing observations are
represented by zeros in both training and test, and RF is still the best classical model. This is the
benign version of the operation that broke our pipeline: the fill is fine when the model learns it as
part of the input distribution, and catastrophic (TabNet 0.61 to 0.51; deep models minus 0.08–0.14)
when it appears only at test time. Their design shows that the remedy is consistency, not avoiding
fills altogether.

**On the dense-model lead (finding 1) and DOY encoding.** SITS-BERT, like L-TAE and TimeMatch's
PSE+LTAE, encodes acquisition day of year and accepts irregular, variable-length series, so a missing
month is simply an absent token rather than a slot that must be filled. Our models operate on a fixed
10-month grid, which is what forced the imputation decision in the first place. If we rebuild the
Transformer and L-TAE inputs as `(observation, month)` tuples with a padding mask, the September
problem becomes a masking problem with no imputed values at all; this paper and Garnot et al. (2022)
both show that attention models trained that way tolerate missing dates well.

**On finding 2.** No sparsity or gating mechanism is studied; the paper does not bear on the
sparsemax gate beyond the general point that noise robustness in their models comes from training
exposure, not from architectural sparsity.

**What this paper cannot support.** Its classifier accuracies (OA 93–98 %) are from random pixel
sampling inside one scene with no field-level split, and must not be compared with our
disjoint-tile macro-F1. It does not test a fine-tuned model on an unlabeled tile, so "positive
transfer" in the paper means transfer of the *pretrained representation*, not of the classifier.

## Evaluation Caveats

- **Pooled pixel sampling, no field split.** Training, validation, and test pixels are drawn from the
  same tile on a 10-pixel grid; nothing prevents pixels of the same field, or adjacent pixels, from
  landing in both training and test. Accuracies of 93–98 % reflect within-scene interpolation. This
  is the leakage pattern our manuscript warns about, in its purest form.
- **"Transfer" refers to the pretrained weights only.** Datasets 2 and 3 test whether California
  pretraining helps fine-tuning in Missouri and Beijing; the fine-tuned classifier is never applied
  outside the tile it was fine-tuned on. No cross-region, cross-tile, or cross-year classifier
  evaluation exists in the paper despite the DOY-encoding motivation.
- **Labels are from products, not ground truth.** CDL (30 m, upsampled) and FROM_GLC10; filtered for
  homogeneity and confidence, but label noise remains, and the authors themselves attribute
  cropland/grassland confusion in Beijing to reference errors.
- **Class imbalance is removed by design.** Exactly 100 (or 50–500) samples per class for training;
  the test set keeps the natural distribution. OA on the test set is therefore dominated by majority
  classes; average accuracy (mean per-class recall) is the balanced metric and is reported, but
  macro-F1 and per-class precision are not. The only per-class numbers in the text are the two
  anecdotes (pistachios, cropland).
- **Tables II–IV are images in the PDF**; the gains quoted here are the averaged differences stated
  in the running text, and absolute baseline scores are not recoverable from the extracted text.
- **Cloud-free inputs.** Scene-classification masking removes clouds, shadows, and snow before
  training; the robustness learned in the pretext task is to synthetic corruption, and the test data
  are clean. No test-time corruption or missing-date experiment is run.
- **Pretraining corpus is modest** by the authors' own admission ("far from enough to fully exploit
  the potential of transformers"), and all of it is Californian; the Beijing gain is the only
  evidence that the representation generalizes across climates.
- **Single run per configuration;** no variance reported in the text for the main tables.
- **Compute reported for inference only;** pretraining cost on 6.5 million series is not given.
