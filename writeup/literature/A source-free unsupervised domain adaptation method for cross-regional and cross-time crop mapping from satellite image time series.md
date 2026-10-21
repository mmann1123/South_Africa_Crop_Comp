# A source-free unsupervised domain adaptation method for cross-regional and cross-time crop mapping from satellite image time series

**Citation:** Mohammadi, S., Belgiu, M., & Stein, A. (2024). A source-free unsupervised domain
adaptation method for cross-regional and cross-time crop mapping from satellite image time series.
*Remote Sensing of Environment*, 314, 114385. DOI: 10.1016/j.rse.2024.114385 (verified against
Crossref; title, authors, volume and article number match).
**BibTeX key:** `mohammadi2024sourcefree`

## Objectives

The paper proposes the first source-free unsupervised domain adaptation (SFUDA) method for crop
mapping from satellite image time series. The practical motivation is that conventional
unsupervised domain adaptation (UDA) needs the labelled source dataset at adaptation time, which
is expensive in storage, compute and energy when the source is a large national label set. The
SFUDA method instead takes an off-the-shelf model pretrained on the source region and adapts it to
an unlabelled target region (or target year) using only the target pixels, by minimising a mutual
information loss whose discriminability and diversity terms are balanced with a novel unsupervised
weighting strategy based on mean softmax confidence.

Three secondary objectives matter for us. First, the authors deliberately test the method with
three different deep backbones (1D CNN, Transformer, attention-based bidirectional LSTM) to show it
is model-agnostic, and they include Random Forest and an MLP as conventional baselines. This makes
the paper a compact multi-architecture source-only transfer benchmark, independent of the
adaptation contribution. Second, they separate cross-time transfer (same site, later year) from
cross-regional transfer (different site, later year) and quantify each. Third, they test
cross-country and cross-sensor transfer (Landsat in the U.S. to Sentinel-2 in France and back) and
a harder six-class setting (Denmark and France, TimeMatch dataset), plus within-season transfer.

## Methods

### Data and features

The main experiments use Landsat 7 ETM+ and Landsat 8 OLI Collection 2 ARD surface reflectance at
30 m over three 45 km x 45 km sites in Iowa (`IA`), Missouri (`MO`) and Mississippi (`MS`),
spanning the Dfa cold and Cfa temperate climate zones, for 2019 (source) and 2020 and 2021
(targets). Six optical bands (blue, green, red, NIR, SWIR1, SWIR2) are used per date. Time series
are built at a fixed seven-day interval from 20 March to 25 September (28 steps), following Xu et
al. (2020). Labels are the USDA Cropland Data Layer (CDL) collapsed to three classes: corn,
soybean and Other. A large 450 km x 300 km site (`L`, six ARD tiles, 145.7 million pixels) is used
for a scale test. Cross-country tests use PASTIS (Sentinel-2, France, corn and soybean plus a
500K-sample Other class) and the TimeMatch dataset (Sentinel-2, Denmark tile 32VNH and France tile
31TCJ, six classes: corn, meadow, winter barley, winter rapeseed, winter wheat, Other; all bands
except B01, B09, B10).

The input is the raw temporal sequence (28 x 6 for Landsat), not engineered time-series
statistics. Granularity is pixel-level; no field or parcel aggregation is applied. Class
proportions differ strongly between sites (corn 54.5 percent in IA 2019 versus 12.7 percent in MS
2019), so the experiments include label shift as well as covariate shift.

### Missing-data handling (relevant to our missingness-shift finding)

This is handled identically for every model and is stated explicitly. A valid observation is one
not flagged as filled, shadow, cloudy, dilated cloud or unclear. A pixel is kept only if it has at
least two valid observations in every third of the growing season; other pixels are excluded from
both training and evaluation (Table 1 reports validity of 86.4 to 99.99 percent per site-year).
Remaining gaps are filled by linear interpolation along time. There is no zero-filling and no
model-specific imputation path. The authors list cloud contamination as an open limitation and
suggest sensor fusion or GAN reconstruction as future work.

### Models

Backbones: a 1D CNN (five conv layers, 64 to 1024 filters, kernel 5, batch norm, ReLU, global
average pooling), a Transformer (hidden 128, two heads, five layers, Russwurm et al. 2020
implementation), and an attention-based bidirectional LSTM (256 hidden, two layers, Xu et al.
2020). RF uses 200 trees and four features per split; the MLP has four hidden layers (256, 512,
512, 256). Pretraining uses Adam at 1e-4 with early stopping on a random 20 percent of source
pixels. The SFUDA stage trains the backbone at 1e-4 and the head at 1e-3 for one epoch per
hyperparameter setting, sweeping the discriminability weight over 0 to 1 and the diversity weight
over 0 to 0.9, keeping models whose predictions assign at least 5 percent of samples to every
class, and finally averaging softmax outputs weighted by each model's mean confidence.

The compared adaptation methods (MMD, DANN, CDAN+E, JUMBOT, SpADANN, SHOT, AaD, and ELECTS for
within-season) all share the same CNN backbone and are initialised from the same source model, so
Table 5 is a fair same-architecture comparison of adaptation strategies.

### Evaluation protocol

**Evaluation protocol:** The load-bearing results are true spatially and temporally disjoint
source-only transfer. Models are trained on all valid pixels of one 45 km site in 2019 (minus a
random 20 percent pixel-level validation set used only for early stopping), then applied without
any retraining to (a) the same site in 2020 and 2021 (cross-time) and (b) the other two sites in
2020 and 2021 (cross-regional), giving 6 cross-time and 12 cross-regional pairs per backbone.
Metric is macro F1 over the three classes; per-class precision, recall and F1 are shown for
selected pairs (Fig. 9). Two-sample t-tests compare adapted and non-adapted scores across pairs.

Where this sits on the spectrum that our manuscript is built around: there is no in-region,
same-year held-out test reported at all. The closest thing to an in-region reference is the
cross-time pair on the same site (for example `IA19 to IA20`), which already includes a one-year
shift. The source-side random 20 percent validation split is pixel-wise within the same scene and
therefore spatially leaky, but it is used only for early stopping and never reported as a result.
Test sets are whole target sites with no pixel overlap with training, so the transfer numbers
themselves are free of spatial leakage. Labels on both sides are CDL, a classification product
(the authors cite high CDL accuracy for corn and soybean and provide a CDL-confidence-filtered
rerun in the supplement), so the evaluation is agreement with a map, not with field survey data.

## Key Findings

### Source-only transfer numbers by model family (Table 2, macro F1 in percent)

Cross-time (same site, 2019 model applied to 2020 and 2021), six pairs, averages: CNN 72.4 (range
52.8 to 90.2), Transformer 72.9 (53.3 to 91.3), LSTM 82.5 (68.3 to 92.9), RF 74.3 (60.1 to 85.1),
MLP 73.2 (60.5 to 86.4). The Iowa pairs are easy for everyone (84.7 to 92.9); Mississippi pairs
are hard (CNN 52.8 and 62.5, Transformer 53.3 and 66.6, RF 60.1 and 70.1, LSTM 76.5 and 82.4).

Cross-regional (2019 model from one site applied to the other two sites in 2020 and 2021), twelve
pairs, averages: CNN 46.1 (29.4 to 63.5), Transformer 51.9 (21.6 to 71.4), LSTM 51.9 (19.8 to
69.4), RF 45.3 (27.7 to 59.8), MLP 46.9 (26.6 to 65.4). The worst pairs are `MS19 to IA20/IA21`,
where the LSTM collapses to 19.8 and 20.2 (its predictions failed the diversity check, so no
adaptation was applied either) and the Transformer drops to 21.6 and 31.0.

Transfer drop, cross-time to cross-regional, averaged per family: CNN minus 26.3 points,
Transformer minus 21.0, LSTM minus 30.6, RF minus 29.0, MLP minus 26.3. The authors attribute the
larger cross-regional loss to larger domain shift, illustrated by NDVI profile divergence and by a
shift in the most discriminative week between sites (Fig. 4).

### Deep versus tree ranking under source-only transfer

No ranking favours the tree model. Cross-time: LSTM (82.5) is clearly best; RF (74.3) is second
but within two points of the MLP, Transformer and CNN. Cross-regional: Transformer and LSTM tie at
51.9, MLP 46.9, CNN 46.1, RF 45.3 last. The margin between the best deep sequence model and RF is
8.2 points cross-time and 6.6 points cross-regional. Rank order among deep backbones is unstable
across pairs (LSTM best on average cross-time but catastrophic on `MS19 to IA`), which is itself a
finding: with a single holdout pair, the deep-model ranking can flip.

No in-region/out-of-region ranking inversion is measurable because no in-region score is given.

### Adaptation gains (the paper's contribution)

SFUDA improves 51 of 54 U.S. transfer scenarios. Averages after adaptation: cross-time CNN 85.3
(plus 12.9), Transformer 80.0 (plus 7.1), LSTM 88.3 (plus 5.8); cross-regional CNN 66.2 (plus
20.1), Transformer 64.4 (plus 12.5), LSTM 60.7 (plus 8.8). Cross-regional gains are significant
at p below 0.05 for all three backbones; cross-time gains are significant only for the CNN. All
adapted models beat RF and MLP in the vast majority of cases. On the ten-pair benchmark (Table 5)
the method averages 65.9 versus 48.5 non-adapted, 55.2 for AaD (second best), 54.6 JUMBOT, 53.4
MMD, 53.2 CDAN+E, 50.9 DANN, 50.3 SHOT and 44.7 SpADANN. On the 450 km x 300 km site, the CNN
goes from 89.2 to 92.6.

### Cross-country and cross-sensor (Table 3, CNN, non-adapted then adapted)

`IA19 to FR` 40.6 to 47.0; `MO19 to FR` 20.0 to 44.2; `MS19 to FR` 21.2 to 39.6; `FR to IA20`
64.8 to 81.1; `FR to IA21` 54.4 to 80.9; `FR to MO20` 56.2 to 54.5 (the one failure); `FR to
MO21` 56.6 to 66.1; `FR to MS20` 53.6 to 63.8; `FR to MS21` 41.7 to 61.9. Six-class TimeMatch:
`DK to FR` 34.8 to 39.5 and `FR to DK` 25.5 to 33.9. These are very low absolute numbers for a
dense CNN under cross-continent, cross-sensor transfer, with or without adaptation.

### Mechanism and failure analysis

Confusion matrices show that non-adapted models over-predict the heterogeneous Other class
(many corn and soybean pixels labelled Other), giving Other high recall and low precision;
adaptation reverses this at the cost of Other recall. t-SNE and silhouette scores show tighter
clusters after adaptation. The weighting strategy matters: using only the single most confident
model, or an unweighted loss, is sometimes worse than no adaptation (Tables 7 and 8). Two
failures (`IA19 to MO20` Transformer, `FR to MO20`) occur when mean confidence does not track
accuracy; an oracle model choice would have scored 62.3 and 68.3 instead of 50.9 and 54.5.

## Relevance to Our Crop-Classification Study

This paper is the best same-architecture, identical-preprocessing, source-only transfer benchmark
in the corpus for the question our revised analysis now asks: once missing data are handled the
same way for every model, do dense temporal pixel models transfer better or worse than trees?

**Missingness shift and identical handling.** Mohammadi et al. apply one preprocessing path to all
models: drop pixels with too few valid observations in any third of the season, then linearly
interpolate the remaining gaps along time. There is no zero-filling and no model-specific branch.
This is the design our corrected pipeline now approximates by filling the missing September month
from training-tile medians rather than zeros. Their results therefore represent the "no
missingness artefact" condition, and in that condition the deep sequence models are never worse
than RF under transfer and are better on average. That is consistent with our corrected holdout
ordering (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63, TabNet 0.62,
tree/linear/MiniRocket 0.59 to 0.61) and inconsistent with the earlier reading in which dense
temporal nets appeared to overfit the training region. Their Limitation 4 (clouds damage the
integrity of phenological information and need better handling) is also a citable statement that
input completeness is an unsolved transfer confounder, which is precisely what our September
artefact turned out to be.

**Deep versus tree ranking.** Under cross-regional source-only transfer, Transformer and LSTM
(51.9) beat MLP (46.9), CNN (46.1) and RF (45.3); under cross-time, LSTM (82.5) beats RF (74.3) by
eight points. There is no condition in which RF leads. The within-country margin of deep over tree
(6 to 8 points) is of the same order as our corrected deep-over-tree margin (0.64 to 0.66 versus
0.59 to 0.61). The paper does not report an in-region score, so it cannot show or refute a
ranking inversion; what it does show is that the ranking among deep backbones is unstable across
transfer pairs (LSTM collapses on `MS19 to IA`), which supports our choice to report the holdout
with multiple seeds and to treat small inter-model gaps as within noise.

**Transfer distance and gap magnitude.** Their cross-time gap (one year, same site) is already
large for the hard sites, and the cross-regional gap (different U.S. states, different climate
zones, plus a year) costs a further 21 to 31 macro-F1 points. Cross-continent and cross-sensor
transfer collapses to 20 to 65 points. Our adjacent-tile, same-year transfer gaps of minus 0.06 to
minus 0.14 from in-region 0.72 to 0.78 are at the mild end of this spectrum, as expected for a
shared climate and growing season. Citing their numbers makes the point that our holdout is a
realistic operational setting (adjacent tile, same season), not a stress test, and that the
remaining gap is plausibly covariate shift in phenology rather than architecture-induced
overfitting.

**Sparse gating and the low-data regime.** The paper has no sparse feature-selection component and
no learning-curve experiment, so it offers nothing directly on findings (2) and (3). Its RF
baseline uses only four features per split (a strong sparsity prior) and still trails the deep
models, which is weak evidence that sparsity per se is not what buys transfer. What the paper
offers instead is a much larger lever than any architectural choice: unsupervised adaptation on
the unlabelled target pixels gives plus 9 to plus 20 macro-F1 points cross-regionally. Our holdout
tile has exactly the ingredients (a pretrained source model, abundant unlabelled target pixels)
and SFUDA needs no access to training data at adaptation time, so this is the obvious next
experiment after the imputation fix. It is also a cleaner comparison than our sparsemax gate for
the robustness question: adaptation handles covariate shift in the signal, whereas the gate only
buys robustness to the zero-filled month (plus 0.06) and nothing on complete inputs.

**What to cite it for.** (i) A multi-backbone source-only transfer table where deep temporal
models match or beat RF when gap handling is identical. (ii) The magnitude of within-country
cross-regional loss (roughly 25 to 30 macro-F1 points, Landsat, three classes) as a scale
reference for our adjacent-tile gap. (iii) Explicit statement that cloud-induced missingness is an
unresolved confounder for transfer. (iv) SFUDA as the natural follow-on for our holdout tile.

## Numbers at a Glance (for cross-paper comparison)

| Item | Value |
|---|---|
| Transfer setting | Cross-state within U.S. (IA, MO, MS; 45 km sites; different climate zones) plus one year; also cross-year same site; cross-country/sensor (Landsat U.S. to Sentinel-2 France); Denmark to France six-class |
| Granularity and input | Pixel-level, raw 28-step weekly sequence, six Landsat bands, three classes |
| In-region reference | Not reported; closest is same-site cross-year (`IA19 to IA20` 84.7 to 90.6) |
| Source-only cross-time macro F1 (avg) | CNN 72.4, Transformer 72.9, LSTM 82.5, RF 74.3, MLP 73.2 |
| Source-only cross-regional macro F1 (avg) | CNN 46.1, Transformer 51.9, LSTM 51.9, RF 45.3, MLP 46.9 |
| Cross-time to cross-regional drop | CNN minus 26.3, Transformer minus 21.0, LSTM minus 30.6, RF minus 29.0, MLP minus 26.3 |
| Cross-country source-only (CNN) | U.S. to France 20.0 to 40.6; France to U.S. 41.7 to 64.8 |
| Deep versus tree direction | Deep at or above RF in both settings; LSTM plus 8.2 cross-time, Transformer/LSTM plus 6.6 cross-regional |
| In-region/out-of-region inversion | Not measurable (no in-region score); deep ranking unstable across pairs |
| Missing data | Valid-observation filter plus linear interpolation, identical for all models; no zero fill |
| Adaptation gain (SFUDA) | Cross-regional plus 20.1 (CNN), plus 12.5 (Transformer), plus 8.8 (LSTM) |
| Learning curve / sparse gate | Not studied |

The two numbers to carry forward are the cross-regional source-only averages (deep 46 to 52
versus RF 45) and the cross-regional adaptation gain (plus 9 to plus 20), which together say
that under identical gap handling dense temporal models are not the weak link in transfer and
that unsupervised adaptation is a larger lever than architecture.

## Evaluation Caveats

- No in-region, same-year reference score is reported, so the paper cannot speak to in-region
  versus out-of-region ranking inversions. The least-shifted reference is a one-year cross-time
  pair on the same site.
- Labels are CDL on both source and target sides. CDL is a classification product; agreement with
  CDL is not agreement with ground truth. The supplement reruns with CDL-confidence filtering, but
  the headline tables use all valid pixels.
- Only three classes in the main experiments (corn, soybean, Other), two of which have very
  distinct phenology. The Other class is a heterogeneous merge of everything else and drives much
  of the transfer error. Our five winter crops with overlapping phenology are a harder
  discrimination, so absolute numbers do not carry over.
- Pixel-level evaluation with no field aggregation. Our field-level FID vote would be expected to
  raise their scores and narrow inter-model gaps.
- The source-side 20 percent validation split is random at pixel level within one scene, so early
  stopping and any hyperparameter choices made on it are spatially leaky. The reported transfer
  numbers are not affected because the test sites are disjoint.
- Macro F1 is the right balanced metric and is used throughout; Kappa and per-class figures are
  given only for selected pairs.
- Pixels with insufficient valid observations are excluded from evaluation (up to 13.6 percent of a
  site-year). The transfer scores therefore describe the cleaner part of the landscape; a pipeline
  that must classify every pixel, as ours does, faces a harder test.
- Adaptation hyperparameters (the gamma and beta grids, alpha equals 5) are fixed a priori and the
  model-selection rule is unsupervised, which is methodologically clean; but the comparison methods
  were tuned using the source validation split, a mild asymmetry the authors acknowledge.
- Cross-country and TimeMatch six-class scores are very low in absolute terms (25 to 65) and are
  reported for the CNN backbone only.
- Deep-learning results are not multi-seed; single runs per pair, with significance assessed across
  pairs rather than across seeds.
- Compute cost per model is not reported beyond the environmental-cost motivation; RF versus deep
  training cost is not compared.
