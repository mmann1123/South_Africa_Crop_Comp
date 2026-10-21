# Transfer Learning for Cross-Regional Crop Mapping

**Citation:** Pandžić, M., Pavlović, D., Marko, O., & Kilibarda, M. (2026). Transfer Learning for
Cross-Regional Crop Mapping. *Geodetski vestnik*, 70(2), 268–288.
DOI: 10.15292/geodetski-vestnik.2026.02.268-288 (verified against Crossref: authors, journal,
volume 70, issue 2, pages 268–288 and year match. Note: Crossref records the title as "Spatial
Transfer Learning for Cross-Regional Crop Mapping", whereas the PDF prints "Transfer Learning for
Cross-Regional Crop Mapping" on the title page and in its own citation footer; the PDF footer
also mis-prints the DOI without the `10.15292/` prefix. Treat the DOI as verified and the title
as the PDF form.)
**BibTeX key:** `pandzic2026transfer`

Identification from the first page: bilingual Slovenian/English peer-reviewed article in
Geodetski vestnik (the journal of the Association of Surveyors of Slovenia), UDK 528.9+633:004.85,
received 20 January 2026, accepted 12 March 2026, authors at the BioSense Institute (University of
Novi Sad) and the University of Belgrade. Code is at
`github.com/milospandzic/TL_4_crossregional_crop_mapping`.

## Objectives

The paper studies cross-regional transfer of crop type classifiers from a label-rich source
(Slovenia, 159,533 parcels from mandatory farmer declarations) to a label-poor target (Vojvodina,
Serbia, 2,399 field-surveyed parcels), for nine crop types in the 2021 season. Three objectives:
(1) test whether transfer learning is applicable between the two regions; (2) compare transfer
strategies under limited target labels against the conventional approach of training within the
target region; (3) analyse what drives transferability. The strategies are naive off-the-shelf
transfer (apply the Slovenian model to Serbia, no target labels), transfer learning with 15
percent of Serbian parcels (RF: add Serbian-trained trees to the Slovenian forest; Transformer:
fine-tune the classification head with a frozen encoder, or fine-tune everything), and
from-scratch Serbian models as the in-region benchmark. Two algorithms (Random Forest and an
encoder-only Transformer) and three modalities (Sentinel-1, Sentinel-2, both) are crossed.

## Methods

### Data and features

Sentinel-2 L2A (12 bands, `B10` excluded) plus ten indices (NDVI, EVI, EVI2, NDMI, SAVI, PSRI,
SeLI, ARVI, ExG, chlorophyll red-edge) and Sentinel-1 GRD IW Gamma0 (VV, VH, VV/VH after border
noise correction, radiometric terrain normalisation and Lee speckle filtering), 25 layers total,
all at 10 m, 1 April to 30 September 2021. Clouds are masked with Cloud Score+. Features are
parcel-level: the median of valid pixels inside each parcel on each acquisition date. Nine
classes: corn, wheat, soybean, sunflower, sugar beet, oilseed rape, barley, clover, orchard.
Parcel sizes differ by an order of magnitude (Slovenia mean 0.78 ha, median 0.46 ha; Serbia mean
23.25 ha, median 14.53 ha), and Serbian surveys deliberately prioritised large homogeneous
parcels. 325 feature-wise Kolmogorov-Smirnov tests reject identical source and target marginal
distributions in 324 cases, which the authors use to establish that a domain shift exists.

### Missing-data handling (relevant to our missingness-shift finding)

The irregular per-parcel time series (different acquisition dates and cloud losses in the two
regions, and between parcels of one region) are resampled onto a common 15-day grid by cubic
spline interpolation, with forward padding before the first observation and backward padding
after the last, so every parcel in both domains has a complete, equal-length sequence with
identical feature count per modality. The authors present this explicitly as the mechanism that
"mitigates the problem of irregular data acquisitions due to e.g. cloudiness". It is applied once,
upstream of both algorithms, so no model sees a different gap treatment. No zero-filling occurs.

### Models

RF: grid search over 100/500/1000 trees, max depth 1 to 50, min leaf 1 to 100; the optimum was
1,000 trees, depth 50, leaf 1 in five of six cases. Transformer: encoder only, input embedding and
positional encoding, classification head; grid over 2 to 8 heads, hidden 64 to 256, 4 to 8
layers, dropout 0 to 0.25, learning rate sampled from 1e-2 to 1e-5; optimal Slovenian
configurations were 4 to 8 layers with 64 to 256 hidden units. 60 RF and 192 Transformer
configurations were evaluated with five-fold cross-validation per algorithm, modality and domain
(12 optimisation runs). F1 is the selection metric because of class imbalance.

### Evaluation protocol

**Evaluation protocol:** Parcel-wise splits with all pixels of a parcel on one side, explicitly
to prevent leakage. Slovenia: 70/15/15 train/validation/test, stratified. Serbia: 15/15/70
train/validation/test to simulate scarce target labels. The scenarios are: `RF-FS` and `TR-FS`
trained and tested within Serbia (the in-region benchmark, about 360 training parcels) and
within Slovenia (source in-region, shown in grey); `RF-naive` and `TR-naive` trained on Slovenia
and tested on the Serbian 70 percent test set with no target data (true cross-border spatial
holdout, same year, no adaptation); `RF-TL`, `TR-TL-F` and `TR-TL-T` initialised from the
Slovenian model and adapted with the Serbian 15 percent training set. Metrics: F1 (a single
class-averaged score, convention not stated but its gap below OA indicates macro averaging),
precision, recall and OA, plus confusion matrices for the S1+S2 benchmark, naive and TL models.
The target region is about 420 km from the source centroid, in a different country, climate and
parcel structure, so this is a genuine spatial holdout at the far end of our spectrum, with the
same calendar year.

## Key Findings

### Full results (Table 5, percent; F1 / OA)

| Scenario | S1 | S2 | S1+S2 |
|---|---|---|---|
| `RF-FS` Serbia to Serbia (in-region benchmark) | 76.34 / 90.48 | 79.17 / 91.19 | 84.35 / 93.27 |
| `TR-FS` Serbia to Serbia | 76.10 / 85.18 | 83.04 / 90.12 | 76.51 / 88.27 |
| `RF-FS` Slovenia to Slovenia (source in-region) | 39.22 / 77.24 | 60.90 / 89.72 | 62.88 / 90.04 |
| `TR-FS` Slovenia to Slovenia | 48.47 / 77.85 | 68.92 / 90.79 | 72.23 / 91.14 |
| `RF-naive` Slovenia to Serbia (source-only transfer) | 35.12 / 52.56 | 40.67 / 55.42 | 46.49 / 60.06 |
| `TR-naive` Slovenia to Serbia | 42.68 / 58.99 | 57.98 / 76.61 | 66.80 / 79.64 |
| `RF-TL` | 85.62 / 89.64 | 83.76 / 86.67 | 87.34 / 89.76 |
| `TR-TL-F` (frozen encoder) | 72.95 / 81.96 | 89.39 / 92.02 | 91.11 / 92.80 |
| `TR-TL-T` (fully fine-tuned) | 75.31 / 82.32 | 90.08 / 93.33 | 90.62 / 94.29 |

### Source-only transfer numbers and transfer drop

Naive transfer, best modality (S1+S2): RF F1 46.49, Transformer 66.80. Relative to the Serbian
in-region benchmark the drops are RF minus 37.9 F1 points (84.35 to 46.49) and Transformer minus
9.7 (76.51 to 66.80). Relative to each model's own source in-region score the drops are RF minus
16.4 (62.88 to 46.49) and Transformer minus 5.4 (72.23 to 66.80). With Sentinel-2 only: RF 40.67
(minus 38.5 from the Serbian benchmark, minus 20.2 from source in-region), Transformer 57.98
(minus 25.1 and minus 10.9). In OA terms naive RF falls to 52.6 to 60.1 and the Transformer to
59.0 to 79.6.

### Deep versus tree ranking and ranking inversion

In-region (Serbia from scratch, S1+S2) the tree model is best: RF 84.35 versus Transformer 76.51.
Under source-only spatial transfer the ranking inverts: Transformer 66.80 versus RF 46.49, a
20-point margin in favour of the dense temporal model. The inversion is also visible in OA (RF
93.3 versus 88.3 in-region; Transformer 79.6 versus 60.1 under transfer). With Sentinel-2 only the
Transformer is ahead in both settings (83.04 versus 79.17 in-region; 57.98 versus 40.67 under
transfer), so there the result is "deep better everywhere" rather than an inversion. The source
in-region scores in Slovenia also favour the Transformer in every modality (72.23 versus 62.88
for S1+S2). The authors describe naive RF transfer as "notable drops, particularly for RF,
highlighting limited generalization without adaptation".

### Transfer learning with 15 percent target labels

Fine-tuned Transformers reach 91.11 (frozen encoder) and 90.62 (fully trainable) F1 on S1+S2 and
89.39 / 90.08 on S2, exceeding the Serbian from-scratch benchmark by about 7 points; RF-TL reaches
87.34, about 3 points above its benchmark. Frozen versus fully trainable makes little difference,
which the authors read as the pretrained encoder already holding transferable features. The only
TL configuration that underperforms its benchmark is the S1-only Transformer (72.95 / 75.31 versus
76.10), attributed to speckle noise and the Transformer having only 15 percent of Serbian data for
training with another 15 percent for validation, while RF used the full 30 percent.

### Class-level behaviour

Transfer learning mostly helps crops under-represented in the source (oilseed rape, sugar beet,
orchard, soybean). Corn, wheat, clover and sunflower already transfer well naively, indicating
more transferable spectral-temporal patterns. Residual confusion is barley versus wheat, and a
small TL trade-off affects barley and clover. The lower Slovenian in-region scores are attributed
to parcel fragmentation, mixed pixels and terrain-induced radar distortion.

## Relevance to Our Crop-Classification Study

Among the five papers in this batch, this is the one whose central number speaks most directly to
our revised finding (1), and it does so in the same direction.

**Deep versus tree under source-only transfer, with an inversion.** With both sensors, RF is the
better model inside the target region (84.35 versus 76.51 F1) and the worse model by 20 points
when transferred without adaptation (46.49 versus 66.80). That is an in-region/out-of-region
ranking inversion in the opposite direction from the one our earlier draft reported: the tree
model looked best in-region and the dense Transformer transferred far better. Their inputs were
complete, equal-length 15-day sequences for both models, so the inversion cannot be an imputation
artefact, and their Kolmogorov-Smirnov tests confirm a real covariate shift. This is the cleanest
external datum we have that a dense attention model trained on raw temporal sequences generalises
across a spatial domain boundary better than a 1,000-tree depth-50 forest, which is the pattern
our corrected holdout now shows (Transformer 0.64, CNN-BiLSTM 0.66 versus trees 0.59 to 0.61).

**Missingness shift and identical handling.** Their preprocessing is close to the fix we applied.
Irregular, cloud-gapped parcel series from two regions with different acquisition histories are
spline-interpolated and edge-padded onto one grid, so neither model ever sees a sentinel value
and the two domains have identical feature counts. The authors frame this as a reproducibility
and cloud-robustness measure, not as a fairness measure between models, but its effect is the
same: no model-specific missingness path. Our September artefact arose precisely because the
raw-sequence models were given zero-filled values that no tree model ever saw; their design shows
how a pipeline avoids that by construction. Note that interpolation and padding here are
within-series (temporal neighbours), whereas our fix uses training-tile medians (cross-sample);
both are defensible, and a within-series interpolation of September from August and October
would be a natural robustness check for us.

**Transfer distance and gap magnitude.** Slovenia to Vojvodina is cross-border, about 420 km,
different parcel structure (0.8 versus 23 ha) and a confirmed distribution shift, same year, nine
classes. The Transformer's naive loss of about 10 F1 points from the target benchmark (and 5 from
source in-region) is of the same order as our adjacent-tile gaps of minus 0.06 to minus 0.14,
while the RF loss of 38 points is far larger than anything we see. The fact that a dense model's
gap at this distance is comparable to ours at tile distance, while the tree's is not, is useful
when a referee asks whether our deep-model gaps are suspiciously small.

**Low-data regime and learning curves.** Their 15 percent Serbian training set is about 360
parcels. Trained from scratch on that, RF and Transformer reach 76 to 84 F1; initialised from the
Slovenian model and fine-tuned on the same 360 parcels, the Transformer reaches 91 and RF 87. This
is a single-point learning-curve statement: at roughly a tenth of the labels, a pretrained deep
model beats both from-scratch models, and the from-scratch tree beats the from-scratch
Transformer. It complements our finding (3) that all models sit near 0.55 macro-F1 at 10 percent
of fields and that deep models pull ahead only as data grow: the lever at the low-data end is
source pretraining, not architecture. It also offers the practical recipe for our holdout tile if
even a small labelled sample there were available.

**Sparse gate.** Not studied. The Transformer here has no feature-selection or sparse attention
component and transfers well regardless, consistent with our observation that the sparsemax gate
buys nothing on complete inputs.

**Modality.** Their S1+S2 fusion is the best modality for RF and for transfer learning, and S1
alone is poor for the Transformer. We are optical-only; the paper is a reminder that SAR fusion
is the usual next lever for cross-region robustness, and that it interacts with model family.

**What to cite it for.** (i) A documented in-region/out-of-region ranking inversion in which the
tree wins in-region and the dense temporal model wins under spatial transfer by 20 F1 points,
with complete inputs and parcel-wise splits. (ii) Interpolation onto a regular grid as the
preprocessing that removes any model-specific missingness. (iii) Pretraining plus 15 percent
target labels as the low-data strategy that beats from-scratch training of either family.
(iv) Naive transfer being "unlikely" to yield universal models, as the authors put it, which
frames our holdout gaps as expected rather than as a failure of any particular family.

### Per-modality transfer drop (naive minus Serbian benchmark, F1 points)

| Modality | RF | Transformer | Transformer minus RF under transfer |
|---|---|---|---|
| S1 | minus 41.2 (76.34 to 35.12) | minus 33.4 (76.10 to 42.68) | plus 7.6 |
| S2 | minus 38.5 (79.17 to 40.67) | minus 25.1 (83.04 to 57.98) | plus 17.3 |
| S1+S2 | minus 37.9 (84.35 to 46.49) | minus 9.7 (76.51 to 66.80) | plus 20.3 |

The RF loss is nearly constant across modalities (about 38 to 41 points), whereas the Transformer
loss shrinks sharply as optical and fused inputs replace radar (33 to 10 points). The dense
model's transfer advantage therefore grows with the information content of the sequence, which
is the same dependence on sequence richness that Wijesingha et al. (2024) invoke, from the other
side, to explain why their four-date CNN did not transfer.

## Numbers at a Glance (for cross-paper comparison)

| Item | Value |
|---|---|
| Transfer setting | Slovenia to Vojvodina (Serbia), about 420 km, cross-border, same year (2021), nine classes, Sentinel-1, Sentinel-2 and fusion |
| Granularity and input | Parcel-level medians, spline-interpolated 15-day sequence, 25 layers for S1+S2 |
| In-region reference (Serbia from scratch, 15 percent train), F1 | RF 76.34 (S1), 79.17 (S2), 84.35 (S1+S2); Transformer 76.10, 83.04, 76.51 |
| Source in-region (Slovenia), F1 | RF 39.22, 60.90, 62.88; Transformer 48.47, 68.92, 72.23 |
| Source-only transfer (naive), F1 | RF 35.12, 40.67, 46.49; Transformer 42.68, 57.98, 66.80 |
| Drop from target benchmark (S1+S2) | RF minus 37.9; Transformer minus 9.7 |
| Drop from source in-region (S1+S2) | RF minus 16.4; Transformer minus 5.4 |
| Deep versus tree direction | Transformer above RF under transfer in every modality (plus 7.6, plus 17.3, plus 20.3) |
| In-region/out-of-region inversion | Yes (S1+S2): RF best in-region by 7.8 F1, Transformer best under transfer by 20.3 F1 |
| Missing data | Cloud Score+ mask, parcel median, cubic spline to 15-day grid with edge padding; identical for both models; no zero fill |
| Fine-tuning with about 360 target parcels, F1 | RF-TL 87.34; Transformer frozen 91.11, full 90.62 (S1+S2) |
| Learning curve / sparse gate | Single low-data point (15 percent); no gate |

The numbers to carry forward are the naive-transfer margin of 20 F1 points for the Transformer
over RF with complete inputs and the direction of the in-region/out-of-region inversion, which
runs opposite to the one our earlier draft reported.

## Evaluation Caveats

- The Serbian in-region benchmark is trained on only 15 percent of 2,399 parcels (about 360) and
  tested on 70 percent; it is a deliberately label-scarce benchmark, so the "in-region" number is
  not a ceiling in the way our field-wise k-fold in-region score is. The inversion is therefore
  between a small-data in-region tree and a large-source-data dense model; part of the
  Transformer's transfer advantage may be the 159,533-parcel source set rather than architecture.
- The grey Slovenia-to-Slovenia rows have a different test set from the Serbia rows, so cross-row
  comparisons of "source in-region" to "naive transfer" compare different test populations.
- Serbian ground truth prioritised large, homogeneous parcels, which makes the target test set
  easier than the landscape and inflates all Serbian scores.
- F1 averaging convention is not stated; the large gap between F1 and OA (for example 46 versus
  60 for naive RF) indicates per-class averaging, but weighted versus macro is not specified.
  Per-class F1 values are not tabulated; class behaviour is read from confusion-matrix figures.
- Single season (2021) and a single source-target pair; no temporal transfer, no reverse
  direction, and no multi-seed runs, so the 20-point naive gap has no uncertainty estimate.
- Parcel-level medians with spline interpolation smooth both the signal and the gaps; pixel-level
  heterogeneity within the large Serbian parcels is averaged away.
- Hyperparameters were optimised separately per domain, including for Serbia with its small
  training set, so the in-region benchmark is tuned in-region while the naive model is tuned on
  Slovenia; this is the realistic setting but it is not a controlled architecture comparison.
- The paper is published in a national surveying journal with a bilingual format; the English
  text is complete and the methodology is clear, but the venue is less widely indexed than the
  other four papers in this batch.
- No computational cost comparison beyond the remark that optimisation runs lasted days.
