# Evaluating the spatial–temporal transferability of models for agricultural land cover mapping using Landsat archive

**Citation:** Wijesingha, J., Dzene, I., & Wachendorf, M. (2024). Evaluating the spatial–temporal
transferability of models for agricultural land cover mapping using Landsat archive. *ISPRS
Journal of Photogrammetry and Remote Sensing*, 213, 72–86. DOI: 10.1016/j.isprsjprs.2024.05.020
(verified against Crossref; title, authors, volume and pages match).
**BibTeX key:** `wijesingha2024transferability`

## Objectives

The authors want to map five agricultural land-cover classes (maize, grassland, winter crops,
summer crops, mixed crops) back to 2010 in two German regions, North Hesse (`NH`) and Weser-Ems
(`WE`), to study the "maizification" caused by biogas subsidies. Because reference parcel data
(IACS) exist only for some region-years, they need models that transfer across years and across
regions. The paper therefore sets out to quantify, not to fix, the loss in classification
performance when a model trained on one spatial-temporal domain is applied to another. Two
research questions are posed: how much does performance decrease under temporal, spatial and
spatial-temporal transfer, and what causes the decrease (field size, elevation, slope, climate,
crop-calendar differences, dissimilarity of the predictor space).

Two classifiers are compared throughout, Random Forest and a small 2D CNN, so the paper is also a
tree-versus-neural transfer comparison. A secondary contribution is applying the dissimilarity
index and area of applicability (AOA) of Meyer and Pebesma (2021) to explain transfer loss without
target labels.

## Methods

### Data and features

Landsat 5, 7 and 8 Collection 2 Level 2 surface reflectance scenes with less than 60 percent
cloud, March to October, 2010 to 2018, processed in Google Earth Engine. Six common bands (blue,
green, red, NIR, SWIR1, SWIR2) plus four indices (NDVI, EVI, OSAVI, NDMI). Scenes are
cloud-masked, then reduced to bimonthly median composites (March-April, May-June, July-August,
September-October), giving four dates x ten layers = 40 features per year. Features are
field-level: the mean of each layer inside each IACS polygon larger than 0.9 ha. Labels are IACS
farmer declarations grouped into the five classes. NH has more than 98,000 fields and WE more
than 280,000 fields over the period; the regions differ in field size (WE larger), elevation (NH
mean around 300 m, WE 0 to 50 m), slope and class mix (maize about 25 percent of fields in WE
versus under 10 percent in NH; winter crops about 35 percent in NH versus 15 percent in WE).

So the design is engineered temporal aggregates (four bimonthly medians) at object level, not a
raw dense sequence and not pixel-level. Optical only. The temporal resolution is deliberately
coarse so that the Landsat archive can be used back to 2010.

### Missing-data handling

Clouds and shadows are masked; Landsat 7 SLC-off stripes are explicitly left uncorrected and
treated as no-data, the same as cloud. Each bimonthly composite is the median of whatever valid
scenes remain. The paper does not state what happens to a field-bimonth with zero valid
observations, and it does not describe any imputation step. Both models receive the identical
40-element vector, so whatever gap behaviour exists is shared. One asymmetry that the authors do
flag: the transfer target years 2010 to 2012 are covered only by Landsat 5 and the gap-striped
Landsat 7, whereas the training years 2013 to 2018 use Landsat 7 and 8. The number of clear scenes
per composite also differs between years. This is a train/test input-availability shift that is
unrelated to crops, analogous in kind (if not in mechanism) to our missing September month.

### Models

RF (scikit-learn): 600 trees, entropy criterion, `max_features` sqrt, `min_samples_split` 8,
tuned on a subsample. CNN (TensorFlow): the 40-vector is reshaped to a 10 x 4 array (bands by
dates) and passed through two blocks of double 3 x 3 convolutions with batch norm, ReLU and 2D max
pooling, then dense layers with 25 percent dropout; about 96,000 parameters; SGD, learning rate
1e-4, momentum 0.9, categorical cross-entropy, 50 epochs. Four models are trained: `NH-RF`,
`NH-CNN`, `WE-RF`, `WE-CNN`, each on its region's 2013 to 2018 data.

### Evaluation protocol

**Evaluation protocol:** Four scenarios. The reference (no transfer) holds out one district from
the training region in 2017 (Schwalm-Eder, DE735, for NH; Cloppenburg, DE948, for WE), trains on
the rest of 2013 to 2018, and tests on the held-out district-year. This is a spatially disjoint
sub-region holdout within the training region and the training period, so it is stronger than
field-wise k-fold but weaker than a different region. S1 (temporal) tests the same region in 2010
to 2012. S2 (spatial) tests the other region in 2013 to 2018. S3 (spatial-temporal) tests the
other region in 2010 to 2012. Metrics: OA, class-level F1 and macro F1 (`mF1`), plus percent
change of each metric relative to the reference. The explanatory analysis computes DI and AOA
(CAST package) for every test observation, then correlates the share of test fields inside the
AOA with OA across scenarios, and compares field size, elevation, slope, temperature and
precipitation distributions between reference and transfer sets.

Splitting is by polygon (objects), so there is no pixel-level leakage; the reference test is a
whole district, so neighbouring-field autocorrelation between train and test is limited to the
district boundary. The S2 and S3 tests are a different region roughly 200 km away with different
soils, relief and class mix, which is a genuine spatial holdout. Mixed-crops and summer-crops are
minority classes and are the ones with low F1, which macro F1 exposes.

## Key Findings

### Reference (in-region, held-out district, 2017), Table 3

`NH-RF` OA 90.2, mF1 0.765; `NH-CNN` OA 89.7, mF1 0.758; `WE-RF` OA 88.3, mF1 0.722; `WE-CNN`
OA 90.1, mF1 0.740. Winter crops exceed 0.95 F1, maize and grassland exceed 0.85, summer crops
0.75 (NH) and 0.65 (WE), mixed crops lowest. The authors report no significant difference between
RF and CNN in-region; the CNN is marginally ahead in WE (plus 0.018 mF1) and marginally behind in
NH (minus 0.007).

### Transfer (Table 4, OA percent and mF1)

| Model | S1 temporal | S2 spatial | S3 spatial-temporal |
|---|---|---|---|
| `NH-RF` | 74.1 / 0.645 | 85.0 / 0.645 | 84.4 / 0.647 |
| `NH-CNN` | 83.7 / 0.678 | 82.8 / 0.614 | 80.5 / 0.598 |
| `WE-RF` | 85.4 / 0.698 | 82.2 / 0.678 | 70.0 / 0.588 |
| `WE-CNN` | 83.6 / 0.695 | 75.5 / 0.643 | 62.1 / 0.563 |

Abstract-level summary: RF mF1 goes from 0.74 (reference, averaged) to 0.67 (S1), 0.66 (S2),
0.62 (S3); CNN from 0.75 to 0.68, 0.62, 0.58.

Transfer drop in absolute mF1 from the model's own reference. RF: NH minus 0.120 (S1), minus
0.120 (S2), minus 0.118 (S3); WE minus 0.024, minus 0.044, minus 0.134. CNN: NH minus 0.080,
minus 0.144, minus 0.160; WE minus 0.045, minus 0.097, minus 0.177. The authors state that
spatial transfer (S2) changed absolute mF1 by between minus 0.04 and minus 0.14, and that S3
costs about 25 percent of reference performance. Order of loss: temporal less than spatial less
than spatial-temporal, except `NH-RF`, whose loss is flat at about minus 15.5 percent across all
three scenarios.

### Deep versus tree ranking and ranking inversion

In-region: CNN ahead or tied (WE 0.740 versus 0.722; NH 0.758 versus 0.765). Under spatial
transfer (S2) RF leads in both regions (NH 0.645 versus 0.614; WE 0.678 versus 0.643), and under
S3 again (0.647 versus 0.598; 0.588 versus 0.563). Under purely temporal transfer the two are
tied in NH (0.645 versus 0.678 favouring CNN) and in WE (0.698 versus 0.695). So the paper does
report a mild in-region versus out-of-region ranking inversion for the WE models: the CNN is best
in-region by 0.018 mF1 and worst under spatial transfer by 0.035 mF1. The authors' explanation is
not overfitting of a dense model; it is that "limited temporal observation in the dataset (four
per variable) might have restricted the potential of CNN models" and that CNN transfer needs
"multiple temporal observations" to learn the difference between domains. Their conclusion
bullet nevertheless says the two algorithms "showed no substantial differences".

### Class-level patterns

Maize F1 is stable or improves under transfer for NH models (transferring from a region with few
maize fields to one with many) and decreases for WE models. Summer and winter crops decline
monotonically from S1 to S3 for NH models. The class whose share differs most between regions
(winter crops for NH to WE, maize for WE to NH) loses the most, which the authors attribute to
label-distribution shift, citing Rusnak et al. (2023).

### Explanatory analysis

AOA share: NH models 88, 90, 89, 78 percent (reference, S1, S2, S3); WE models 93, 90, 90, 87
percent. Correlation between OA and AOA share is 0.9. Field size, elevation and slope differ
sharply between regions; the model trained on the small-field, hilly NH region loses less when
sent to the large, flat WE fields (minus 15 percent for `NH-CNN`) than vice versa. Temperature and
precipitation show no visible shift between regions or periods, so climate is not the explanation
in this case. Four observations per year gave 79.8 percent mean S1 accuracy against 77.3 percent
for the two-date Kyere et al. (2019) setup, so densification helps temporal transfer.

## Relevance to Our Crop-Classification Study

This is the clearest paper in the batch in which a tree model out-transfers a neural model, and
it is therefore the one a referee would cite against our revised finding (1). Read carefully, it
supports the revised reading rather than the old one.

**What it actually shows about deep versus tree.** The CNN here is not a dense temporal sequence
model. It is a 96,000-parameter 2D convolution over a 10 x 4 band-by-date grid built from four
bimonthly medians, and both models consume the same 40 engineered field-level aggregates. Its
in-region edge over RF is 0.018 mF1 in one region and minus 0.007 in the other; its spatial
transfer deficit is 0.03 mF1. The authors attribute the deficit to too few temporal observations
for a convolutional model, not to density or overfitting of the training region, and their
headline conclusion is that the two algorithms do not differ substantially. Our corrected holdout
has the opposite sign (deep pixel sequence models 0.64 to 0.66 versus trees 0.59 to 0.61, with
dense 12-step sequences), and the two results are reconcilable through the authors' own
mechanism: the advantage of a temporal network appears only when there is a temporal sequence to
exploit. Cite the paper as the boundary case where sequence density is too low for a deep model to
have any inductive advantage, rather than as evidence of a general tree-transfers-best rule.

**Missingness shift.** The paper's temporal scenario embeds exactly the kind of non-crop input
shift that produced our artefact: the target years 2010 to 2012 are observed only by Landsat 5 and
the SLC-off Landsat 7, so composites in the target period are built from fewer and gappier scenes
than those in the training period. The authors acknowledge this ("there could be an impact from
changes in the satellite sensors to the model transferability") but do not quantify it, and they
do not state how a field-bimonth with no valid scene is encoded. Both models see the same vector,
so the shift is at least applied identically, which is the condition under which our corrected
ranking holds. The DI/AOA analysis is a reusable diagnostic for us: a field whose 40-vector falls
outside the AOA is exactly a field whose zero-filled September pushed it out of the training
distribution, and the 0.9 correlation between AOA share and OA is the kind of evidence that would
have flagged our September problem before any model was trained.

**Transfer distance and gap magnitude.** Their spatial transfer (another region of the same
country, roughly 200 km, same years, five classes, field-level) costs minus 0.04 to minus 0.14
absolute mF1. Our adjacent-tile, same-year transfer costs minus 0.06 to minus 0.14 for deep pixel
models from in-region 0.72 to 0.78. The two ranges overlap almost exactly, with the same spread.
This is the most directly comparable gap-magnitude reference in the corpus, and it is model
family agnostic: RF and CNN lose similar amounts in absolute terms. Their spatial-temporal
scenario (about 25 percent loss) shows what adding a year shift does on top, which we do not test.

**Sparse gate and learning curves.** Nothing on either. RF with `max_features` sqrt is itself a
sparse-split model and does not transfer better in absolute mF1 than the CNN in the temporal
scenario, which is weak evidence that sparsity is not the operative lever. Training set sizes
(98,000 to 280,000 fields) are far larger than ours and no size ablation is run, so the paper
cannot speak to our finding (3). Their observation that the NH model (small fields, more mixed
pixels) transfers better to WE than the reverse is a data-quality-versus-quantity point worth
keeping in mind when interpreting our learning curves: harder training data can produce a more
transferable model at equal size.

**What to cite it for.** (i) The only clean in-region versus out-of-region table in the batch with
a tree and a neural model under identical features, showing a small (0.03 mF1) inversion that the
authors themselves attribute to sequence length rather than model density. (ii) Spatial transfer
gap of minus 0.04 to minus 0.14 mF1 at a comparable distance and class count. (iii) Sensor and
scene-availability shift between training and target periods as an acknowledged but
unquantified confounder. (iv) DI/AOA as a label-free diagnostic of input-distribution shift.

## Numbers at a Glance (for cross-paper comparison)

| Item | Value |
|---|---|
| Transfer setting | Two German regions about 200 km apart (NH hilly, small fields; WE flat, large fields); training years 2013 to 2018; target years 2010 to 2012; Landsat 5/7/8 |
| Granularity and input | Field-level means of four bimonthly median composites x ten layers (40 features), five classes |
| In-region reference (held-out district 2017), mF1 | `NH-RF` 0.765, `NH-CNN` 0.758, `WE-RF` 0.722, `WE-CNN` 0.740 |
| Temporal transfer S1, mF1 | `NH-RF` 0.645, `NH-CNN` 0.678, `WE-RF` 0.698, `WE-CNN` 0.695 |
| Spatial transfer S2, mF1 | `NH-RF` 0.645, `NH-CNN` 0.614, `WE-RF` 0.678, `WE-CNN` 0.643 |
| Spatial-temporal S3, mF1 | `NH-RF` 0.647, `NH-CNN` 0.598, `WE-RF` 0.588, `WE-CNN` 0.563 |
| Spatial drop from reference, mF1 | RF minus 0.120 (NH), minus 0.044 (WE); CNN minus 0.144 (NH), minus 0.097 (WE) |
| Deep versus tree direction | In-region CNN ahead or tied; under S2 and S3 RF ahead by 0.025 to 0.049 mF1 |
| In-region/out-of-region inversion | Yes, mild (WE: CNN plus 0.018 in-region, minus 0.035 under S2); authors attribute to only four time steps |
| Missing data | Cloud and SLC-off gaps treated as no-data, bimonthly median; no imputation described; identical vector for both models; sensor mix differs between training and target years |
| Label-free diagnostic | DI/AOA share correlates 0.9 with OA across scenarios |
| Learning curve / sparse gate | Not studied; 98,000 to 280,000 training fields |

The numbers to carry forward are the S2 absolute mF1 change (minus 0.04 to minus 0.14), which
brackets our adjacent-tile gaps, and the size of the tree-over-CNN margin under transfer (about
0.03), which is small, single-run, and explained by the authors through sequence length rather
than model density.

## Evaluation Caveats

- The reference scenario holds out one district-year inside the training region and period; it is
  a sub-regional spatial holdout, not field-wise k-fold, but neighbouring fields across the
  district boundary share management and weather, so it is still an optimistic "in-region" number.
- The CNN is a 2D convolution over a 10 x 4 engineered-feature grid, not a temporal sequence model;
  treating its transfer deficit as a statement about deep temporal networks would be a category
  error. The authors say as much.
- Only four observations per year. The paper is explicitly a sparse-Landsat-archive study; its
  model comparison should not be generalised to dense Sentinel-2 sequences.
- Gap handling for field-bimonths with no clear scene is not described. The training period uses
  Landsat 7 and 8; the temporal target period uses Landsat 5 and SLC-off Landsat 7. This is an
  acknowledged, unquantified train/test input shift.
- Five coarse classes, including a heterogeneous "mixed crops" class with the lowest F1; the
  class-level F1 figures are given as percent change plots (Fig. 9), not as absolute values in a
  table, so minority-class performance under transfer must be read off graphics.
- Macro F1 and class-level F1 are reported, which is appropriate; OA alone would have hidden the
  mixed-crop and summer-crop losses.
- Single training run per model; no seeds, no confidence intervals, and no significance test on
  the RF versus CNN differences, which are of the order of 0.02 to 0.03 mF1.
- Label distribution differs between regions (maize share 25 versus 10 percent), so part of the
  spatial loss is label shift, which the authors note but do not separate from covariate shift.
- Climate variables are inspected visually only; the authors state they did not explore this in
  depth.
- No computational cost comparison and no in-season or early-season analysis.
- IACS reference data are not shareable; code is public, data are not reproducible by third
  parties.
