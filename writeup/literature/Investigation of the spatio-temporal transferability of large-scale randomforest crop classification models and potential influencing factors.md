# Investigation of the spatio-temporal transferability of large-scale random forest crop classification models and potential influencing factors

**Citation:** Xie, S., Zhang, N., Liu, L., & Sun, L. (2025). Investigation of the spatio-temporal
transferability of large-scale random forest crop classification models and potential
influencing factors. *International Journal of Digital Earth*, 18(2), 2563211.
DOI: 10.1080/17538947.2025.2563211 (verified against Crossref; title, authors, volume, issue and
article number match).
**BibTeX key:** `xie2025transferability`

## Objectives

The paper asks whether a random forest crop classifier trained on freely available U.S. labels
(the Cropland Data Layer) can be applied, without any local samples, to national-scale corn and
wheat mapping in six European countries, and what determines how well such a transferred model
performs. Three experiments are designed: (1) spatial transfer of a Kansas-2019 model to Germany,
France, Romania, Poland, Spain and Italy in 2018; (2) temporal transfer, using Kansas models from
2019 through 2023 on the same 2018 European data; (3) comparison of models trained in different
states (Kansas, South Dakota, Minnesota) and on merged states and years, to test whether broader
training data yields better transfer. A fourth analysis compares transferred models to "local"
models trained on the target country's own LUCAS points, and a fifth correlates transfer accuracy
with the similarity of annual temperature and precipitation profiles between source and target.
The practical contribution is a source-region selection rule based on temperature-pattern
similarity.

## Methods

### Data and features

Landsat 8 Collection 2 surface reflectance and Sentinel-2 L2A, harmonised to six matched bands
(Landsat `B2` to `B7`; Sentinel-2 `B2`, `B3`, `B4`, `B8A`, `B11`, `B12`, resampled to 30 m), plus
seven indices (NDVI, EVI, GCVI, MSAVI, LSWI, NDSVI, NDTI). Monthly median composites from March to
October (eight months) give 13 x 8 = 104 features per pixel. Training labels are CDL corn and
wheat (durum, spring and winter wheat merged), edge pixels removed by an eight-neighbour rule,
3,000 stratified random pixels per class per state-year (6,000 total per model). Validation
labels are LUCAS 2018 in-situ points harmonised to corn and wheat, 2,069 to 6,046 points per
country. Only pixels labelled corn or wheat in the EUCROPMAP reference are classified, so the
task is strictly binary. Training areas span 0.20 to 0.23 million km2 each; mapping areas 0.24 to
0.54 million km2.

Input is a short monthly sequence of engineered composites, pixel-level, optical only, with
indices.

### Missing-data handling

Clouds are masked (`QA_PIXEL` for Landsat, `QA60` for Sentinel-2); monthly medians are computed
over all valid observations from both sensors; any month still missing is filled by linear
interpolation between the nearest valid months before and after. The result is explicitly a
"gap-free" 104-feature vector for every pixel in both training and mapping areas, with the same
procedure on both continents. 2018 U.S. data were excluded from training because Sentinel-2
coverage was insufficient to build robust time series. No zero-filling is used, and the single
model family means there is no model-specific path to compare.

### Models

Random forest with 500 trees, all other parameters at scikit-learn or GEE defaults. GEE runs the
wall-to-wall maps; a locally deployed RF on exported samples runs the scenario experiments.
Variable importance tables are in the supplement. The paper states that some deep-learning tests
were also run and that they show "varying transferability and sample aggregation effects
depending on model types and regions", but these are confined to Sections II and III of the
supplement and are not reported in the main text.

### Evaluation protocol

**Evaluation protocol:** Cross-continent, cross-year, source-only transfer validated against
independent field-survey points. Models are trained on CDL-labelled pixels in a U.S. state in
year Y (2019 to 2023) and applied, with no retraining, to Landsat/Sentinel-2 composites over a
European country in 2018; accuracy is measured against LUCAS 2018 ground points with confusion
matrices, OA, producer's and user's accuracy per class. The "local model" comparison splits each
country's LUCAS points 7:3 at random, trains an RF with identical features on 70 percent and
evaluates both the local and the transferred model on the remaining 30 percent. There is no
in-region score for the U.S. source states and no field-wise or spatially blocked split for the
local model; the 7:3 split is a random point split, which for sparse LUCAS points is unlikely to
place two points in the same field but is still pooled rather than spatially blocked. Transfer
distance is the largest of any paper in this batch (different continent, different year, mixed
sensors), but the class problem is the simplest (two phenologically distinct crops).

## Key Findings

### Spatial transfer, Kansas 2019 to Europe 2018 (Table 4, OA percent)

Germany 93.8, France 95.4, Romania 88.1, Poland 93.3, Spain 96.8, Italy 95.0. Corn producer's
accuracy 88.3 to 91.9, user's 90.1 to 99.1; wheat producer's 86.2 to 99.1, user's 85.3 to 97.5.
Romania is the hardest country (both classes lowest), Spain the easiest.

### Temporal transfer, Kansas models from different years (Fig. 3)

OA ranges across the five training years: Germany 90.0 to 95.0, France 93.0 to 95.8, Romania
85.2 to 88.1, Poland 91.0 to 93.0, Spain 95.4 to 96.8, Italy 94.5 to 95.8; year-to-year variation
0.6 to 2.1 points. The source year matters far less than the source state.

### Effect of source state and of aggregation (Figs. 3 and 4)

Kansas is best, Minnesota second, South Dakota worst in every country. In 2019 the spread across
states is 21.5 (Germany), 9.0 (France), 15.2 (Romania), 11.1 (Poland), 1.0 (Spain) and 11.9
(Italy) OA points. Merging states, merging years, or both, is worse on average than the best
single state or year: regional aggregation costs 8.9, 1.5, 0.8, 0.6, 0.2 and 4.8 points
(Germany to Italy), temporal aggregation 7.9, 1.1, 0.2, 1.6, minus 0.4 and 0.2, full aggregation
11.2, 2.0, 1.0, 1.2, 0.4 and 5.6. Regional aggregation hurts more than temporal aggregation; the
authors interpret this as the merged source domain being less similar to the target than the best
single source.

### Transferred versus local models (Fig. 5)

The transferred model's OA is lower than the local model's by 2.0 (Germany), 1.4 (France), 1.3
(Romania), 0.5 (Poland), 0.6 (Spain) and 0.8 (Italy) points. Maps from the two models agree
closely in close-up comparisons.

### Climatic driver

Pearson correlation between OA and temperature-profile similarity (negative Euclidean distance
between monthly ERA5-Land temperature series) exceeds 0.6 with p below 0.01 in five of six
countries (Spain excepted); pooled over 90 pairs the correlation is 0.36, p below 0.01.
Regression R2 is 0.41 to 0.68. Precipitation similarity shows no significant relationship,
attributed to irrigation. Mechanistically, Kansas and the European countries share mild winters
and a May NDVI peak for wheat (winter wheat), whereas Minnesota's wheat peaks in July (spring
wheat), so temperature similarity encodes phenological synchrony.

### Deep versus tree ranking

Not reported in the main text. The supplementary deep-learning tests are described only as
showing varying transferability by model type and region, so no number can be extracted and the
paper cannot be cited for a family ranking or an inversion.

## Relevance to Our Crop-Classification Study

This paper is the tree-only, long-distance end of the transfer literature and is useful to us
mainly as a scale reference and as a source of two mechanisms, not as evidence on the
deep-versus-tree question.

**Missingness handling.** Their pipeline is the "gap-free by construction" design: cloud-masked
monthly medians from two sensors, then linear interpolation of any remaining monthly gap, applied
identically across continents. This is the standard against which our original pipeline failed:
we had a month absent for 41 percent of holdout pixels and present for all training pixels, and
only the raw-sequence deep models were handed a zero where the trees were handed a feature
computed without it. Xie et al. never confront a structurally missing month because the 2018 U.S.
year that would have had sparse Sentinel-2 coverage was simply dropped from training. Their
decision to drop a year rather than interpolate across a thin one is the conservative choice we
now approximate with median filling, and it is citable as precedent for treating completeness as
a prerequisite of transfer rather than a nuisance.

**Deep versus tree.** The paper has no usable deep baseline in the main text, so it neither
supports nor contradicts our revised ordering. What it does establish is the ceiling for a tree
model under identical complete inputs when the task is easy: 88 to 97 OA on a binary problem with
distinct winter-wheat versus summer-corn phenology, within 2 points of a locally trained model.
Our five winter crops with overlapping calendars are a different problem class, and the paper
should not be cited as evidence that trees transfer well in general; the authors' own
supplementary note that transferability "varies by model type" is the honest summary.

**Transfer distance and gap magnitude.** Cross-continent, cross-year, with a transferred-minus-
local gap of 0.5 to 2.0 OA points. Two lessons for our adjacent-tile setting. First, the choice of
source matters more than the distance in kilometres: the Kansas-versus-South-Dakota spread
(up to 21.6 OA points) dwarfs the transferred-versus-local gap. Our training tiles and holdout
tile share one growing season and climate, which by this paper's temperature-similarity logic
puts us in the favourable regime, consistent with our deep-model gaps of minus 0.06 to minus 0.14
being modest. Second, the temperature-similarity correlation (R2 0.41 to 0.68) is a label-free
predictor of transfer success; a thermal-time comparison between our tiles would be a cheap way
to characterise how hard our holdout is relative to other pairs in the literature.

**Low-data regime and learning curves.** The paper fixes training at 6,000 pixels per model and
never varies it, so it has no learning curve. It does make a point that bears on how we interpret
finding (3): adding more training data by merging heterogeneous states or years made transfer
worse, by up to 11 OA points. Our learning curves rise monotonically from about 0.55 at 10
percent of fields to about 0.65 for deep models at 100 percent, but all of our added data come
from the same two training tiles. Xie et al. show that the monotonic relationship holds only
when the added data are from the same domain; once data from a less similar domain are added,
the tree model's transfer degrades. That is consistent with the view that our deep models are
learning the shared phenology rather than tile-specific signal, and it cautions against naively
pooling additional tiles to extend the learning curve.

**Sparse gate.** Not applicable; no feature selection beyond RF's own split sampling, and the
104-feature set is used in full. Variable importance is in the supplement only.

**What to cite it for.** (i) A cross-continent RF transfer within 2 OA points of a local model on
an easy binary task, as the optimistic end of the transfer spectrum. (ii) Source-region choice
and temperature-profile similarity as the dominant driver of transfer accuracy, larger than any
model or year effect. (iii) Aggregating heterogeneous training domains can reduce transferability.
(iv) Gap-free monthly compositing with linear interpolation applied identically to source and
target as the standard preprocessing.

### Per-class detail for the Kansas 2019 transfer (Table 4, percent)

| Country | Corn PA | Corn UA | Wheat PA | Wheat UA | OA |
|---|---|---|---|---|---|
| Germany | 88.9 | 99.1 | 99.1 | 89.0 | 93.8 |
| France | 89.2 | 97.9 | 98.9 | 94.2 | 95.4 |
| Romania | 89.4 | 90.1 | 86.2 | 85.3 | 88.1 |
| Poland | 88.3 | 93.3 | 96.3 | 93.3 | 93.3 |
| Spain | 90.0 | 94.0 | 98.5 | 97.5 | 96.8 |
| Italy | 91.9 | 92.7 | 96.5 | 96.1 | 95.0 |

The error structure is consistent: corn is under-detected (PA 88 to 92) and wheat is
over-detected in the countries where wheat dominates the validation sample (Germany, France,
Poland), which is the usual majority-class pull. Romania is the only country where both classes
fall below 90 on both PA and UA. With only two classes these per-class figures are an adequate
balanced summary, but they would not be for a multi-class problem such as ours.

## Numbers at a Glance (for cross-paper comparison)

| Item | Value |
|---|---|
| Transfer setting | U.S. states (Kansas, South Dakota, Minnesota; 2019 to 2023) to six European countries (2018); cross-continent, cross-year, Landsat 8 plus Sentinel-2 |
| Granularity and input | Pixel-level, eight monthly median composites x 13 layers (104 features), binary corn versus wheat on EUCROPMAP-masked pixels |
| In-region reference ("local" RF on 70 percent of LUCAS points), OA | Transferred model lower by 2.0 (DE), 1.4 (FR), 1.3 (RO), 0.5 (PL), 0.6 (ES), 0.8 (IT) |
| Source-only transfer, Kansas 2019, OA | DE 93.8, FR 95.4, RO 88.1, PL 93.3, ES 96.8, IT 95.0 |
| Source-year sensitivity | 0.6 to 2.1 OA points across 2019 to 2023 |
| Source-state sensitivity (2019) | 21.5 (DE), 9.0 (FR), 15.2 (RO), 11.1 (PL), 1.0 (ES), 11.9 (IT); Kansas best, South Dakota worst |
| Aggregation penalty (all states and years merged) | minus 11.2, 2.0, 1.0, 1.2, 0.4, 5.6 versus best single source |
| Deep versus tree direction | Not reported in main text; supplement says transferability varies by model type |
| In-region/out-of-region inversion | Not measurable (single family) |
| Missing data | QA cloud mask, two-sensor monthly median, linear interpolation of residual gaps, gap-free by construction, identical on both continents |
| Climatic driver | Temperature-profile similarity: PCC above 0.6 in five of six countries, R2 0.41 to 0.68; precipitation not significant |
| Learning curve / sparse gate | Not studied; 6,000 training pixels per model |

The numbers to carry forward are the 0.5 to 2.0 OA gap between transferred and local RF models
on an easy binary task, and the far larger source-state effect (up to 21.5 points), which says
that domain similarity, not model family, dominated transfer in this setting.

## Evaluation Caveats

- Binary corn-versus-wheat classification restricted to pixels that EUCROPMAP already labels as
  corn or wheat; the task excludes all other land cover and all other crops, so OA in the 90s is
  not comparable to multi-class crop mapping accuracies and says little about minority-crop
  recall. No macro F1 or Kappa; PA and UA per class are given, which for two classes is
  sufficient but not a balanced summary.
- Training labels are CDL, a classification product with roughly 95 percent accuracy for these
  crops; validation uses LUCAS field points, so the transfer scores are true ground-truth
  scores, but the model is trained to reproduce another classifier.
- The local model uses a random 7:3 split of LUCAS points with no spatial blocking; given point
  sparsity this is unlikely to leak at field level but it is pooled rather than spatially
  disjoint, so the local benchmark may be mildly optimistic.
- Only one target year (2018), chosen because LUCAS 2018 is the last available survey; temporal
  transfer is tested only through the source year, not the target year.
- Only one model family in the main text. Deep-learning results exist in the supplement and are
  summarised as showing varying transferability by model and region; this silence is itself a
  caveat on any claim that the RF result generalises across families.
- Pixel-level evaluation at 30 m with no field aggregation; EUCROPMAP masking already removes
  mixed and non-crop pixels.
- Composite quality differs between the U.S. (Landsat plus Sentinel-2 from 2019) and Europe
  (2018) and between countries (6,820 to 16,093 images), but the effect of observation density on
  transfer is not analysed.
- Temperature similarity is a 15-sample correlation per country (three states x five years);
  the authors acknowledge the small sample and pool to 90 pairs for the headline p-value. Spain
  is a clear outlier with no significant correlation.
- Non-temperature drivers (sowing date, cultivar, soil, management) are discussed but not
  measured.
- Single RF run per configuration; no seeds, no confidence intervals, and the 0.5 to 2.0 point
  transferred-versus-local gaps are reported without uncertainty.
- No computational cost comparison.
