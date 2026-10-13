# Deploying Geospatial Foundation Models in the Real World: Lessons from WorldCereal

**Citation:** Christina Butsko, Kristof Van Tricht, Gabriel Tseng, Giorgia Milli, David Rolnick, Ruben Cartuyvels, Inbal Becker-Reshef, Zoltan Szantoi, and Hannah Kerner (2025). *Proceedings of TerraBytes: Towards Global Datasets and Models for Earth Observation Workshop at the 42nd International Conference on Machine Learning (ICML 2025)*. arXiv:2508.00858v1 [cs.LG], 16 Jul 2025. DOI: not found in PDF (Crossref title search on 2026-10-09 returned no matching record; the paper is cited here by its arXiv ID).
**BibTeX key (suggested):** `butsko2025worldcereal`

## Objectives

The paper has two linked aims. The first is methodological and generic: to propose a three-step
protocol for moving a geospatial foundation model (FM) from benchmark results into an
operational mapping system. The three steps are (1) *Requirements and Hypotheses* (state the
operational constraints, metrics, and what the FM is expected to buy you), (2) *Adaptation
Strategy* (how the FM must be modified to match the operational data pipeline, e.g. freeze vs.
fine-tune, extra self-supervised adaptation), and (3) *Empirical Testing* (experiments that
simulate deployment: geographic holdout, temporal holdout, label efficiency, and visual map
quality).

The second aim is a case study: applying that protocol to the ESA-funded WorldCereal global
crop-mapping system, replacing (or competing with) its deployed CatBoost classifier with the
Presto pixel-time-series foundation model (Tseng et al., 2024). Three hypotheses are stated
explicitly and tested:

- `H1`: the FM outperforms fully supervised models for both binary cropland and multiclass
  crop-type classification.
- `H2`: the FM shows significantly improved spatial and temporal generalization.
- `H3`: an extra self-supervised learning (SSL) round before fine-tuning helps adapt the FM to
  a shifted data-processing pipeline (Presto was pre-trained on Sentinel-2 `L1C` least-cloudy
  composites; WorldCereal uses cloud-free `L2A` observations).

For our manuscript the interest is almost entirely in `H2` and in how the paper operationalizes
"spatial generalization": a country-level holdout that is a genuinely spatially disjoint test,
alongside a random split and a cross-year split, all reported for the same set of models.

## Methods

**System context.** WorldCereal is an open, modular service with a Reference Data Module (RDM)
for harmonized in-situ and map-derived labels, a Processing Module that runs default or
user-retrained classifiers as openEO process graphs on the Copernicus Data Space Ecosystem,
and a visualization/dissemination module. Default products are annual cropland extent and
seasonal crop-type layers for 9 major crops at 10 m. A hard operational constraint drives model
choice: users must be able to retrain on their own labels *without GPUs*, so the model has to
be lightweight on CPU. This is the stated reason Presto was chosen (38.37M MACs for a
12-timestep series, versus 89.40M for Galileo-Nano and 889.94M for AnySat).

**Inputs.** Pixel-level time series of 18 months built from Sentinel-1, Sentinel-2 (`L2A`,
cloud-free observations only), the Copernicus GLO-30 DEM, and AgERA5 weather. This is
multi-sensor (SAR + optical + topography + meteorology), in contrast to our optical-only
setup. The deployed baseline instead uses expert-engineered features computed from the
pixel series at 10-day resolution, conceptually similar to our `xr_fresh` summary statistics.

**Reference data.**

- Cropland task: about 1.3M sampled points, 26% cropland, from 176 countries and 121 sources,
  spanning 2017 to 2021. The USA, Spain, and Belgium together account for 40% of points.
  Sources include the USDA Cropland Data Layer, LUCAS Copernicus 2018, and European farmer
  declaration (LPIS-type) datasets.
- Crop-type task: a 255K-point subset with detailed labels, collapsed to 8 classes (`maize`,
  `wheat`, `barley`, `soybeans`, `millet/sorghum`, `sunflower`, `rapeseed`, `other_crop`).
  Class shares (Fig. 8): Other 27.6%, Maize 20.1%, Wheat 16.6%, Barley 8.9%, Sunflower 7.0%,
  Rapeseed 5.9%, Soybeans 5.4%, Millet/Sorghum 1.3 to 1.6%. Geographic coverage is described
  as sparser and more skewed than the cropland set, dominated by Europe and the USA.
- Labels are *points*, not field polygons. There is no field-level aggregation and no notion
  of field IDs in the evaluation.

**Models compared.**

- `Deployed baseline` (cropland only): the operational WorldCereal CatBoost classifier, trained
  *separately per continent* on expert-defined features.
- `Unprocessed CatBoost`: the same CatBoost model trained directly on the raw pixel time series
  (no engineered features). This is the only supervised baseline for crop type, because no
  deployed crop-type baseline existed.
- `Finetuned Presto-Rnd`: the Presto transformer architecture randomly initialized and trained
  end-to-end. This is the key ablation: it separates the value of pre-training from the value
  of the architecture.
- `Finetuned Presto`: pre-trained Presto fine-tuned on the labels.
- `SSL + Finetuned Presto`: an extra SSL round on the (larger) cropland dataset, then
  fine-tuning.

No hyperparameter search, seed count, training budget, or CatBoost configuration is reported.

**Evaluation protocol:** Three splits, each with full retraining, all evaluated at the *point
(pixel)* level. (a) **Random split:** an 80/20 random train/validation split over all points,
pooled globally; this is the in-distribution reference and is the protocol most crop papers
report as "test" accuracy. Whether multiple points can come from the same field or from
adjacent pixels is not stated, so field-level leakage in the random split cannot be ruled out.
(b) **Geographic split:** entire countries are removed from training and validation and used
only for testing. Cropland holdout: Spain, Nigeria, Latvia, Tanzania, Ethiopia, Argentina.
Crop-type holdout: Spain, Latvia, Austria, Brazil, Tanzania, Ethiopia, Madagascar, Mozambique,
Morocco, Indonesia, "supplemented by several smaller-label European countries" (Italy and
Greece appear in the per-country table). This is a true spatially disjoint holdout at country
scale, a harsher and more heterogeneous shift than our adjacent-tile holdout, and it has zero
target labels in training, i.e. domain generalization rather than domain adaptation (Presto's
pre-training corpus is global, though, so the FM has seen unlabeled imagery from the held-out
regions). (c) **Temporal split:** all 2021 samples held out and used as the test set, i.e. a
cross-year test (cropland labels from 2021 are 16.4% of points; crop-type 2021 labels are
25.7%). There is no combined space-and-time holdout. Metrics: "Crop F1" for the binary task
and macro-F1 for crop type, plus per-class F1 (random split only) and per-country F1
(geographic split). Map quality is judged visually on patches from models trained on the
*random* split. On the spectrum this manuscript uses, the paper sits at the strong end: it
reports pooled random, spatially disjoint (country), and cross-year results side by side for
the same models, which is exactly the contrast our study argues for.

**Africa / South Africa coverage.** South Africa is not among the held-out countries in either
task. Held-out African countries are Nigeria, Tanzania, and Ethiopia (cropland) and Tanzania,
Ethiopia, Madagascar, Mozambique, and Morocco (crop type). The label-density choropleths
(Figs. 4 and 5) show southern Africa at the lightest shade, so any South African labels in
training are sparse; the exact count cannot be read from the figure. No South Africa-specific
result is reported. The paper notes the general scarcity of crop-type labels in Sub-Saharan
Africa (Nakalembe and Kerner, 2023) as a motivation for FMs.

## Key Findings

**Binary cropland, overall F1 (Table 1):**

| Model | Random | Geographic | Temporal | Random to Geo drop |
|---|---|---|---|---|
| `Deployed baseline` (CatBoost, expert features) | 0.856 | 0.810 | 0.830 | -0.046 |
| `Unprocessed CatBoost` (raw series) | 0.828 | 0.777 | 0.874 | -0.051 |
| `Finetuned Presto-Rnd` | 0.810 | 0.705 | 0.806 | -0.105 |
| `Finetuned Presto` | 0.861 | 0.829 | 0.886 | -0.032 |
| `SSL + Finetuned Presto` | 0.861 | 0.826 | 0.884 | -0.035 |

(Drop column computed by us from the paper's table.)

**Binary cropland, per-country Crop F1 under the geographic split (Table 2):** Pre-trained
Presto is best or near-best in every held-out country (Argentina 0.910, Ethiopia 0.683 to 0.732
with SSL, Latvia 0.882, Nigeria 0.872, Spain 0.748, Tanzania 0.442). The deployed CatBoost
baseline is competitive in Nigeria (0.857) and Ethiopia (0.692). The African holdouts are the
hardest: Tanzania is 0.26 to 0.44 for all models, with Unprocessed CatBoost worst (0.258).
Randomly initialized Presto is the weakest model in most countries (Ethiopia 0.511, Latvia
0.749, Spain 0.650).

**Multiclass crop type, overall macro-F1 (Table 3):**

| Model | Random | Geographic | Temporal | Random to Geo drop |
|---|---|---|---|---|
| `Unprocessed CatBoost` | 0.728 | 0.563 | 0.649 | -0.165 |
| `Finetuned Presto-Rnd` | 0.782 | 0.620 | 0.646 | -0.162 |
| `Finetuned Presto` | 0.809 | 0.650 | 0.686 | -0.159 |
| `SSL + Finetuned Presto` | 0.820 | 0.645 | 0.674 | -0.175 |

(Drop column computed by us.)

**Crop type, per-country macro-F1 under the geographic split (Table 4)** (CatBoost vs.
Finetuned Presto): Austria 0.519 vs 0.605, Spain 0.400 vs 0.516, Brazil 0.563 vs 0.745,
Italy 0.614 vs 0.623, Madagascar 0.432 vs 0.479, Mozambique 0.397 vs 0.437, Ethiopia 0.492 vs
0.559, Greece 0.581 vs 0.663, Morocco 0.228 vs 0.384. Pre-trained Presto beats CatBoost in all
nine reported countries, but the margins in African countries are small and unstable across
variants: in Mozambique, Presto-Rnd is 0.110 *below* CatBoost and SSL + Presto is 0.001 below.
Note that Latvia, Tanzania, and Indonesia (named as held out) do not appear in Table 4, and
validation sample sizes are tiny for most countries (Ethiopia, Greece, Morocco about 0.2K
points each).

**Crop type, per-class F1 under the random split (Table 5):** Presto's gains are largest on
the rarest class, `millet/sorghum` (CatBoost 0.415 to Finetuned Presto 0.530, SSL + Presto
0.563, with only 0.1K validation points). Majority classes gain 0.02 to 0.06.

**Hypotheses.** The authors conclude `H1` and `H2` are supported (pre-trained Presto wins on
all three splits for both tasks) and `H3` is not (the SSL adaptation round gives no
consistent benefit, and in one USA patch it produces visibly worse maps). Pre-training, not
the architecture, carries the cropland result, since randomly initialized Presto is the worst
cropland model. Visual patches show cleaner field boundaries for pre-trained Presto.

**Lessons stated.** Task-specific alignment matters, but further domain adaptation gives only
modest gains, so choose an FM whose pre-training inputs already match your pipeline;
pre-trained models distill transferable patterns; benchmarks miss deployment needs (geographic
and temporal generalization plus map quality); compute efficiency matters; compatibility with
legacy pipelines eases adoption.

## Relevance to Our Crop-Classification Study

This is a new and directly relevant true-spatial-holdout comparator. It evaluates the same
models under a pooled random split, a country-level spatially disjoint holdout, and a
cross-year holdout, and it reports macro-F1 and per-class F1, which is the evaluation
discipline our manuscript argues for. It is also one of the few papers in the corpus that pits
a gradient-boosted-tree baseline (`CatBoost`) against a transformer under a geographic
holdout, and it includes the decisive ablation (pre-trained vs. randomly initialized
transformer) that lets us separate pre-training from architecture.

Read carefully, the results are more consistent with our thesis than the headline suggests.

1. **A from-scratch transformer transfers worst for cropland.** `Finetuned Presto-Rnd` drops
   0.105 F1 from random to geographic, twice the drop of either CatBoost model, and becomes
   the worst cropland model under transfer (0.705, vs. 0.810 for the engineered-feature
   CatBoost). This is the same pattern we observe: a dense temporal network trained from
   scratch loses the most when moved to new terrain, while tree models on engineered
   features are robust. The engineered-feature deployed baseline also degrades less than raw
   `Unprocessed CatBoost` under the geographic split (0.810 vs 0.777), which echoes our
   `xr_fresh` vs. raw comparison, although the ordering flips on the temporal split (0.830 vs
   0.874).
2. **Pre-training is what rescues the transformer.** Pre-trained Presto is the only
   transformer variant that beats the GBDT under geographic transfer in the cropland task.
   This aligns with how we already read Cropformer (`cropformer2023`): dense temporal models
   can transfer when given a strong pre-training prior. Our dense models had no such prior.
   WorldCereal is therefore the strongest available evidence for the "pre-training as the
   missing ingredient" framing in our Discussion and future work, and it is the natural
   citation for that point.
3. **The crop-type task does not show ranking inversion.** For crop type, the random-to-
   geographic drop is nearly identical for all four models (about 0.16 to 0.18 macro-F1),
   and the ranking is preserved; even randomly initialized Presto beats raw CatBoost under
   transfer (0.620 vs 0.563). This is a counterpoint to our finding that the dense cluster
   fractures under transfer. Plausible reasons: the baseline here is CatBoost on *raw* series
   rather than on engineered features (the expert-feature baseline was not run for crop
   type); the training pool is global and multi-sensor (S1 + S2 + DEM + weather), so the
   holdout country is less out-of-distribution relative to training than our single-region
   training is to an adjacent tile; and with 255K labels the transformer is in a data regime
   our 2-tile training set is not. We should cite this honestly as a case where, with large
   global multi-sensor training data, a transformer did not lose ground to a GBDT under
   country-level transfer.
4. **Evaluation framing.** The paper's own protocol (Section 2.3) recommends exactly the
   geographic and temporal holdouts we use, and its lesson "evaluating beyond benchmarks"
   supports our central claim that in-distribution validation is insufficient for model
   selection. It is a high-profile, operational-system citation (ESA, NASA Harvest, Mila
   co-authors) for the statement that operational maps require spatial-holdout evaluation.
5. **South Africa and Africa.** South Africa is not tested. African holdouts (Tanzania,
   Ethiopia, Mozambique, Madagascar, Morocco, Nigeria) are consistently the hardest, with
   crop-type macro-F1 of 0.23 to 0.63 and cropland F1 as low as 0.26 to 0.44 in Tanzania. This
   supports our motivation that African mapping is the transfer-limited case, and our ST
   macro-F1 of about 0.60 for the best models is in the same range as WorldCereal's held-out
   African crop-type scores (though tasks, classes, and units of evaluation differ, so this is
   context, not a comparison).
6. **Features and granularity.** Point/pixel-level only, no field aggregation; we find field
   aggregation is a substitutable robustness lever, which this paper does not test. Its
   inputs are multi-sensor whereas ours are optical-only, so the absolute numbers are not
   comparable.

**Recommended uses in the manuscript.** Cite in the Introduction alongside
`pankajakshan2025`, `jin2019`, and `rustowicz2019` as a study that evaluates under a spatially
disjoint (country) holdout; cite in the transfer-learning sentence alongside `cropformer2023`
and `mai2025` as a pre-training/foundation-model approach that uses target-domain unlabeled
data (Presto's global pre-training) and therefore is not the no-target-access setting we
study; and cite in the Discussion for the Presto-Rnd vs. Presto contrast, which shows that a
dense transformer without a pre-trained prior transfers worst for cropland, consistent with
our dense-net collapse. Because it materially adds a true-spatial-holdout comparator,
`0_findings_summary.md` should be refreshed via `brief-synthesize-crop-findings`.

## Evaluation Caveats

- **Venue and status.** ICML 2025 workshop paper posted on arXiv; not a peer-reviewed journal
  article, and no DOI. Several authors developed Presto (Tseng, Rolnick, Cartuyvels, Kerner)
  and WorldCereal (Van Tricht), so the comparison is run by the method's developers.
- **Single run, no uncertainty.** No seeds, confidence intervals, or significance tests are
  reported, yet `H2` claims "significantly improved" generalization. Many per-country
  differences (e.g. Italy +0.009, Mozambique +0.040, Ethiopia cropland -0.009 for Presto vs.
  deployed baseline) are well within what a single random seed could produce, particularly
  with 0.2K to 0.5K validation points per country.
- **Possible leakage in the random split.** Points are sampled from label sources that
  include LPIS parcels and the CDL; whether points from the same parcel can land in both train
  and validation of the 80/20 split is not stated. The random-split numbers should be treated
  as in-distribution fit, not out-of-sample accuracy. The geographic split is immune to this
  by construction.
- **Baseline asymmetry.** The strongest supervised baseline (expert features, per-continent
  CatBoost) exists only for cropland. For crop type, the only GBDT is CatBoost on raw time
  series, which is the weakest possible tree configuration, so "FM beats GBDT under transfer"
  for crop type is a comparison against an under-engineered tree model. No CatBoost tuning is
  described. The per-continent training of the deployed baseline also means it was trained on
  less data per model than Presto.
- **Pre-training exposure.** Presto's self-supervised pre-training corpus is globally sampled,
  so it has seen unlabeled imagery from the held-out countries. The geographic split holds out
  *labels*, not *imagery*; this is not the strict no-target-access setting of our study.
- **Imbalance.** Macro-F1 is reported for crop type, which is good practice, but the binary
  task reports only "Crop F1" (positive-class F1), and per-class F1 is shown only for the
  random split. Per-class performance under the geographic split, where minority-class
  collapse would be most informative, is not reported.
- **Incomplete tables.** Latvia, Tanzania, and Indonesia are listed as crop-type holdouts but
  are missing from Table 4, while Italy and Greece appear without being listed; how the
  overall geographic macro-F1 is aggregated across countries (pooled points vs. mean of
  countries) is not stated.
- **Visual evaluation.** All qualitative map patches come from models trained on the random
  split, so they say nothing about map quality under geographic or temporal transfer.
- **What is not measured.** No combined spatial-and-temporal holdout; no label-efficiency
  experiment despite recommending one in the protocol; no field-level or object-level
  evaluation; no comparison to other FMs (Galileo, AnySat, Prithvi) or to lighter
  non-pretrained temporal models (TempCNN, L-TAE) or sparse/feature-selecting deep models
  such as TabNet; compute cost is discussed only as forward-pass MACs, with no measured
  training or inference times.
- **Bottom line for citation.** Cite the country-holdout protocol and the Presto-Rnd vs.
  Presto contrast, not the headline "FM beats conventional methods." Their 0.809 random-split
  crop-type macro-F1 comes from a pooled global 80/20 point split and is not comparable to
  our spatially disjoint holdout macro-F1; their 0.650 geographic-split macro-F1 is the
  number that corresponds in kind to ours, though it is computed over different classes,
  sensors, and point-level units.
