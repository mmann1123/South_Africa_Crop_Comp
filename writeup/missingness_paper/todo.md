# Missingness-shift paper: to-do

Branch `missingness-shift`. Background: `experiments/missingness/README.md` and
`reports/Missing data robustness novelty check.md` ("Ways to push this forward"), and the
literature review of missingness in transfer papers, `reports/Missingness confound audit of
transfer papers.md` (notes in `research_notes/Missingness confound audit of transfer papers/`).
Tick items
with the date and the files or commits that close them.

## Withdrawal of the CEA submission

- [x] Email co-authors (draft agreed in the session of 2026-10-21) and agree on withdrawal.
- [x] Contact the editor (Fernando Auat Cheein) before the 2026-11-01 deadline: withdraw, or ask
  whether a substantially reframed manuscript can be submitted as a new paper.
- [x] Freeze the CEA version: git tags `cea-submitted`, `cea-revision1-final` (2026-10-21).

## Needed for the paper

- [x] **Month-dropout control for the sparse gate** (2026-10-23;
  `experiments/missingness/month_dropout/`). 2 x 2 dropout x gate on TempCNN, CNN-BiLSTM and L-TAE.
  Under zero fill dropout restores every backbone (+0.085 to +0.162, all significant) and the gate
  adds nothing on top (interaction −0.088 CNN-BiLSTM, −0.061 L-TAE, significant). Under the
  training-median fill nothing differs; dense L-TAE + dropout 0.664 is the best single model.
  The gate is a partial substitute for training-time exposure to missing months.
- [x] **Imputation-free check** (2026-10-15): published predictions scored on the 1,352
  September-observed vs 1,063 September-missing fields
  (`experiments/holdout_fill_check/results/sept_observed_subset.csv`).
  - [x] Bootstrap intervals per subset (2026-10-21; `experiments/missingness/subset_bootstrap.py`,
    `results/subset_bootstrap.csv`). Published all-field ranking correlates 0.86 with the
    September-missing ranking and 0.33 with the September-observed one; observed vs missing −0.08.
    Changed-field counts per fill are in `fill_sweep/results/zero_vs_median.csv`.
  - [ ] Put it in the main text next to the before/after ranking.
- [x] **Per-model fill sheet** for the supplement (2026-10-21; `experiments/missingness/fill_sheet.py`,
  `results/fill_sheet.md` and `.csv`). One row per model: fill value
  used, source of the fill statistic (training vs holdout), fill before or after scaling,
  retrained vs re-scored, and whether the re-run reproduces the published predictions. Sources:
  `experiments/holdout_fill_check/README.md`, `results/fill_comparison.csv`. Template:
  Kapoor & Narayanan's model info sheet.
- [ ] **Name and document the missingness shift** (no compute).
  - [x] Per-month missing-data table (2026-10-21; `experiments/missingness/missing_data_table.py`,
    `results/missing_data_by_month.csv`). Holdout September: 41.5% of pixels (B2), 40.2% of
    fields fully and 44.0% partly missing; 0.96% of pixels miss only some bands; training tiles
    miss nothing but May (missing everywhere). Use 40.2% / 44.0% of fields, not the 39.3% quoted
    earlier (that required all six bands missing).
  - [ ] State the fill rule in the Data section.
  - [x] Overlay the September gap on the holdout error map (2026-10-23;
    `experiments/missingness/sept_gap_map.py`, `figures/sept_gap_error_map.pdf`). Confirmed: the
    gap covers 76% of the 8 to 16 km zone's fields (26% elsewhere); September-observed fields in
    the zone score like anywhere else; the published distance effect (about −0.31 log-odds per
    10 km for L-TAE, CNN-BiLSTM, TempCNN) vanishes once September share is in the logit.
- [x] **Fill-value sweep** (2026-10-21; `experiments/missingness/fill_sweep/`, README there).
  25 models x 5 fills, no retraining (about 4 h, not the 1 h first estimated). In-range fills are
  interchangeable (median per-model spread 0.014; ranking Spearman 0.89 to 0.97 with training
  median); zero fill hurts every family including the classical models (mean −0.075 raw, −0.096
  xr_fresh); the published ranking flip came from the mixed treatment (Spearman 0.56 vs 0.74 when
  everyone is zero-filled). Also fixed `interp` to skip months never observed in training.
- [x] **Step deletion** for L-TAE and the Transformer (2026-10-24; `experiments/missingness/step_deletion.py`,
  inference only, 10 minutes). Masking the missing month out of attention equals the
  training-median fill for L-TAE, L-TAE-S, the Transformer and the month-dropout L-TAEs (all
  within ±0.008, no CI excludes zero) and beats zero fill by +0.04 to +0.11. The retraining
  variant (train with masked steps) is now low priority: inference-time deletion already matches
  the best fill and month dropout.

### Generality (added 2026-10-28, after the novelty-report review)

A reviewer's first objection will be one dataset, one tile, one missing month: a preprocessing bug
found in our own pipeline. These three answer it.

- [x] **Month-by-month missing-month injection** (2026-10-28; `experiments/missingness/month_injection.py`,
  13 models, holdout and in-region). Zero fill damages every month for every model trained
  without dropout, in-region as much as on the holdout (mean −0.08 to −0.35, worst month −0.19 to
  −0.60), so the failure is neither September- nor transfer-specific. Month dropout removes it in
  every month. Median fill and deletion agree on the real information loss: under 0.03 for most
  months, 0.04 to 0.16 for October.
- [ ] **Second dataset** (retraining; scope depends on the target journal). BreizhCrops or the
  Germany track of AI4FoodSecurity, with its natural cloud gaps or simulated ones. Turns the
  paper from a case study into a general finding.
- [x] **Paper-by-paper audit of published transfer failures** (steps 1, 2, 4 done 2026-10-28;
  `research_notes/Transfer failure missingness audit/summary.md`, details in part1.md, part2.md).
  14 papers: none "likely", none fills a target gap with an out-of-distribution constant. Six
  "possible" through differing observation patterns (Sen4AgriNet strongest, TimeMatch, T3S,
  Orynbaikyzy, Chen & Liu, Valinho), each also confounded; the rest unlikely. Our mechanism
  appears partially in the Togo crop-mask code (raw 0 before the normaliser at inference, no
  missing-share cap), checked, but that paper makes no transfer claim. Side finding: Sen4AgriNet's
  public object-level loader may train on its test file (unverified for the published runs).
  - [ ] Step 3 (later): compare missingness between source and target in the public data for
    Sen4AgriNet, TimeMatch/T3S and Orynbaikyzy (checks listed in summary.md).
- [x] **Literature review: missingness confound in transfer papers** (user's separate report,
  `reports/Missingness confound audit of transfer papers.md`). 45 papers plus 13 benchmarks,
  ranked by a five-condition rubric: 14 rank 3 ("highly suspect"), 19 rank 2, 12 rank 1. Only 5 of
  about 33 papers reporting a transfer loss publish a per-domain valid-observation statistic.
  Benchmarks carry the confound (TimeSen2Crop zero-fills months, Sen4AgriNet interpolates,
  CropHarvest uses the pixel mean, Planet Fusion hides synthesised days). Granule cloud cover for
  our 2017 tiles is near identical, so the September shift is sub-tile.
  - [ ] Reconcile with our 14-paper audit: the review ranks TimeMatch, Wijesingha, Sen4AgriNet and
    Kerner (GSN, NaN to 0 in code) higher than `summary.md` does.
  - [ ] Reconcile the September figure: the review quotes 40.6% of holdout pixels, our table says
    41.5% (B2) and 40.2% / 44.0% of fields.
  - [ ] **Prior art to engage:** Pham et al. 2024 (IJAEOG 129, 103867; and RSE 308, 114206) zero-fill
    cloudy days, show cross-year transfer fails with observation density, and fix it with Random
    Observation Selection (training-time observation dropout). That is close to our month-dropout
    result, it is not in the novelty check, and it must be briefed and cited. Our contribution
    over it: spatial (not cross-year) transfer, the model-ranking reversal, fill value vs
    information loss (step deletion), the sparse-gate comparison.
- [x] **Background: nodata handling in general-purpose SITS codebases** (2026-10-28;
  `research_notes/SITS nodata handling audit/audit.md`, 17 codebases, commit-pinned links). Result
  does NOT support "probably common in public code": none reproduces our failure mode. 4 mask
  consistently (WorldCereal also simulates gaps in training), 3 interpolate identically in train
  and predict (R `sits`, eo-learn, Sen4AgriNet), 3 composite gaps away upstream (NASA Harvest),
  5 do no gap handling (irregular sequences; clouds left raw), TSViT zero-pads consistently.
  Closest case: OpenMapFlow/Presto fill an all-NaN tile band with raw 0 before normalising (edge
  path). Claim to make instead: reference pipelines protect against this through masking,
  interpolation or compositing; hand-built fixed monthly grids that reuse reference
  architectures without that data handling (like ours) fall outside that protection. Only 6 of
  17 come from two groups, so the sample is not independent. MCTNet and Che et al. code not
  checked. OpenMapFlow `_fillna` spot-checked against the pinned commit (confirmed); spot-check any
  other line before quoting it.

### Caveats to fix (added 2026-10-28)

- [x] **Learning curves, 100% points** (2026-10-28; `learning_curves/single_seed_100.py`). Each
  seed scored singly; the seed average reproduces every saved ensemble. Single models sit 0.01 to
  0.04 below their ensembles (largest for CNN-BiLSTM and TabNet); ensembles shown as open markers.
- [ ] **Learning curves, paired comparisons at 5 to 15%** (about 8 to 10 GPU hours; decide once
  the low-data result's prominence is settled). Rerun 5/10/15% with one shared train/validation
  split and identical field draws for every model, so draws and fields pair across models. Needed
  if the paper claims logistic regression matches the deep models at low data (a null claim from
  the current unpaired tests is weak).
- [x] **Mechanism figure** (inference only). Flip-rate bar chart done
  (`experiments/missingness/flip_rate_figure.py`, `figures/flip_rate.pdf`). The leave-one-month-out
  occlusion is the training-median arm of the month-by-month injection (fields changed and F1 per
  month; done). Integrated-gradients figure goes to the supplement.

## Would strengthen it

- [x] **Gradient attribution at the September slot** (2026-10-23; `experiments/missingness/attribution.py`,
  about 10 GPU hours, not 1: the gated models need up to 2,048 IG steps). A zero-filled September
  draws 1.5 to 2.6x the attribution of a real one; the fill flips 37% (TempCNN) and 64%
  (CNN-BiLSTM) of dense pixel predictions, 9 to 15% with month dropout. IG share itself is
  confounded by distance from the baseline; flip rate is the robust measure.
  - [ ] L-TAE / Transformer attention (Rußwurm & Körner 2020; Obadić) not done.
- [ ] **Missing-data indicator channel, or step deletion with day-of-year encoding** (about 10 to
  20 GPU hours). A "no imputation" row and a direct test of Zhou 2023 Proposition 1. The
  indicator arm can share runs with the month-dropout control.
- [x] **Tree + deep ensemble on the filled holdout** (2026-10-23; `experiments/missingness/ensemble.py`).
  Training-median fill: mixed-10 vote 0.661 = deep-5 0.660 (n.s.); trees add nothing. Zero fill:
  mixed-10 0.603 beats deep-5 and tree-5 by about +0.03 (significant), a hedge against the
  mismatch but weaker than month dropout (0.65).
- [x] **Learning curves down to 5%, plus TabNet at 10 to 15%** (2026-10-23, about 12 GPU hours;
  `experiments/missingness/learning_curves/`). At 5 to 15% raw-sequence models and logistic
  regression are indistinguishable (about 0.50 to 0.59); XGBoost and TabNet trail by 0.04 to 0.10
  (significant). Deep models separate from about 25%. No low-data gain from sparsity.
- [ ] **Sparsity Normalization at September** (cheap; Yi 2020). Closest precedent found no gain.

## Loose ends from the audit

- [ ] Re-run the patch CNNs (3D, multi-channel 2D) with a consistent fill; their patches come
  from a separate pipeline (`out_of_sample/create_test_patches.py`) and still use zero fill.
- [x] Fix or document the xr_fresh `autocorr_lag_k` bug (three features are copies of
  `absolute_sum_of_changes`; `light_learn/2_SA_xr_fresh_extraction.py`). Documented at the source
  (2026-10-23; the key should be `autocorrelation`); not fixed, to keep the published features
  reproducible. 174 features = 6 bands x 29 statistics; the 18 `autocorr_lag_k` columns duplicate
  `absolute_sum_of_changes`, so 156 are distinct. The new paper must say so, and must not list
  autocorrelation among the features (the CEA draft's Sec. 3 does, `cea-article.tex:85`).
- [ ] Note the pixel TabNet checkpoint mismatch (published predictions predate the saved
  checkpoints; 221 of 2,415 fields differ, 0.603 vs 0.609). Recorded in
  `experiments/holdout_fill_check/README.md`; still to state in the paper's methods.
- [ ] Do not carry over the Belgiu 2016 and Obadić 2022 citations as used in the CEA draft
  (both mis-cited; see `writeup/literature/0_findings_summary.md`, Section 6).
- [ ] Add the new papers to `missingness.bib` and to `writeup/literature/references.bib`
  (`briefer-bibliography` skill).
- [ ] Get and brief BreizhCrops (Rußwurm et al. 2020) and Hoppe et al. 2024; regenerate the
  Mask-PSTIN briefer from the full text.

## Writing

- [ ] Choose the target journal.
- [ ] Write the manuscript (`article.tex`) around the missingness-shift finding.
- [ ] Supplement, figures, data and code availability, generative-AI declaration.

## Future work (discussion only)

- Source-free adaptation or masked pretraining on the unlabelled holdout pixels (Mohammadi 2024,
  SITS-BERT, Presto).
- SAR fusion and multi-year source training.
