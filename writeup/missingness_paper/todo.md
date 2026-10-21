# Missingness-shift paper: to-do

Branch `missingness-shift`. Background: `experiments/missingness/README.md` and
`reports/Missing data robustness novelty check.md` ("Ways to push this forward"). Tick items
with the date and the files or commits that close them.

## Withdrawal of the CEA submission

- [ ] Email co-authors (draft agreed in the session of 2026-10-21) and agree on withdrawal.
- [ ] Contact the editor (Fernando Auat Cheein) before the 2026-11-01 deadline: withdraw, or ask
  whether a substantially reframed manuscript can be submitted as a new paper.
- [x] Freeze the CEA version: git tags `cea-submitted`, `cea-revision1-final` (2026-10-21).

## Needed for the paper

- [ ] **Month-dropout control for the sparse gate** (retraining, about 15 to 30 GPU hours).
  Retrain CNN-BiLSTM, TempCNN and L-TAE with random month dropout; ideally a 2 x 2 of dropout x
  gate. Score on the zero-filled, training-median-filled and published holdouts. Builds on
  `experiments/sparse_gate_backbones/train.py` and `experiments/field_reduction/train_ltae_pixel.py`.
  Any outcome is reportable: dropout matches the gate (first measurement of the standard remedy
  under train/test missingness mismatch), the gate wins (it has a niche), or they add.
  Precedent: Garnot 2022, Mask-PSTIN 2024, Wang/Belgiu 2025, Zhou 2023 Algorithm 1.
- [x] **Imputation-free check** (2026-10-15): published predictions scored on the 1,352
  September-observed vs 1,063 September-missing fields
  (`experiments/holdout_fill_check/results/sept_observed_subset.csv`).
  - [ ] Add paired bootstrap intervals per subset and the number of fields that change class.
  - [ ] Put it in the main text next to the before/after ranking.
- [ ] **Per-model fill sheet** for the supplement (no compute). One row per model: fill value
  used, source of the fill statistic (training vs holdout), fill before or after scaling,
  retrained vs re-scored, and whether the re-run reproduces the published predictions. Sources:
  `experiments/holdout_fill_check/README.md`, `results/fill_comparison.csv`. Template:
  Kapoor & Narayanan's model info sheet.
- [ ] **Name and document the missingness shift** (no compute).
  - [ ] Per-month missing-data table, training tiles vs holdout, pixel and field level.
  - [ ] State the fill rule in the Data section.
  - [ ] Overlay the September gap on the holdout error map (the old Sec. 4.3 "south-central
    error zone" is probably its footprint); error maps in `out_of_sample/scoring_results/` and
    `experiments/distance_decay/`.
- [x] **Fill-value sweep** (2026-10-21; `experiments/missingness/fill_sweep/`, README there).
  25 models x 5 fills, no retraining (about 4 h, not the 1 h first estimated). In-range fills are
  interchangeable (median per-model spread 0.014; ranking Spearman 0.89 to 0.97 with training
  median); zero fill hurts every family including the classical models (mean −0.075 raw, −0.096
  xr_fresh); the published ranking flip came from the mixed treatment (Spearman 0.56 vs 0.74 when
  everyone is zero-filled). Also fixed `interp` to skip months never observed in training.
- [ ] **Step deletion** for L-TAE and the Transformer (attention mask over the missing month), as
  a "no imputation" row next to the fill sweep.

## Would strengthen it

- [ ] **Gradient attribution at the September slot** (under 1 hour): dy/dX share per month under
  zero fill vs training-median fill, plus L-TAE / Transformer attention (Rußwurm & Körner 2020;
  Obadić). Mechanism figure.
- [ ] **Missing-data indicator channel, or step deletion with day-of-year encoding** (about 10 to
  20 GPU hours). A "no imputation" row and a direct test of Zhou 2023 Proposition 1. The
  indicator arm can share runs with the month-dropout control.
- [ ] **Tree + deep ensemble on the filled holdout** (under 1 hour): voting/stacking over
  `experiments/holdout_fill_check/predictions/` (Shwartz-Ziv & Armon).
- [ ] **Learning curves down to 5%, plus TabNet at 10 to 15%** (about 10 to 20 GPU hours) via
  `experiments/holdout_fill_check/run_sweep.py`. Locates the floor; 15 fields per class is noisy.
- [ ] **Sparsity Normalization at September** (cheap; Yi 2020). Closest precedent found no gain.

## Loose ends from the audit

- [ ] Re-run the patch CNNs (3D, multi-channel 2D) with a consistent fill; their patches come
  from a separate pipeline (`out_of_sample/create_test_patches.py`) and still use zero fill.
- [ ] Fix or document the xr_fresh `autocorr_lag_k` bug (three features are copies of
  `absolute_sum_of_changes`; `light_learn/2_SA_xr_fresh_extraction.py`).
- [ ] Note the pixel TabNet checkpoint mismatch (published predictions predate the saved
  checkpoints; 221 of 2,415 fields differ, 0.603 vs 0.609).
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
