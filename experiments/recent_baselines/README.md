# Recent time-series baselines (reviewer R2-Q3a)

Reviewer 2 asked for one or two recent time-series classifiers run under the same
protocol as the paper. This folder adds two, chosen after the literature review
(`reports/Cross region crop mapping literature.md`):

- **MiniRocket** (Dempster et al., KDD 2021). It produces about 10,000 random
  convolution features with no feature selection and passes them to a linear head. In
  the bake-off redux (Middlehurst et al., DMKD 2024) it is in the top family of time
  series classifiers. It tests the paper's thesis directly: dense features, no learned
  selection.
- **TimesNet** (Wu et al., ICLR 2023). A high-capacity modern architecture, and the first
  model the reviewer named.

Presto was considered and deferred. A fair run would need Sentinel-2 bands we have not
extracted (B4, B8 and others) and re-processing of the holdout tile.

Results are written up separately in `RESULTS.md`. Nothing here edits the manuscript, the
SI, `out_of_sample/` or any published model.

## Fair comparison

Both models are trained with exactly the inputs and protocol of L-TAE-S
(`experiments/field_reduction/train_ltae_sparse_{field,pixel}.py`), reusing its helpers:

| Step | Setting |
|---|---|
| Inputs | Same 60 features (10 months: Jan–Apr, Jul–Dec × B2, B6, B11, B12, EVI, hue), from `get_chrono_feature_cols`. May is all-null and June is absent, as for every published deep model. |
| Split | FID-wise 64/16/20 (`get_fid_split_dl`, random_state 42): 2653 / 664 / 830 fields. |
| Scaling | StandardScaler fitted on training rows. |
| Field level | Pixels averaged per field before the split, as published. |
| Seeds | 42, 101, 202, 303, 404. |
| Holdout | Field: average the test pixels per field, then predict. Pixel: predict each pixel, then majority vote per field. |
| Scoring | `experiments/field_reduction/score_oos.score` (Table 1 methodology, official hard-label Xent). |
| Model selection | Validation split only. No hyperparameter was tuned on the holdout. |

`common.check_against_published` asserts that the run sees exactly the published
L-TAE-S inputs:
- the same feature columns and label classes
- the same field counts and (for the pixel model) pixel counts
- a fitted scaler whose mean and scale match the published `scaler.joblib` to within 1e-12

Both levels pass in the new environment.

**TimesNet** reuses the L-TAE-S training loop unchanged:
- focal loss (γ = 2, α = 1/count)
- AdamW (lr 1e-3, weight decay 1e-4), cosine schedule
- early stopping at patience 15 on validation macro-F1
- AMP and gradient clipping
- batch 128 (field) or 2048 (pixel)
- 5-seed mean logits

The architecture is TSLib's classification model with defaults scaled to a 10-step
series: d_model 32, d_ff 64, e_layers 2, top_k 2, num_kernels 6, dropout 0.1. These were
fixed before training.

**MiniRocket** uses aeon 1.6.0 with 10,000 kernels and `random_state` = seed:
- **Features:** the MiniRocket features are standardised.
- **Head:** multinomial logistic regression with balanced class weights, the analogue of
  the focal α. A ridge head would give no probabilities.
- **C:** chosen on validation macro-F1 from {1e-4, 1e-3, 0.01, 0.1, 1, 10}. If the best C
  is the smallest tried, a decade below is added, repeating down to 1e-8.
- **Ensembling:** 5-seed mean probabilities.
- **Solver:** the head is fitted by full-batch L-BFGS in PyTorch. It minimises
  scikit-learn's LogisticRegression objective exactly, but scikit-learn took about 9 min
  per fit at this width. `check_head_equivalence.py` compares the two on field-level
  features (`results/head_equivalence.csv`): validation predictions agree on 99.7–100% of
  fields at every C, and the PyTorch head is 20–60× faster. It converges in 343–588
  L-BFGS iterations, so the cap is 1,000, and each fit's convergence is recorded in
  `seed_<s>.json`.

Known differences, disclosed:

1. **Pixel-level MiniRocket** trains on at most 50 pixels per training field, drawn with
   the seed: about 130k of 4.8M pixels. All pixels would need about 190 GB of features.
   Validation, in-region test and holdout use every pixel. L-TAE-S pixel trained on all
   pixels.
2. **TimesNet** finds its periods with an FFT of a 10-step series, so it has at most 5
   non-zero frequencies and its period search is close to degenerate. It also picks
   periods from the batch-mean spectrum, so a sample's output depends on its batch.
   Evaluation therefore uses a fixed batch size and order.
3. **C grid changed after a first run (post hoc).** The first field run used
   C ∈ {0.01, 0.1, 1, 10}. All 5 seeds chose C = 0.01, and validation F1 was still rising
   at that edge, which suggested under-regularisation. We extended the grid and added the
   lower-edge rule above, then reran both levels. This change was made after the first
   run's holdout score (macro-F1 0.46) had been seen. Selection still uses only the
   validation split. The first run is kept in `superseded/` for transparency.
4. **Positional encoding.** TimesNet encodes position by index (0–9), not calendar month,
   so the May–June gap is not encoded. L-TAE and L-TAE-S use the true month numbers.

## Reproducibility

| What | How |
|---|---|
| Environment | `env/setup_env.sh` builds `ts_baselines` by cloning deep_field with `--copy` and adding aeon 1.6.0 and einops 0.8.2. Installing these into deep_field would have downgraded numpy 2.4.2 → 2.3.5 and numba 0.65 → 0.63, so deep_field is not touched. The script checks deep_field's `pip freeze` before and after (unchanged). Lock files: `env/pip_freeze.txt` and `env/environment.yml`. Differences from deep_field: `env/diff_vs_deep_field.txt`. |
| TimesNet code | Vendored unmodified from THUML Time-Series-Library at commit `4e938a1767106324dd753b2a44832bf870a0252e` (MIT). The files are byte-identical to upstream: TimesNet.py `f64c4bed…`, layers/Embed.py `17e7c357…`, layers/Conv_Blocks.py `16d9f2d9…`. |
| Determinism | Deterministic cuDNN and cuBLAS settings, a seeded DataLoader generator, fixed torch CPU threads, and aeon transforms that are deterministic for a given seed. A forced rerun of the TimesNet field smoke test gave a byte-identical prediction file. `run_all.sh` retrains seed 42 of both field models and records the result in `results/reproducibility_check.txt`. |
| Provenance | Every `models/*/metadata.json` records: git commit and dirty flag, package versions, GPU, SHA-256 of both input parquets, split sizes, input checks, hyperparameters, per-seed training time (true per seed, not 5-seed total ÷ 5), epochs or chosen C, and in-region metrics. |
| Published files untouched | `guard.py` hashes 622 published files: input parquets, `out_of_sample/`, published model directories and shared helper code. It does this before and after the runs, and `run_all.sh` fails on any difference. |
| Comparators | `score.py` first re-scores the ten published comparators and requires their holdout macro-F1 to match Table 1 (`results/provenance_check.csv`; all match). |

Timing note: the SI cost table (`stab:cost`) appears to mix 5-seed totals with per-seed
values for L-TAE, TempCNN and Transformer (pixel). Check this before comparing the new
per-seed times with it.

## Run

```bash
bash experiments/recent_baselines/env/setup_env.sh   # once
bash experiments/recent_baselines/run_all.sh         # train, repro check, score, report, guard
```

## Outputs

| Path | Content |
|---|---|
| `models/<model>_<level>/` | checkpoints, preprocessing artifacts, per-seed JSON, `metadata.json` |
| `results/predictions/` | `<model>_<level>.csv` (`fid,crop_name`, 2415 holdout fields), per-seed CSVs, `_proba.parquet` |
| `results/summary.csv` | Table-1-style row per model, new and published |
| `results/paired_bootstrap.csv` | paired field bootstrap (B = 10,000, seed 42) and exact McNemar for the pairs in `score.py` |
| `results/per_class/` | per-class report for each new model |
| `RESULTS.md` | the standalone write-up |
