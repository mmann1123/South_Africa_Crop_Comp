# Paired bootstrap (reviewer R1-3), staging area

Nothing here has been moved into the manuscript or `out_of_sample/` yet.

## Scripts

`paired_bootstrap.py` runs a paired field bootstrap (B = 10,000, seed 42) and an exact McNemar test on the key model pairs. It first checks each prediction file against the published Table 2 values. Output: `results/provenance_check.csv`, `results/paired_bootstrap.csv`.

`rerun_ltae_inference.py` re-runs L-TAE pixel inference from the existing checkpoints, with outputs redirected to `rerun/`.

`ltae_aggregation_variants.py` tries five pixel-to-field aggregation rules on those checkpoints and checks each against the published confusion matrix.

## Findings (2026-10-05)

1. L-TAE-S (pixel), L-TAE-S (field), L-TAE (field), Transformer (pixel), and TabNet (pixel) prediction files reproduce Table 2. Where a canonical confusion matrix exists, the match is exact.
2. **L-TAE (pixel) does not.** `out_of_sample/predictions_ltae.csv` gives F1m 0.54 / κ 0.40, against 0.58 / 0.49 published. Fresh inference from the current checkpoints reproduces 0.54 exactly. None of the 5 aggregation variants, and none of 128 prediction CSVs in the repo, reproduce the published confusion matrix. The published row most likely came from earlier checkpoints that were later overwritten.
3. Table 2 "Xent" is computed from hard labels clipped at 1e-7 (`compare_predictions.py:379-386`), so it equals 16.1 × error rate. **Correction (2026-10-09):** this is the official AI4FoodSecurity metric as the challenge defined it ("Cross Entropy with binary outcome for each crop (field level)", scored on hard-label submissions; `AI4EO_scoring.pdf`), so it is not an error.
4. The Methods describe a *paired* bootstrap, but `out_of_sample/bootstrap_ci.py` runs an unpaired one.

## Option (c): retrain L-TAE pixel (started 2026-10-05)

`experiments/field_reduction/train_ltae_pixel.py --fraction 1.0` retrains L-TAE pixel into `ltae_pixel_retrain/frac_1.00/`, with log `ltae_pixel_retrain.log`. This is the same harness, split, seeds, loss, optimizer and schedule as the published L-TAE-S frac_1.00 run; the only difference is the sparse gate. `predict_retrain.py` writes `rerun/predictions_ltae_retrain.csv`, and `paired_bootstrap.py` then adds the retrained-L-TAE pairs automatically.

### Pre-stated reading (written before results)

| Outcome | Effect on the paper's claims |
|---|---|
| L-TAE-S − L-TAE: Δ macro-F1 significant > 0 | Supports "the sparse gate improves transfer" in a controlled comparison. |
| L-TAE-S − L-TAE: Δ macro-F1 n.s., but Δκ/accuracy significant | Supports "associated with / consistent gains", not "drives". Same as the field-level result. |
| L-TAE-S − L-TAE: no significant Δ on any metric | Undercuts the controlled evidence. The claim must rest on the family-level association only. |
| Transformer ≈ L-TAE | Supports "capacity is not the lever". |
| Transformer significantly > L-TAE | Undercuts "capacity is not the lever". The argument becomes "capacity helps somewhat, but sparsity helps more", which only holds if L-TAE-S > Transformer (currently κ/accuracy yes, F1 borderline). |
| L-TAE in-region F1 and gap | If the pixel gap stays ≈ −0.20, Fig. bias and the "dense nets lose most" story hold. A smaller gap weakens the family contrast. |

### Results (2026-10-05)

- **Reproducibility.** The retrain is deterministic. All five checkpoints are bit-identical to `deep_learn/src/models/ltae_seed_*.pt` (Feb 17), and the holdout predictions are identical to `out_of_sample/predictions_ltae.csv`. So the reproducible L-TAE (pixel) result is: in-region F1m 0.776, OOR F1m **0.540** (95% CI 0.514 to 0.564), κ **0.399**, wF1 0.624, accuracy 0.602, gap **−0.24**. The published row (0.58 / κ 0.49 / Xent 5.20 ⇒ accuracy ≈ 0.68) does not come from these checkpoints; its source is unknown.
- **Paired tests vs. the reproducible L-TAE (pixel):** L-TAE-S − L-TAE: Δ F1m **+0.059** (0.038, 0.081), Δκ +0.129, Δacc +0.111, McNemar 384 vs. 116. Transformer − L-TAE: Δ F1m **+0.038** (0.019, 0.057), Δκ +0.104. L-TAE-S − Transformer: Δ F1m +0.022 (0.000, 0.044, p = 0.05), Δκ +0.025 (sig), Δacc +0.019 (sig). TabNet − L-TAE-S: n.s. on all metrics.

### Against the pre-stated reading

- **Sparse gate (pixel): supports.** Significant on all metrics; the F1m gain exceeds the paper's 0.05 threshold.
- **Sparse gate (field): partly supports.** F1m n.s.; κ and accuracy significant.
- **"Capacity is not the lever": undercut as written.** The Transformer clearly beats L-TAE, so the sentence "transfers no better than L-TAE (both 0.58)" is false. The defensible version: added attention capacity helps (+0.04), the sparse gate on the *smaller* L-TAE helps more (+0.06), and L-TAE-S still edges the Transformer (κ/acc significant, F1m borderline) with far fewer parameters.
- **Dense-net gap: strengthened.** L-TAE pixel gap −0.20 → −0.24, and field aggregation now recovers more for L-TAE (−0.24 → −0.07).

## Table 2 audit (2026-10-05)

1. **Transcription.** All 23 rows (in-region, gap, F1m, κ, wF1, Xent) match the canonical confusion matrices and `f1_macro_train_vs_oos.csv` to rounding.
2. **Traceability.** For 18 of 21 models with a canonical matrix, the current prediction file reproduces it exactly. The 3 that don't are all pixel-level deep models whose prediction files were rewritten on 2026-06-23. Re-running inference from the current checkpoints (Feb 17) reproduces the June files exactly (`rerun_pixel_dl_inference.py`, `rerun/`), so the published matrices (dated Aug 19) came from an untraceable earlier state.

| Model | Published OOR F1m / κ | Reproducible F1m / κ / wF1 / acc |
|---|---|---|
| L-TAE (pixel) | 0.58 / 0.49 | 0.540 / 0.399 / 0.624 / 0.602 (also confirmed by deterministic retrain) |
| CNN-BiLSTM (pixel) | 0.46 / 0.32 | 0.513 / 0.387 / 0.611 / 0.585 |
| TempCNN (pixel) | 0.56 / 0.49 | 0.574 / 0.489 / 0.683 / 0.692 |

3. **Feature labels.** Table 2 correctly lists the pixel LR/RF/LightGBM/XGBoost baselines as xr_fresh (`base_ml_models.py` reads `final_data.parquet`). The "raw pixel (band x month)" labels in `compare_predictions.py:45-48` are wrong (cosmetic only).
4. Not audited: in-region values (taken from training metadata), supplement ablation tables, per-class figure values.
