# L-TAE-S hyperparameter sensitivity (reviewer R1-2)

`run_sweep.py` runs the field-level grid; `run_pixel_check.py` runs two pixel-level settings. Both are resumable, and each configuration keeps its own checkpoints (`models/`, `pixel/`), predictions (`predictions/`, ensemble and per seed) and log (`logs/`). Summary tables are in `results/`.

The published configuration is γ = 1.5, 16 heads, d_k = 8, no entropy penalty, with the legacy constant L1 term λ = 10⁻³ (R1-4). Retraining it reproduces the published field-level predictions on 100% of fields (`results/reproducibility_check.txt`). Runs made while the constant term was removed are in `archive_without_legacy_term/`.

## Field-level results (2026-10-07; 5 seeds each; out-of-region macro-F1)

| γ \ heads | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| 1.0 | 0.607 | 0.608 | 0.586 | 0.578 |
| 1.2 | 0.601 | 0.612 | 0.595 | 0.597 |
| 1.5 | 0.608 | 0.598 | **0.601 (published)** | 0.584 |
| 2.0 | 0.615 | 0.611 | 0.597 | 0.590 |
| 3.0 | 0.615 | 0.587 | 0.581 | 0.161 (failed to train) |

The mask-entropy penalty λ (γ = 1.5, 16 heads) gives 0.601 (λ = 10⁻³), 0.594 (10⁻²) and 0.606 (10⁻¹).

- **Stability:** 22 of 23 configurations fall within 0.578 to 0.615 (κ 0.47 to 0.54). That range is comparable to seed-to-seed variation (per-configuration SD of single-seed F1m 0.004 to 0.028). The published setting is mid-range, not the best.
- **Heads:** fewer heads transfer slightly better (mean 0.609 for 4 heads and 0.603 for 8, against 0.592 for 16 and 0.587 for 32, excluding the failed run).
- **γ:** no consistent effect.
- **λ:** a genuine entropy penalty gives no measurable change, so sparsemax and the γ prior already provide the sparsity.
- **Failure:** γ = 3.0 with 32 heads did not train on any of the 5 seeds (validation F1 stayed at about 0.2 and early stopping ended each run after about 25 epochs). Each head multiplies the relaxation prior by up to γ, so it reaches about 3³¹ by the last head and swamps the learned gate. This is an extreme corner, outside any reasonable setting.

## Pixel-level check (2026-10-08)

| Setting | OOR F1m | κ | In-region F1m | Train time |
|---|---|---|---|---|
| γ = 2.0, 4 heads | 0.614 | 0.547 | 0.755 | 131 min |
| γ = 1.5, 16 heads (published) | 0.599 | 0.527 | 0.772 | (published) |
| γ = 1.0, 32 heads | 0.570 | 0.510 | 0.782 | 690 min |

Paired bootstrap (4,000 replicates): γ2/h4 − L-TAE ΔF1m +0.074 [0.053, 0.096], Δκ +0.148; γ1/h32 − L-TAE ΔF1m +0.031 [0.010, 0.052], Δκ +0.111; γ1/h32 − Transformer ΔF1m −0.007 (n.s.); γ2/h4 − TabNet ΔF1m +0.011 (n.s.); γ1/h32 − published L-TAE-S ΔF1m −0.028 [−0.046, −0.011]. The sparse gate improves on L-TAE at every setting tested, but the size of the gain depends on the number of heads: with 32 heads L-TAE-S only matches the Transformer.
