# MiniRocket with L-TAE-S-style sparsity: results

Design, hypotheses and disclosures: see README.md.

## Results

<!-- BEGIN GENERATED (make_report.py) -->
### Holdout and in-region scores
Reference rows: MiniRocket from `../`, others from `out_of_sample/` (provenance-checked).

| Model | In-reg F1m | Δ | OOR F1m | 95% CI | κ | Xent | Seed SD | Train / seed |
|---|---|---|---|---|---|---|---|---|
| MiniRocket-S (pixel) **(new)** | 0.76 | -0.21 | 0.545 | [0.52, 0.57] | 0.44 | 5.61 | 0.024 | 22.6 min |
| MiniRocket-S (field) **(new)** | 0.66 | -0.17 | 0.496 | [0.47, 0.52] | 0.38 | 5.49 | 0.032 | 71 s |
| MiniRocket-softmax (pixel) **(new)** | 0.75 | -0.26 | 0.487 | [0.46, 0.51] | 0.34 | 7.09 | 0.032 | 7.6 min |
| MiniRocket-softmax (field) **(new)** | 0.68 | -0.18 | 0.502 | [0.48, 0.53] | 0.37 | 6.69 | 0.024 | 16 s |
| MiniRocket-L1 (pixel) **(new)** | 0.72 | -0.21 | 0.508 | [0.48, 0.53] | 0.32 | 7.82 | 0.010 | 7.8 min |
| MiniRocket-L1 (field) **(new)** | 0.66 | -0.18 | 0.486 | [0.46, 0.51] | 0.29 | 8.09 | 0.024 | 4 s |
| MiniRocket-ridge (pixel) **(new)** | 0.72 | -0.23 | 0.491 | [0.47, 0.52] | 0.29 | 8.23 | 0.011 | 9.7 min |
| MiniRocket-ridge (field) **(new)** | 0.66 | -0.20 | 0.455 | [0.43, 0.48] | 0.26 | 8.57 | 0.033 | 1 s |
| MiniRocket-ridge-ext (pixel) **(new)** | 0.71 | -0.22 | 0.491 | [0.47, 0.52] | 0.29 | 8.14 | 0.010 | 10.0 min |
| MiniRocket-ridge-ext (field) **(new)** | 0.68 | -0.21 | 0.468 | [0.44, 0.49] | 0.28 | 8.34 | 0.017 | 1 s |
| MiniRocket (pixel) | 0.72 | -0.22 | 0.500 | [0.48, 0.52] | 0.33 | 7.70 |  |  |
| MiniRocket (field) | 0.67 | -0.19 | 0.482 | [0.46, 0.51] | 0.30 | 7.91 |  |  |
| L-TAE-S (pixel) | 0.77 | -0.17 | 0.599 | [0.57, 0.62] | 0.53 | 4.63 |  |  |
| L-TAE-S (field) | 0.64 | -0.04 | 0.601 | [0.58, 0.62] | 0.52 | 4.77 |  |  |
| L-TAE (pixel) | 0.78 | -0.24 | 0.539 | [0.51, 0.56] | 0.40 | 6.42 |  |  |
| L-TAE (field) | 0.67 | -0.07 | 0.594 | [0.57, 0.62] | 0.47 | 5.51 |  |  |
| Logistic Regression (pixel) | 0.61 | -0.05 | 0.563 | [0.54, 0.59] | 0.47 | 4.93 |  |  |

### How sparse is each head?
Gate statistics on validation samples (mean over seeds); L1 = features with any non-zero weight. MiniRocket has 9,996 features.

| Model | Non-zero features / head / sample | Effective features / head (exp entropy) | Union of features selected | L1 non-zero features | L1 chosen C (per seed) |
|---|---|---|---|---|---|
| MiniRocket-S (pixel) | 1.8 | 1.6 | 9973 |  |  |
| MiniRocket-S (field) | 1.9 | 1.6 | 6743 |  |  |
| MiniRocket-softmax (pixel) | 5590.5 | 671.8 | 9996 |  |  |
| MiniRocket-softmax (field) | 5769.4 | 798.0 | 9996 |  |  |
| MiniRocket-L1 (pixel) |  |  |  | 4125 | 0.01;0.01;0.01;0.01;0.01 |
| MiniRocket-L1 (field) |  |  |  | 1175 | 0.1;0.1;0.1;0.1;0.1 |
| MiniRocket-ridge (pixel) |  |  |  |  |  |
| MiniRocket-ridge (field) |  |  |  |  |  |
| MiniRocket-ridge-ext (pixel) |  |  |  |  |  |
| MiniRocket-ridge-ext (field) |  |  |  |  |  |

### Paired field bootstrap (B = 10,000, seed 42)

| A | B | Question | Δ F1m [95% CI] | Δ κ [95% CI] | McNemar p |
|---|---|---|---|---|---|
| MiniRocket-S (pixel) | MiniRocket-softmax (pixel) | sparsity itself (sparsemax vs softmax gate, same network) | +0.058 [+0.040, +0.075] | +0.096 [+0.074, +0.117] | 1.79e-28 |
| MiniRocket-S (pixel) | MiniRocket (pixel) | does the sparse gate help? | +0.045 [+0.023, +0.066] | +0.110 [+0.083, +0.137] | 6.05e-33 |
| MiniRocket-softmax (pixel) | MiniRocket (pixel) | does added capacity alone help? | -0.013 [-0.034, +0.007] | +0.015 [-0.008, +0.038] | 7.06e-05 |
| MiniRocket-L1 (pixel) | MiniRocket (pixel) | does global (L1) selection help? | +0.008 [+0.001, +0.015] | -0.007 [-0.016, +0.003] | 0.0662 |
| MiniRocket-S (pixel) | L-TAE-S (pixel) | gap to the paper's sparse model | -0.054 [-0.080, -0.028] | -0.090 [-0.119, -0.061] | 1.11e-10 |
| MiniRocket-S (field) | MiniRocket-softmax (field) | sparsity itself (sparsemax vs softmax gate, same network) | -0.005 [-0.027, +0.016] | +0.010 [-0.017, +0.038] | 2.5e-14 |
| MiniRocket-S (field) | MiniRocket (field) | does the sparse gate help? | +0.014 [-0.009, +0.038] | +0.075 [+0.045, +0.106] | 4.03e-36 |
| MiniRocket-softmax (field) | MiniRocket (field) | does added capacity alone help? | +0.020 [-0.000, +0.039] | +0.065 [+0.042, +0.088] | 6.01e-17 |
| MiniRocket-L1 (field) | MiniRocket (field) | does global (L1) selection help? | +0.004 [-0.009, +0.017] | -0.009 [-0.025, +0.007] | 0.0842 |
| MiniRocket-S (field) | L-TAE-S (field) | gap to the paper's sparse model | -0.105 [-0.134, -0.076] | -0.138 [-0.168, -0.107] | 4.55e-07 |
| MiniRocket-L1 (pixel) | Logistic Regression (pixel) | selected random features vs. raw features | -0.055 [-0.081, -0.029] | -0.150 [-0.181, -0.119] | 3.16e-46 |
| MiniRocket-ridge (pixel) | MiniRocket (pixel) | canonical ridge head vs. L2 logistic head | -0.009 [-0.022, +0.004] | -0.042 [-0.058, -0.025] | 6.48e-07 |
| MiniRocket-ridge-ext (pixel) | MiniRocket (pixel) | ridge head (extended alpha grid) vs. L2 logistic head | -0.009 [-0.021, +0.004] | -0.036 [-0.051, -0.020] | 1.91e-05 |
| MiniRocket-ridge-ext (pixel) | L-TAE-S (pixel) | best-practice MiniRocket vs. the paper's sparse model | -0.107 [-0.135, -0.079] | -0.236 [-0.266, -0.205] | 4.5e-68 |
| MiniRocket-ridge (field) | MiniRocket (field) | canonical ridge head vs. L2 logistic head | -0.028 [-0.043, -0.011] | -0.047 [-0.065, -0.029] | 1.21e-08 |
| MiniRocket-ridge-ext (field) | MiniRocket (field) | ridge head (extended alpha grid) vs. L2 logistic head | -0.014 [-0.028, -0.001] | -0.024 [-0.038, -0.009] | 7.09e-06 |
| MiniRocket-ridge-ext (field) | L-TAE-S (field) | best-practice MiniRocket vs. the paper's sparse model | -0.133 [-0.159, -0.107] | -0.237 [-0.266, -0.208] | 4.6e-78 |
| MiniRocket-ridge-ext (pixel) | Logistic Regression (pixel) | ridge on random features vs. logistic on raw features | -0.071 [-0.097, -0.045] | -0.179 [-0.209, -0.149] | 1.86e-55 |

### Per-class holdout F1

| class | minirocket_l1_field | minirocket_l1_pixel | minirocket_ridge_ext_field | minirocket_ridge_ext_pixel | minirocket_ridge_field | minirocket_ridge_pixel | minirocket_s_field | minirocket_s_pixel | minirocket_softmax_field | minirocket_softmax_pixel |
|---|---|---|---|---|---|---|---|---|---|---|
| Barley | 0.41 | 0.48 | 0.39 | 0.46 | 0.34 | 0.47 | 0.43 | 0.47 | 0.48 | 0.44 |
| Canola | 0.66 | 0.65 | 0.66 | 0.66 | 0.66 | 0.66 | 0.62 | 0.66 | 0.65 | 0.66 |
| Lucerne/Medics | 0.63 | 0.64 | 0.62 | 0.61 | 0.60 | 0.60 | 0.81 | 0.83 | 0.78 | 0.75 |
| Small grain grazing | 0.34 | 0.36 | 0.36 | 0.37 | 0.33 | 0.36 | 0.24 | 0.33 | 0.28 | 0.28 |
| Wheat | 0.39 | 0.41 | 0.30 | 0.35 | 0.35 | 0.36 | 0.39 | 0.43 | 0.33 | 0.31 |

### L1 solver check (FISTA vs scikit-learn saga, field, seed 42)

| C | ours_iter | ours_converged | ours_nonzero_features | sk_nonzero_features | ours_objective | sk_objective | sk_iter | val_pred_agreement | max_abs_proba_diff | ours_s | sk_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 154 | True | 134 | 116 | 0.012495 | 0.012495 | 20000 | 0.995482 | 0.013356 | 3.8 | 11295.0 |
| 0.1 | 487 | True | 1171 | 1280 | 0.068728 | 0.068797 | 20000 | 0.99247 | 0.068052 | 7.7 | 8321.4 |

### Reproducibility check

```
Reproducibility check 2026-10-09T19:09:57-04:00, git 704f9d7
Field seed 42 of each variant retrained from scratch (--force) into *_repro; holdout CSVs compared byte for byte.
minirocket_s field seed 42: IDENTICAL
minirocket_softmax field seed 42: IDENTICAL
minirocket_l1 field seed 42: IDENTICAL
minirocket_ridge field seed 42: IDENTICAL
minirocket_ridge_ext field seed 42: IDENTICAL
```

### Published and baseline files unchanged

```
guard: 767 files checked, no changes
```
<!-- END GENERATED -->

## Reading

Written 2026-10-11, after all runs and the verification retrains finished. OOR = holdout macro-F1 on 2,415 fields. Δ values are paired field-bootstrap differences (B = 10,000) with 95% CIs.

### H1 (sparsity, not capacity): supported at pixel level, null at field level

At **pixel level** the sparsemax gate beats the identical network with a softmax gate by **+0.058 OOR macro-F1 [+0.040, +0.075]** and +0.096 κ. The dense softmax gate adds nothing over plain MiniRocket (Δ −0.013 [−0.034, +0.007]), so the gain comes from sparsity, not from the gate's added capacity. The two gated models are almost equal in-region (0.76 vs 0.75); the sparse one loses less when moved (Δ in-region→OOR −0.21 vs −0.26).

This replicates the paper's controlled comparison on an unrelated backbone. L-TAE → L-TAE-S at pixel level gave +0.06 [0.04, 0.08]; MiniRocket + softmax gate → MiniRocket + sparsemax gate gives +0.058 [0.040, 0.075].

At **field level** there is no sparsity effect (Δ −0.005 [−0.027, +0.016]). Both gates raise κ by about the same amount over the linear head (+0.075 and +0.065), which is a capacity effect. This also mirrors the paper: the L-TAE-S field-level macro-F1 gain over L-TAE was not significant. A plausible reason is that 2,653 training fields are too few to train a gate with about 10M parameters over 9,996 features. This is an inference, not tested here.

### H2 (global selection): essentially null

The L1 head discards most features (it keeps 1,175 of 9,996 at field level and 4,125 at pixel level), yet changes holdout accuracy very little: +0.008 [+0.001, +0.015] at pixel level (κ n.s.), and +0.004 n.s. at field level. Removing features globally does not reproduce the gate's gain.

The gate's selection is *instance-dependent*. Each head uses about 1.8 features per sample, but across samples the heads draw on almost every feature (9,973 at pixel level). The working mechanism is therefore per-sample sparse selection (TabNet / L-TAE-S style), not a sparse model. This sharpens the paper's thesis: "feature selection" should be read as instance-wise selection.

### H3 (gap to L-TAE-S): holds

Even the best variant, MiniRocket-S (pixel), stays below L-TAE-S: Δ −0.054 [−0.080, −0.028]. It also stays below logistic regression on the raw 60 features (0.545 vs 0.563 OOR). Sparsity recovers about half of MiniRocket's deficit; the random-convolution representation limits the rest.

### Canonical ridge head (robustness check, added 2026-10-12)

MiniRocket's standard head is `RidgeClassifierCV` (alphas = logspace(−3, 3, 10), leave-one-out CV), as in aeon, sktime and the bake-off. We ran it on the same features with balanced class weights, in two versions: the canonical grid, and an extended grid, because the canonical one chose its upper edge (α = 1000) on every seed. Extended up to 10⁸, the grid settles at α = 10⁴.

Ridge does **not** rescue MiniRocket; it is slightly worse than our L2 logistic head:

| Comparison | Pixel Δ F1m [95% CI] | Field Δ F1m [95% CI] |
|---|---|---|
| Canonical ridge vs logistic head | −0.009 [−0.022, +0.004] (κ −0.04, significant) | −0.028 [−0.043, −0.011] |
| Extended-α ridge vs logistic head | −0.009 [−0.021, +0.004] (κ −0.04, significant) | −0.014 [−0.028, −0.001] |
| Extended-α ridge vs L-TAE-S | −0.107 [−0.135, −0.079] | −0.133 [−0.159, −0.107] |

Holdout macro-F1 is 0.49 (pixel) and 0.45–0.47 (field). The choice of head therefore does not explain MiniRocket's weak transfer, and the canonical configuration a reviewer might ask for does no better than ours.

### Per class

The sparse gate's pixel gain is concentrated in lucerne/medics and wheat:
- **lucerne/medics:** 0.83 (sparsemax) vs 0.75 (softmax) and 0.64–0.66 (linear or L1 heads)
- **wheat:** 0.43 vs 0.31

Canola is flat (0.65–0.66) across all heads.

### Caveats

- **Post hoc.** The MiniRocket result motivated this experiment. The hypotheses were written in the README before any training here.
- **The gate is adapted, not copied.** A 64-unit bottleneck produces the gate logits, because a full gate over 9,996 features per head is too large.
- **Pixel-level early stopping** uses ≤50 validation pixels per field. The MiniRocket training rows are capped at 50 per field, as for the reference.
- **The L1 C path** stops after two successive validation declines (rule added mid-run, disclosed). At field level it changes nothing.
- **aeon transform race** (README). aeon's parallel MiniRocket transform intermittently returned one wrong row per 130k–200k-row batch: 16 events, only in processes that also run PyTorch. Every feature batch is now computed twice, with a third pass resolving any disagreement.
- **Verification.** All 25 per-seed holdout files of the runs made before that check were reproduced byte-for-byte by retrains with checked features (`results/verification.txt`). Pixel MiniRocket-softmax was rerun from scratch with checks; its first attempt, made during the Oct 9 system fault, is archived in `superseded/`.

### For the paper

This is the strongest new evidence for the thesis from the revision experiments. The sparsity effect replicates, with the same size, on an independent backbone and against a capacity-matched dense control. The field-level null result matches the paper's own field-level result.

A natural home is the SI section that will hold the R2-Q3a baselines. Add one paired-bootstrap table and two or three sentences, plus one sentence in Discussion 5.1, for example: "Adding the same sparse gate to MiniRocket's random-convolution features raised out-of-region macro-F1 by 0.06 over an identical dense-gated network, matching the L-TAE to L-TAE-S gain, whereas global L1 selection did not help."
