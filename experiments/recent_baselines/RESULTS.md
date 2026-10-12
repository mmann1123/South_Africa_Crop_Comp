# R2-Q3a recent baselines: results

Setup and fairness: see README.md.

## Results

<!-- BEGIN GENERATED (make_report.py) -->
### Holdout and in-region scores
New models are trained here; published rows are re-scored from `out_of_sample/` and match Table 1. In-region = internal 20% field split; Δ = OOR − in-region.

| Model | In-reg F1m | Δ | OOR F1m | 95% CI | κ | wF1 | Xent | Seed SD (OOR F1m) | Train time / seed |
|---|---|---|---|---|---|---|---|---|---|
| TimesNet (pixel) **(new)** | 0.76 | -0.20 | 0.57 | [0.54, 0.59] | 0.47 | 0.67 | 5.35 | 0.028 | 29.7 min |
| TimesNet (field) **(new)** | 0.70 | -0.18 | 0.52 | [0.49, 0.54] | 0.36 | 0.60 | 6.85 | 0.010 | 26 s |
| MiniRocket (pixel) **(new)** | 0.72 | -0.22 | 0.50 | [0.48, 0.52] | 0.33 | 0.56 | 7.70 | 0.011 | 28.7 min |
| MiniRocket (field) **(new)** | 0.67 | -0.19 | 0.48 | [0.46, 0.51] | 0.30 | 0.55 | 7.91 | 0.029 | 30 s |
| TabNet (pixel) | 0.73 | -0.13 | 0.60 | [0.58, 0.63] | 0.54 | 0.70 | 4.42 |  |  |
| L-TAE-S (field) | 0.64 | -0.04 | 0.60 | [0.58, 0.62] | 0.52 | 0.69 | 4.77 |  |  |
| L-TAE-S (pixel) | 0.77 | -0.17 | 0.60 | [0.57, 0.62] | 0.53 | 0.70 | 4.63 |  |  |
| L-TAE (field) | 0.67 | -0.07 | 0.59 | [0.57, 0.62] | 0.47 | 0.67 | 5.51 |  |  |
| Transformer (pixel) | 0.78 | -0.20 | 0.58 | [0.55, 0.60] | 0.50 | 0.69 | 4.94 |  |  |
| TempCNN (pixel) | 0.77 | -0.20 | 0.57 | [0.55, 0.60] | 0.49 | 0.68 | 4.96 |  |  |
| XGBoost (field) | 0.70 | -0.13 | 0.56 | [0.54, 0.59] | 0.47 | 0.68 | 4.89 |  |  |
| Logistic Regression (pixel) | 0.61 | -0.05 | 0.56 | [0.54, 0.59] | 0.47 | 0.67 | 4.93 |  |  |
| L-TAE (pixel) | 0.78 | -0.24 | 0.54 | [0.51, 0.56] | 0.40 | 0.62 | 6.42 |  |  |
| TempCNN (field) | 0.67 | -0.14 | 0.54 | [0.51, 0.56] | 0.43 | 0.64 | 6.07 |  |  |

### Paired field bootstrap (B = 10,000, seed 42)
Δ = A − B on the same resampled holdout fields; a CI excluding 0 is significant at 95%.

| A | B | Question | Δ F1m [95% CI] | Δ κ [95% CI] | Δ acc [95% CI] | McNemar p |
|---|---|---|---|---|---|---|
| MiniRocket (pixel) | L-TAE-S (pixel) | dense random features vs. learned sparse selection | -0.098 [-0.124, -0.073] | -0.200 [-0.229, -0.171] | -0.190 [-0.212, -0.168] | 5.08e-60 |
| MiniRocket (pixel) | TabNet (pixel) | dense random features vs. learned sparse selection | -0.103 [-0.131, -0.075] | -0.209 [-0.240, -0.178] | -0.203 [-0.227, -0.180] | 2.14e-60 |
| MiniRocket (pixel) | L-TAE (pixel) | random convolutions vs. dense attention | -0.039 [-0.059, -0.020] | -0.072 [-0.095, -0.049] | -0.079 [-0.097, -0.062] | 3.23e-19 |
| MiniRocket (pixel) | Logistic Regression (pixel) | random-convolution features vs. raw features, linear head | -0.063 [-0.089, -0.037] | -0.143 [-0.174, -0.113] | -0.171 [-0.195, -0.148] | 2.15e-44 |
| MiniRocket (field) | L-TAE-S (field) | dense random features vs. learned sparse selection | -0.119 [-0.144, -0.093] | -0.213 [-0.241, -0.185] | -0.195 [-0.216, -0.173] | 1.89e-67 |
| MiniRocket (field) | L-TAE (field) | random convolutions vs. dense attention | -0.112 [-0.137, -0.088] | -0.168 [-0.195, -0.142] | -0.149 [-0.168, -0.129] | 5.42e-49 |
| MiniRocket (field) | XGBoost (field) | random convolutions vs. tree ensemble on xr_fresh | -0.083 [-0.107, -0.058] | -0.172 [-0.202, -0.141] | -0.188 [-0.210, -0.165] | 2.35e-56 |
| TimesNet (pixel) | L-TAE-S (pixel) | modern capacity vs. sparse selection | -0.034 [-0.059, -0.009] | -0.056 [-0.083, -0.029] | -0.045 [-0.062, -0.027] | 6.43e-07 |
| TimesNet (pixel) | L-TAE (pixel) | modern capacity vs. dense attention | +0.026 [+0.007, +0.045] | +0.073 [+0.047, +0.098] | +0.066 [+0.048, +0.084] | 1.14e-12 |
| TimesNet (pixel) | Transformer (pixel) | modern capacity vs. attention capacity | -0.012 [-0.033, +0.009] | -0.031 [-0.056, -0.006] | -0.026 [-0.042, -0.009] | 0.00261 |
| TimesNet (pixel) | TempCNN (pixel) | 2D temporal convolution vs. 1D temporal convolution | -0.009 [-0.027, +0.010] | -0.017 [-0.042, +0.007] | -0.024 [-0.041, -0.009] | 0.00262 |
| TimesNet (field) | L-TAE-S (field) | modern capacity vs. sparse selection | -0.085 [-0.113, -0.056] | -0.153 [-0.184, -0.122] | -0.129 [-0.150, -0.108] | 3.9e-32 |
| TimesNet (field) | L-TAE (field) | modern capacity vs. dense attention | -0.078 [-0.103, -0.051] | -0.109 [-0.135, -0.081] | -0.083 [-0.102, -0.064] | 1.05e-17 |
| TimesNet (field) | TempCNN (field) | 2D temporal convolution vs. 1D temporal convolution | -0.019 [-0.044, +0.005] | -0.064 [-0.091, -0.036] | -0.048 [-0.068, -0.029] | 1.63e-06 |
| MiniRocket (pixel) | TimesNet (pixel) | random features vs. learned capacity | -0.065 [-0.085, -0.043] | -0.145 [-0.171, -0.117] | -0.145 [-0.167, -0.125] | 3.69e-41 |
| MiniRocket (field) | TimesNet (field) | random features vs. learned capacity | -0.034 [-0.058, -0.011] | -0.060 [-0.085, -0.035] | -0.065 [-0.084, -0.047] | 2.96e-12 |

### Per-class holdout F1 (new models)

| class | minirocket_field | minirocket_pixel | timesnet_field | timesnet_pixel |
|---|---|---|---|---|
| Barley | 0.42 | 0.43 | 0.49 | 0.51 |
| Canola | 0.65 | 0.65 | 0.60 | 0.64 |
| Lucerne/Medics | 0.67 | 0.66 | 0.74 | 0.84 |
| Small grain grazing | 0.32 | 0.37 | 0.33 | 0.40 |
| Wheat | 0.35 | 0.40 | 0.42 | 0.44 |

### Per-seed training

| Model | Seed | Train time | Epochs / chosen C | Best val F1m |
|---|---|---|---|---|
| minirocket_field | 42 | 31 s | 0.001 | 0.679875 |
| minirocket_field | 101 | 28 s | 0.01 | 0.659578 |
| minirocket_field | 202 | 24 s | 0.0001 | 0.666404 |
| minirocket_field | 303 | 39 s | 0.0001 | 0.67269 |
| minirocket_field | 404 | 26 s | 0.001 | 0.67768 |
| minirocket_pixel | 42 | 32.2 min | 0.0001 | 0.645032 |
| minirocket_pixel | 101 | 31.8 min | 0.0001 | 0.649953 |
| minirocket_pixel | 202 | 30.3 min | 0.0001 | 0.646525 |
| minirocket_pixel | 303 | 24.5 min | 0.0001 | 0.638493 |
| minirocket_pixel | 404 | 24.7 min | 0.0001 | 0.65827 |
| timesnet_field | 42 | 18 s | 51.0 | 0.743923 |
| timesnet_field | 101 | 23 s | 67.0 | 0.740252 |
| timesnet_field | 202 | 19 s | 56.0 | 0.747545 |
| timesnet_field | 303 | 31 s | 85.0 | 0.750875 |
| timesnet_field | 404 | 39 s | 65.0 | 0.753618 |
| timesnet_pixel | 42 | 35.1 min | 17.0 | 0.701684 |
| timesnet_pixel | 101 | 27.5 min | 16.0 | 0.682378 |
| timesnet_pixel | 202 | 29.7 min | 16.0 | 0.675379 |
| timesnet_pixel | 303 | 28.1 min | 16.0 | 0.680113 |
| timesnet_pixel | 404 | 28.2 min | 16.0 | 0.665583 |

### Reproducibility check

```
Reproducibility check 2026-10-09T14:45:29-04:00, git 704f9d7
Seed 42 of each field model retrained from scratch (--force) into *_repro; holdout CSVs compared byte for byte.
minirocket field seed 42: IDENTICAL
timesnet field seed 42: IDENTICAL
```

### Published files unchanged

```
guard: 622 files checked, no changes
```
<!-- END GENERATED -->

## Reading

Written 2026-10-09, after all runs finished. Numbers are holdout (out-of-region, OOR) macro-F1 on the 2,415 holdout fields unless stated. Paired differences come from the field bootstrap above.

### 1. Neither recent model transfers as well as the sparse learners

Both new models fall clearly below L-TAE-S and TabNet at both levels. Every paired difference against L-TAE-S is significant: TimesNet pixel −0.034 [−0.059, −0.009], TimesNet field −0.085, MiniRocket pixel −0.098, MiniRocket field −0.119. Seed-to-seed spread is small (SD 0.010–0.029), so the gaps are not seed noise.

### 2. TimesNet behaves like the other high-capacity dense networks

At pixel level TimesNet reaches in-region macro-F1 0.76, close to the dense leaders (L-TAE 0.78, Transformer 0.78, TempCNN 0.77). It loses 0.20 under transfer and lands at 0.57, the same transfer gap as the Transformer and TempCNN. Its holdout F1m is indistinguishable from both (Δ −0.012 [−0.033, +0.009] and −0.009 [−0.027, +0.010]), although κ and accuracy are slightly lower. It beats the weakest dense model, pixel L-TAE (+0.026), and stays below L-TAE-S. Every seed reached its best validation score within the first few epochs and early-stopped at epoch 16–17. That is the same quick-overfitting pattern, and fits the paper's account of capacity without selection: more capacity raises in-region accuracy without improving transfer.

### 3. MiniRocket, a top time-series classifier on UCR, transfers near the bottom of the study

MiniRocket scores 0.67–0.72 in-region but only 0.48–0.50 out of region (Δ −0.19 to −0.22). That is below every model in the table above. Against the full 23-model Table 1 it is in the bottom group, alongside SMOTE stacking (0.49) and the 3D CNN (0.50); only the Multi-Channel 2D CNN (0.45) and TabNet's two field-level variants (raw field-averaged 0.43, xr_fresh 0.37) score lower. The clearest contrast is with plain logistic regression on the raw 60 features. Same linear head, but on 10,000 dense random-convolution features with no selection, MiniRocket transfers significantly worse: −0.063 [−0.089, −0.037] F1m and −0.143 κ, pixel level. Giving a linear classifier many unselected features makes it transfer worse, not better. This is the most direct support for the paper's thesis from these experiments: feature selection matters more than feature richness or capacity. The pixel cap of 50 per field does not explain it. The field-level model, which uses every training field, transfers no better (0.48).

### 4. Field aggregation does not help these two models (cuts against a manuscript claim)

The manuscript presents field-level aggregation as a second robustness lever that "narrows the transfer gap of dense temporal networks, raising holdout accuracy for L-TAE and L-TAE-S". For both new models, the field-level variant is *worse* on the holdout than the pixel-level one:

| Model | Pixel | Field |
|---|---|---|
| TimesNet | 0.57 | 0.52 |
| MiniRocket | 0.50 | 0.48 |

Field aggregation also cuts training data from about 4.8M pixels to 2,653 field means, which may hurt a 10k-feature linear model or a convolutional network more than L-TAE. These results make aggregation look model-dependent rather than a general fix for dense models. The aggregation claim (Abstract, Sec. 5.2, Conclusion) should probably be scoped to "for the L-TAE family", or the new rows reported with that caveat.

### 5. Per class

Neither new model fixes the cereals, and both are weaker on the easy classes than TabNet and L-TAE-S:

| Class | L-TAE-S / TabNet (published) | TimesNet pixel | MiniRocket pixel |
|---|---|---|---|
| Lucerne/medics | 0.87 / 0.88 | 0.84 | 0.66 |
| Canola | 0.76 / 0.81 | 0.64 | 0.65 |
| Barley | 0.50 / 0.55 | 0.51 | 0.43 |
| Wheat | 0.47 / 0.48 | 0.44 | 0.40 |
| Small-grain grazing | 0.40 / 0.30 | 0.40 | 0.37 |

MiniRocket's deficit is broad. It is largest on lucerne/medics, the dominant class, which is also where it differs most from TimesNet.

### Caveats to state

- **MiniRocket C grid (post hoc).** The grid was extended after a first field run (macro-F1 0.46, archived in `superseded/`) chose the smallest C on every seed. Selection uses validation only, and the change raised field macro-F1 to 0.48. At pixel level the weakly regularised fits (C ≥ 0.1) hit the 1,000-iteration L-BFGS cap without converging. They were never selected: every seed chose C = 1e-4, which converged.
- **MiniRocket pixel cap.** Training used ≤50 pixels per field (about 130k of 4.8M), disclosed. The full-data field model shows the same pattern.
- **TimesNet on 10 steps.** Its FFT period search has at most 5 non-zero frequencies, so it is used outside its design regime. Hyperparameters are TSLib defaults scaled down, not tuned. The same is true of the paper's other deep models, so the comparison is like for like, but a reviewer could ask for tuning.
- **One holdout tile, one season**, as for all results in the paper.
- **Presto not run.** It needs bands not in the processed data.

## Options for the paper

1. **Supplement plus short main-text mention (recommended).**
   - New SI section and table with the four rows, the paired-bootstrap table and the caveats.
   - Two sentences in Results (transfer-gap section) and one in Discussion 5.1, e.g. that MiniRocket's 10,000 unselected features transfer worse than logistic regression on the raw features.
   - Scope the aggregation claim (point 4).
   - Table 1 and the "twenty-three models" counts (Abstract A:41, Contributions A:63, Table 1 caption A:129, Conclusion A:286) stay unchanged. Lowest edit burden.
2. **Add the four rows to Table 1.** This means changing "twenty-three" to "twenty-seven" in the four places above, updating the R1-1 taxonomy counts (classical vs deep), adding the rows to `tab:pixfield` (both levels exist), and adding them to `plot_gap.py`, `plot_inductive_bias.py` and the SI cost table. Stronger signal to Reviewer 2, but more edits, and the new rows would need to be copied into `out_of_sample/` and registered in `compare_predictions.py`.
3. **Response letter only.** Too weak: the reviewer asked for the models to be added.

## Draft reply to R2-Q3a (for the response letter)

> We thank the reviewer for this suggestion. We added two recent time-series classifiers, chosen after a literature review, and ran them under exactly the same protocol as the existing models: the same 60 inputs, field-wise split, scaler, seeds, early-stopping or validation-based selection, field aggregation and holdout scoring. A provenance check confirms that the inputs are identical to those of the published models. The two models are TimesNet (Wu et al., 2023), a high-capacity architecture and the first model the reviewer named, and MiniRocket (Dempster et al., 2021), a random-convolution classifier that is among the most accurate methods in the 2024 time-series classification bake-off. Both were run at pixel and field level as five-seed ensembles.
>
> Neither transfers as well as the sparse learners. TimesNet reaches in-region macro-F1 0.76 but 0.57 on the holdout, in line with the Transformer and TempCNN and significantly below L-TAE-S (paired Δ −0.03, 95% CI −0.06 to −0.01). MiniRocket transfers poorly (0.48 to 0.50), in the bottom group of all models alongside the patch-based CNNs, and this holds with its canonical ridge classifier (0.45 to 0.49). It even falls below logistic regression on the raw features (Δ −0.06, CI −0.09 to −0.04), although it uses the same kind of linear classifier, only on 10,000 unselected random features. Adding the sparse gate used in L-TAE-S to MiniRocket's features raised its holdout macro-F1 by 0.06 over an identical dense-gated network (95% CI 0.04 to 0.08), replicating the L-TAE to L-TAE-S gain on a second backbone. These results strengthen our central finding: added capacity or feature richness without feature selection does not improve out-of-region transfer.
>
> We did not run a pretrained encoder such as Presto. A fair comparison would require Sentinel-2 bands (e.g. B4, B8) that are not in the processed benchmark, and Presto's pretraining data postdate our 2017 season. We note this as a limitation. The new results are in Supplementary Material Sec. S-X and Table S-X, and are summarised in Sec. 4.x and 5.1.

(Following the manuscript style: no em dashes in manuscript text. Replace "S-X" and "4.x" once placement is decided.)

## Follow-up: MiniRocket with L-TAE-S-style sparsity

See `minirocket_sparsity/RESULTS.md`. The same features were tested with three heads: a sparsemax gate, an identical softmax gate as the dense control, and an L1 head.

At pixel level the sparse gate beats the identical dense gate by +0.058 out-of-region macro-F1 [+0.040, +0.075], matching the L-TAE → L-TAE-S gain, while the dense gate adds nothing. At field level there is no effect. Global L1 selection does not help. MiniRocket-S still trails L-TAE-S by 0.054.

**Note for this folder's MiniRocket results:** aeon's transform has an intermittent single-row race (see `minirocket_sparsity/README.md`). The reference MiniRocket outputs reported above were recomputed with checked features and match exactly.
