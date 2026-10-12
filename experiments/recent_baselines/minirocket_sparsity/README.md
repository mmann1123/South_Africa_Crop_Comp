# MiniRocket with L-TAE-S-style sparsity

A follow-up to the R2-Q3a baselines (`../`). MiniRocket transferred poorly
(holdout macro-F1 0.48 field, 0.50 pixel), in the bottom group of the study alongside the
patch-based CNNs. It fell below logistic regression on the raw
60 features while using the same kind of linear head, only on 10,000 unselected
random-convolution features.

This experiment asks whether adding sparse feature selection on top of the **same**
MiniRocket features recovers transfer. It mirrors the paper's L-TAE → L-TAE-S comparison
on a second backbone. The experiment is post hoc: the MiniRocket result motivated it.

## Variants (each at field and pixel level, 5 seeds)

| Model | Head on the frozen MiniRocket features |
|---|---|
| MiniRocket-S | L-TAE-S sparse gate: 16 heads, each with an instance-dependent sparsemax mask over the 9,996 features and the γ = 1.5 prior, then L-TAE-S's downstream MLP |
| MiniRocket-softmax | Identical network with softmax in place of sparsemax (dense masks). Isolates sparsity from the capacity the gate adds. |
| MiniRocket-L1 | L1-regularised multinomial logistic regression: global feature selection, no added capacity |
| MiniRocket-ridge (robustness, added later) | MiniRocket's canonical `RidgeClassifierCV` head (balanced class weights), with the canonical α grid and with a grid extended upward because the canonical one hit its edge |
| MiniRocket (reference) | The existing L2 logistic head (`../models/minirocket_*`) |

Gate details:
- The gate logits pass through a shared 64-unit bottleneck (Linear 9,996→64, then
  Linear 64→9,996 per head). A full 9,996 × 9,996 gate per head would be about 100M
  parameters; this follows TabNet's attentive transformer.
- Otherwise the gate mechanics are those of `models_arch.SparseFeatureGate` (BN, then
  prior, sparsemax, and the prior update), and the training harness is that of L-TAE-S
  unchanged.

## Same features, same rows

`features.py` reloads each seed's fitted MiniRocket and feature scaler from
`../models/minirocket_{level}/` and recomputes the features on exactly the rows the
reference model used. At pixel level that means the same ≤50-pixels-per-field draw.

Before training it checks that the saved L2 head, applied to the regenerated test
features, reproduces the saved test probabilities. Only the head differs from the
reference.

## Pre-stated hypotheses (written 2026-10-09, before any training in this folder)

- **H1:** MiniRocket-S > MiniRocket-softmax on holdout macro-F1. Sparsity, not the gate's
  added capacity, is what helps.
- **H2:** MiniRocket-L1 ≥ MiniRocket (L2 head). Global selection alone helps.
- **H3:** None of the variants reaches L-TAE-S (0.60). The random features themselves may
  limit transfer.

Results are reported whichever way they fall.

## Disclosed differences

- **Pixel-level gated nets:** early stopping uses ≤50 validation pixels per field (drawn
  with the seed). Sparsemax over 10k features × 16 heads on all 1.2M validation pixels
  every epoch would take hours. In-region test and holdout use every pixel.
- **L1 head:** fitted by FISTA (proximal gradient) in PyTorch, which minimises
  scikit-learn's L1 objective. `check_l1_equivalence.py` compares it with scikit-learn's
  saga solver. Its convergence tolerance is 1e-10 on the relative objective change. A
  first setting of 1e-6 stopped early on field data: the objective was within 0.2% of
  the optimum, but 2–3× too many features were left non-zero (354 against 134 at
  C = 0.01). The number of selected features is a reported quantity, so the tolerance was
  tightened before any L1 run.
- **L1 C path (rule added mid-run).** C is fitted from strong to weak regularisation, and
  the path stops once validation macro-F1 has fallen at two successive C values. Weak
  regularisation took hours per C at pixel level (seed 42 spent more than 1.5 h on
  C = 1), and fits past the validation peak cannot be selected. Selection still uses
  validation only. At field level the full grid had already peaked at C = 0.1 on every
  seed, so the rule changes nothing there; field was rerun with the same code anyway.
- **Sparsemax backward.** Pixel MiniRocket-S ran out of GPU memory, because autograd
  through sparsemax's sort and cumulative sum stores large intermediates for 16 heads.
  The gate now uses sparsemax's closed-form backward, which stores only the output. Its
  values and gradients match `models_arch.sparsemax` to about 2e-16. Batch size and
  BatchNorm are unchanged. Field MiniRocket-S was retrained with it for consistency. The
  retrain is not bit-identical to the first field run: in-region macro-F1 went from 0.673
  to 0.662, because rounding differences compound over training. That gap is about the
  size of seed-to-seed variation. Only the retrained run is reported.
- **aeon MiniRocket transform race (found and handled).** aeon 1.6.0's parallel MiniRocket
  transform intermittently returned one wrong row per 130k–200k-row batch: 16 events over
  Oct 9–11, each exactly one row. It happened only inside processes that also run PyTorch.
  Ten repeated passes over 300k rows in a separate process gave 0 differences.
  - **What it caused:** it made the per-seed identity check fail twice (pixel MiniRocket-S
    seed 303 and pixel softmax seed 42), each time stopping the run.
  - **Fix:** `features.checked_transform` computes every batch twice and requires
    agreement. On disagreement a third pass decides 2-to-1 and the event goes to
    `results/transform_mismatches.log`. All 16 events resolved this way.
  - **Earlier runs:** the 25 per-seed runs made before this check were retrained with it
    and all reproduced byte-for-byte (`results/verification.txt`). The reference
    MiniRocket test and holdout outputs were recomputed exactly
    (`results/verify_reference.csv`).
  - **Separate incident:** on Oct 9 (kernel 7.0.11) the machine also had kernel
    memory-corruption errors and later crashed. A 45-minute stressapptest of 90 GB then
    passed with 0 errors, and the race recurred on kernel 7.1.1 with no kernel errors, so
    the two are unrelated. The pixel softmax run made during that fault is archived in
    `superseded/softmax_pixel_faulty_2026-10-09/` and was rerun from scratch.
- **Gate precision:** the gate runs in fp32. Sparsemax's cumulative sum over about 10k
  features is inaccurate in fp16; L-TAE-S gates only 128 channels.

## Reproducibility

- **Environment and helpers:** the same `ts_baselines` environment and the same
  determinism settings and provenance metadata as `../`.
- **Guard:** a hash manifest covers the published files plus the completed baselines
  (`../models/minirocket_*`, `../results/`). It is taken before the runs and compared
  after; any change fails the run.
- **Repro check:** field seed 42 of each variant is retrained from scratch, and the
  holdout file must be byte-identical.

```bash
bash experiments/recent_baselines/minirocket_sparsity/run_all.sh
```
