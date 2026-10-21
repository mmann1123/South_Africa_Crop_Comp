# Fill-value sweep

Holdout macro-F1 of 25 models (all except the two patch CNNs) under five ways of filling the
missing holdout values (September for 40.6% of pixels; May for all pixels), with no
retraining. The question: is the corrected ranking an artifact of choosing the training median,
or does any sensible fill give the same answer?

## Methods

| Fill | Value used for a missing month |
|---|---|
| `zero` | 0 (published rule for the raw-sequence models) |
| `self_median` | the holdout's own column median (published rule for pixel TabNet) |
| `train_mean` | training-tile column mean |
| `train_median` | training-tile column median (the corrected re-analysis) |
| `interp` | linear interpolation between the neighbouring months of the same pixel and band |
| `published` (classical only) | no fill: xr_fresh features computed on the shortened series |

Months never observed in training (May) get the training value 0 under every method. For the
xr_fresh classical models the raw series is filled first and the 174 features are rebuilt
(`experiments/holdout_fill_check/rebuild_xrfresh.py --variants fill_*`), with May left missing as
in training.

Scripts: `../fill_sweep.py` (raw-input models; `--group deep` in deep_field, `--group ts` in
ts_baselines), `../make_filled_variants.py` and `run_classical.sh` (classical models),
`../score_fill_sweep.py` (scoring). Predictions in `predictions/`, results in `results/`.

Reuse: for the raw-input models, the published-rule and `train_median` predictions are copied
from `experiments/holdout_fill_check/predictions/` (`{key}_orig.csv`, `{key}_trainmed.csv`);
`tests/e2e_fill_pipeline.py` and a TimesNet check showed recomputing them gives identical files.

## Checks

- `interp` matches a row-by-row `np.interp` reference on random gap patterns
  (`tests/test_missing.py`); on the holdout every September-missing pixel gets the midpoint of
  its August and October values.
- The classical `train_median` series equals the re-analysis input file exactly.
- On fields where September was observed, the fill changes macro-F1 by at most 0.002 (TimesNet
  predictions depend slightly on the other samples in each batch).
- A first `interp` run interpolated May for pixel TabNet (the only model whose inputs include
  May), giving 0.282; `interp` now only interpolates months observed in training. The fix
  changed 946 TabNet fields and left the L-TAE `interp` predictions byte-identical.

## Results (`results/f1_by_fill.csv`, `zero_vs_median.csv`, `ranking.csv`)

1. **Any in-range fill gives the same answer.** Across `self_median`, `train_mean`,
   `train_median` and `interp`, the per-model spread in macro-F1 has median 0.014 (largest 0.036,
   pixel XGBoost), and the model ranking correlates 0.89 to 0.97 (Spearman) with the
   training-median ranking. The top five are pixel deep temporal models under every in-range fill.
2. **Zero fill hurts every family when applied to everyone.** Zero minus training median:
   raw-sequence models mean −0.075 (range −0.143 to +0.002), xr_fresh classical models mean −0.096
   (−0.117 to −0.043), every CI excluding zero except field L-TAE (−0.006) and field L-TAE-S
   (+0.002). Pixel TabNet loses 0.107. The classical models are not intrinsically robust to the
   gap; the published pipeline simply never zero-filled them.
3. **The published ranking flip came from the mixed treatment.** Spearman correlation with the
   fair (training-median) ranking:

   | Treatment | Spearman | Top five |
   |---|---|---|
   | As published: raw models zero-filled, TabNet holdout median, classical models unfilled | 0.56 | TabNet px, L-TAE-S fd, L-TAE-S px, L-TAE fd, logistic regression |
   | Everyone zero-filled | 0.74 | L-TAE-S fd, L-TAE-S px, L-TAE fd, Transformer px, TempCNN px |
   | Everyone training median | 1.00 | CNN-BiLSTM px, TempCNN px, L-TAE px, L-TAE-S px, Transformer px |

4. **What does resist zero fill**: field-level L-TAE and L-TAE-S (no loss) and the sparse gate at
   pixel level (L-TAE-S −0.044 against −0.113 for L-TAE). These are the robustness results the
   month-dropout control (to-do) has to be compared against.

Not covered: the two patch CNNs (separate patch pipeline) and deletion of the missing step for
L-TAE and the Transformer (needs an attention mask; listed in the to-do).
