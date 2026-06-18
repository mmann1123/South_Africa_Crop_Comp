# L-TAE-S full-data (fraction = 1.0) results

Metrics scored with `score_oos.py` (mirrors `out_of_sample/compare_predictions.py`:
weighted/macro F1, Cohen kappa, one-vs-rest hard-label cross-entropy). In-region F1
is the field-wise in-region test split from each run's `metadata.json`.
Δ = ST F1_macro − in-region F1_macro.

| Variant        | In-reg F1m | Δ      | ST F1m | ST κ   | ST wF1 | ST Xent | OOS fields |
|----------------|-----------:|-------:|-------:|-------:|-------:|--------:|-----------:|
| L-TAE-S field  | 0.6431     | −0.04  | 0.6012 | 0.5155 | 0.6938 | 4.7720  | 2415       |
| L-TAE-S pixel  | _pending_  | _pend_ | _pend_ | _pend_ | _pend_ | _pend_  | _pending_  |

- Field run: trained in 92 s, early-stopped ~epoch 39/seed, artifacts in
  `models/ltae_sparse_field/frac_1.00/`, predictions in
  `results/predictions/ltae_sparse_field_frac_1.00.csv`.
- Pixel run: launched 11:34, ~100 s/epoch × 5 seeds, log `ltae_s_full.log`.
  CRASHED ~15:58 on a transient CUDA illegal-memory-access during seed 303
  (seeds 42/101/202 trained & saved OK; seed 303 partial, 404 not started; scaler/
  encoder/metadata not yet written). Recovered via `resume_ltae_sparse_pixel.py`
  (relaunched 21:40): reuses seeds 42/101/202, retrains 303/404, regenerates
  artifacts + 5-seed in-region metadata, then predicts. Log `ltae_s_full_resume.log`.

## Pending table edits (do when pixel run finishes)
- Table 3 (tab:gap): add two rows — `L-TAE-S | pixel | raw` and `L-TAE-S | field | raw`.
- Table 4 (tab:pixfield): add `L-TAE-S` row with pixel (ST F1m, Δ) and field
  (ST F1m=0.60, Δ=−0.04) columns, Field input = raw-avg.
- Re-check per-column bolding in Table 3 after inserting rows.

## FINAL (both variants complete)
| Variant        | In-reg F1m | Δ      | ST F1m | ST κ   | ST wF1 | ST Xent |
|----------------|-----------:|-------:|-------:|-------:|-------:|--------:|
| L-TAE-S pixel  | 0.7722     | -0.17  | 0.5987 | 0.5274 | 0.7032 | 4.6319  |
| L-TAE-S field  | 0.6431     | -0.04  | 0.6012 | 0.5155 | 0.6938 | 4.7720  |
