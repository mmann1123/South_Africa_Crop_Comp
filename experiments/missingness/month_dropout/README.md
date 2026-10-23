# Month-dropout control for the sparse gate

Does the sparse gate's robustness to a zero-filled September come from sparsity, or would any model
that sees missing months during training be just as robust? A 2 x 2 of month dropout x gate on three
backbones, scored on the holdout with September filled by zero and by the training median.

Dropout: during training only, each month of each pixel is replaced with probability 0.2 by the
scaled raw zero (all six bands; `deep_learn/src/month_dropout.py`). Months are drawn at random;
September is not targeted, since that would use knowledge of the holdout.

| Backbone | dense, no dropout | gated, no dropout | dense + dropout | gated + dropout |
|---|---|---|---|---|
| TempCNN | `tempcnn_none` (harness control) | `tempcnn_sparsemax` | `tempcnn_none_md0.2` | `tempcnn_sparsemax_md0.2` |
| CNN-BiLSTM | `cnn_bilstm_none` (harness control) | `cnn_bilstm_sparsemax` | `cnn_bilstm_none_md0.2` | `cnn_bilstm_sparsemax_md0.2` |
| L-TAE | published L-TAE | published L-TAE-S | `models/ltae_pixel_md0.2` | `models/ltae_sparse_pixel_md0.2` |

TempCNN and CNN-BiLSTM arms are trained by `experiments/sparse_gate_backbones/train.py` (5 seeds,
published recipes; predictions in `../../sparse_gate_backbones/predictions/`). The L-TAE arms use
`experiments/field_reduction/train_ltae{,_sparse}_pixel.py --month-dropout 0.2` (predictions in
`predictions/`). Gated arms use the sparsemax gate on embedded channels.

Run: `bash run.sh` (about 24 GPU hours; waits for an idle GPU), then
`python score.py` in the deep_field env, which writes `results/cells.csv` and `results/contrasts.csv`.
The first launch ran out of GPU memory because `@torch.no_grad()` had slipped off the evaluation
helper; its logs are kept as `*_failed_nograd_bug.log` and `tests/test_train_helpers.py` guards it.

## Results (2026-10-23)

Holdout macro-F1 (2,415 fields). Paired field bootstrap, B = 10,000, seed 42.

| Backbone | Fill | dense | gated | dense + dropout | gated + dropout |
|---|---|---|---|---|---|
| TempCNN | zero | 0.568 | 0.569 | 0.653 | 0.655 |
| TempCNN | training median | 0.653 | 0.654 | 0.647 | 0.653 |
| CNN-BiLSTM | zero | 0.495 | 0.576 | 0.657 | 0.650 |
| CNN-BiLSTM | training median | 0.642 | 0.638 | 0.647 | 0.644 |
| L-TAE | zero | 0.539 | 0.599 | 0.656 | 0.655 |
| L-TAE | training median | 0.652 | 0.643 | 0.664 | 0.657 |

Under zero fill, dropout raises every arm significantly: by +0.085 (TempCNN), +0.162 (CNN-BiLSTM)
and +0.117 (L-TAE) for the dense models, all with 95% CIs above zero and McNemar p < 0.001. The gate
helps without dropout where the dense model is hurt (CNN-BiLSTM +0.081, L-TAE +0.059; TempCNN +0.001,
n.s.), but once dropout is used it adds nothing (−0.007, −0.002, +0.001, all n.s.). The interaction
is significantly negative for CNN-BiLSTM (−0.088 [−0.114, −0.062]) and L-TAE (−0.061 [−0.086,
−0.035]): the gate is a partial substitute for dropout, and dropout recovers more.

Under the training-median fill no bootstrap interval excludes zero. The one hint of an effect is a
small dropout gain for dense L-TAE (+0.012, CI [−0.002, 0.026]; McNemar p = 0.003 on field
correctness). Dropout costs nothing when the input is
sensibly filled; dense L-TAE with dropout (0.664) is the best single model in the study.

Harness check: the retrained dense TempCNN matches the published one (−0.006 zero, −0.002
training median, n.s.). The retrained dense CNN-BiLSTM is 0.014 to 0.018 below the published model
(bootstrap CI just excludes zero, McNemar n.s.), so CNN-BiLSTM contrasts use the harness control,
not the published model.

Reading: the sparse gate's robustness is real but is a weaker form of what the standard remedy,
training with missing months, already provides. Missingness robustness is a property of training
exposure, not of sparse architectures.
