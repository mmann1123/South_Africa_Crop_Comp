# Sparse gate on the top deep backbones

## Question

Does the L-TAE-S sparse gate help the strongest deep temporal models once every model gets a
fair holdout input (missing September filled from training medians; see
`../holdout_fill_check/`)? And does it help when training data are scarce?

## Design

- **Backbones:** pixel TempCNN and pixel CNN-BiLSTM, each with its exact published recipe
  (data layout, scaling, split, seeds, optimizer, batch size, focal loss, early stopping).
- **Gate block** (`train.py`, mirrors `field_reduction/models_arch.SparseFeatureGate`): per
  time step a 6 → 128 embedding; 16 heads, each with a sparsemax mask over the 128 channels
  (BatchNorm logits times a relaxation prior, gamma = 1.5); each masked view projected to 8
  channels and the heads concatenated (128); then the unchanged backbone. The gate runs in fp32.
- **Control:** the identical network with softmax instead of sparsemax, to separate sparsity
  from the capacity the gate adds.
- **Holdout:** scored under both fills: training-median (fair) and zero (as published).

## Scripts

| Script | What it does |
|---|---|
| `train.py` | full-data 5-seed runs: `--backbone {tempcnn,cnn_bilstm} --gate {sparsemax,softmax}` |
| `score.py` | paired bootstrap (B = 10,000, seed 42) of gated vs. control vs. published dense → `results.csv` |
| `sweep_backbones.py` | data-efficiency sweep, dense and gated, identical subsamples (`field_reduction.subsample`), one training seed → `results_sweep.csv` |
| `run_all.sh`, `run_sweep_option2.sh`, `run_final_when_gpu_free.sh` | run order used, with retries |

Checkpoints (`models/`) are git-ignored. Run in the `deep_field` env.

## Results

**Full data, 5 seeds** (`results.csv`), holdout macro-F1:

| | Fill | Sparse gate | Softmax gate | Dense (published) |
|---|---|---|---|---|
| TempCNN | training median | 0.654 | 0.663 | 0.655 |
| TempCNN | zero | 0.569 | 0.561 | 0.574 |
| CNN-BiLSTM | training median | 0.638 | 0.621 | 0.656 |
| CNN-BiLSTM | zero | 0.576 | 0.543 | 0.513 |

TempCNN: no significant difference. CNN-BiLSTM: on clean input any gate lowers accuracy
(sparse −0.018, softmax −0.035 vs. dense); on the zero-filled input the sparse gate adds +0.063
[+0.040, +0.086] over dense and +0.034 over softmax. The gate's robust effect is robustness to
missing or corrupted inputs, not better transfer on complete data.

**Data efficiency** (`results_sweep.csv`, plus L-TAE/L-TAE-S and the classical models in
`../holdout_fill_check/results/sweep_*_trainmed.csv`): mean holdout macro-F1 over 10 draws
(10%, 15% of fields) or 5 draws (25 to 75%); 100% is one run.

| Model | 10% | 15% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| L-TAE | 0.554 | 0.568 | 0.591 | 0.636 | 0.638 | 0.647 |
| L-TAE-S | 0.557 | 0.565 | 0.612 | 0.636 | 0.644 | 0.652 |
| TempCNN | 0.547 | 0.581 | 0.607 | 0.621 | 0.633 | 0.647 |
| TempCNN + sparse gate | 0.565 | 0.590 | 0.608 | 0.634 | 0.639 | 0.645 |
| CNN-BiLSTM | 0.554 | 0.561 | 0.580 | 0.592 | 0.616 | 0.606 |
| CNN-BiLSTM + sparse gate | 0.543 | 0.557 | 0.575 | 0.592 | 0.596 | 0.603 |
| Logistic regression (pixel) | 0.547 | 0.563 | 0.574 | 0.599 | 0.606 | 0.605 |
| XGBoost (field) | 0.513 | 0.524 | 0.549 | 0.557 | 0.593 | 0.612 |
| TabNet (pixel) | | | 0.539 | 0.561 | 0.582 | 0.572 |

With 10% of fields every model except XGBoost is near 0.55; the deep temporal models pull
ahead as data grow. The gate has no consistent low-data effect: a small gain for TempCNN
(+0.017 at 10%, 8/10 draws), none for L-TAE, a small loss for CNN-BiLSTM.

## Run notes

- One native segfault (no Python error) stopped the first `tempcnn_softmax` run; the rerun
  completed (`logs_tempcnn_softmax_failed_*.log`).
- An automatic NVIDIA driver update (580.173 → 580.178) on 2026-10-19 blocked CUDA until a
  reboot; the CNN-BiLSTM 100% runs were completed afterwards. No finished run was affected.
