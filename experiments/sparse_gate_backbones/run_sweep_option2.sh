#!/usr/bin/env bash
# Option 2 (2026-10-17): data-efficiency sweep on the September-filled holdout for
#  - pixel TempCNN and CNN-BiLSTM, dense and sparse-gated (sweep_backbones.py -> results_sweep.csv)
#  - L-TAE and L-TAE-S at the new low fractions (holdout_fill_check/run_sweep.py)
# Low fractions 0.10/0.15 use 10 draws; 0.25/0.50/0.75 use the original 5 draws; 1.0 one draw.
# Each step is retried up to 3 times; finished rows are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")"
PY=~/miniconda3/envs/deep_field/bin/python3
D10="42 101 202 303 404 505 606 707 808 909"
D5="42 101 202 303 404"
retry() { for i in 1 2 3; do "$@" && return 0; echo "RETRY $i: $*"; done; echo "GAVE UP: $*"; }
for bb in tempcnn cnn_bilstm; do
  retry $PY sweep_backbones.py --backbone $bb --fractions 0.10 0.15 --draws $D10 >> logs_sweep_$bb.log 2>&1
done
retry $PY ../holdout_fill_check/run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel ltae_sparse_pixel \
  --fractions 0.10 0.15 --subsample-seeds $D10 --n-seeds 1 --out results/sweep_deep_trainmed.csv >> logs_sweep_ltae_low.log 2>&1
for bb in tempcnn cnn_bilstm; do
  retry $PY sweep_backbones.py --backbone $bb --fractions 0.25 0.5 0.75 --draws $D5 >> logs_sweep_$bb.log 2>&1
  retry $PY sweep_backbones.py --backbone $bb --fractions 1.0 --draws 42 >> logs_sweep_$bb.log 2>&1
done
echo "OPTION 2 SWEEP DONE"
