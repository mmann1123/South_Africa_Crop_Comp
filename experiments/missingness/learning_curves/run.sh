#!/usr/bin/env bash
# Learning curves down to 5% of training fields on the training-median-filled holdout, plus pixel
# TabNet at 10 and 15% (it only had 25 to 100%). 10 draws per low fraction, as in the Option 2
# sweep (sparse_gate_backbones/run_sweep_option2.sh). Rows are appended to the existing sweep
# files, so every model's full curve stays in one place:
#   sparse_gate_backbones/results_sweep.csv              TempCNN, CNN-BiLSTM (dense, sparsemax)
#   holdout_fill_check/results/sweep_deep_trainmed.csv     L-TAE, L-TAE-S
#   holdout_fill_check/results/sweep_tabnet_trainmed.csv   pixel TabNet
#   holdout_fill_check/results/sweep_classical_trainmed.csv  field XGBoost, pixel logistic regression
# At 5% the stratified draw keeps 132 of 2,653 training fields, at least 15 per class.
set -uo pipefail
cd "$(dirname "$0")/../.."
DF=~/miniconda3/envs/deep_field/bin/python3
ML=~/miniconda3/envs/ml_field/bin/python
D10="42 101 202 303 404 505 606 707 808 909"
LOG=missingness/learning_curves
retry() { for i in 1 2 3; do "$@" && return 0; echo "RETRY $i: $*"; sleep 30; done; echo "GAVE UP: $*"; }
echo "start $(date)"
retry $ML holdout_fill_check/run_sweep.py --script seed_sweep_classical --variant trainmed -- \
  --fractions 0.05 --subsample-seeds $D10 --out results/sweep_classical_trainmed.csv > $LOG/log_classical.log 2>&1 && echo "done classical"
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel ltae_sparse_pixel \
  --fractions 0.05 --subsample-seeds $D10 --n-seeds 1 --out results/sweep_deep_trainmed.csv > $LOG/log_ltae.log 2>&1 && echo "done ltae"
for bb in tempcnn cnn_bilstm; do
  retry $DF sparse_gate_backbones/sweep_backbones.py --backbone $bb --fractions 0.05 --draws $D10 > $LOG/log_$bb.log 2>&1 && echo "done $bb"
done
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep_tabnet --variant trainmed -- \
  --fractions 0.05 0.10 0.15 --subsample-seeds $D10 --out results/sweep_tabnet_trainmed.csv > $LOG/log_tabnet.log 2>&1 && echo "done tabnet"
echo "LEARNING CURVES DONE $(date)"
