#!/usr/bin/env bash
# Paired low-data sweep: 5, 10 and 15 percent of training fields, 10 draws each, every model on the
# identical fields (PAIRED_FID_FILE = splits.json, see make_splits.py), scored on the
# training-median-filled holdout. One training seed per draw, as in the earlier curves.
# Results go to results/<group>.csv in this folder (never into the unpaired sweep files).
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
cd ../../..   # experiments/
DF=~/miniconda3/envs/deep_field/bin/python3
ML=~/miniconda3/envs/ml_field/bin/python
export PAIRED_FID_FILE="$HERE/splits.json"
FR="0.05 0.10 0.15"
D10="42 101 202 303 404 505 606 707 808 909"
RES="$HERE/results"; LOG="$HERE/logs"; mkdir -p "$RES" "$LOG"
retry() { for i in 1 2 3; do "$@" && return 0; echo "RETRY $i: $*"; sleep 60; done; echo "GAVE UP: $*"; }
wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 1500 ]; do sleep 120; done; }
echo "start $(date)  PAIRED_FID_FILE=$PAIRED_FID_FILE"
wait_gpu
retry $ML holdout_fill_check/run_sweep.py --script seed_sweep_classical --variant trainmed -- \
  --models xgboost_field base_lr_pixel --fractions $FR --subsample-seeds $D10 --out "$RES/classical.csv" > "$LOG/classical.log" 2>&1 && echo "done classical $(date)"
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep_tabnet --variant trainmed -- \
  --fractions $FR --subsample-seeds $D10 --out "$RES/tabnet.csv" > "$LOG/tabnet.log" 2>&1 && echo "done tabnet $(date)"
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel ltae_sparse_pixel \
  --fractions $FR --subsample-seeds $D10 --n-seeds 1 --out "$RES/ltae.csv" > "$LOG/ltae.log" 2>&1 && echo "done ltae $(date)"
for bb in tempcnn cnn_bilstm; do
  retry $DF sparse_gate_backbones/sweep_backbones.py --backbone $bb --fractions $FR --draws $D10 --out "$RES/$bb.csv" > "$LOG/$bb.log" 2>&1 && echo "done $bb $(date)"
done
echo "PAIRED SWEEP DONE $(date)"
