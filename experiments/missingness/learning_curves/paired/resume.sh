#!/usr/bin/env bash
# Resume of run.sh after the first launch stopped on 2026-11-09 (classical done, TabNet 5 of 30
# rows). Same settings; TabNet's finished 5 percent draws (42 101 202 303 404) are skipped.
# Launch detached: setsid nohup bash resume.sh > resume.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
cd ../../..   # experiments/
DF=~/miniconda3/envs/deep_field/bin/python3
export PAIRED_FID_FILE="$HERE/splits.json"
FR="0.05 0.10 0.15"
D10="42 101 202 303 404 505 606 707 808 909"
RES="$HERE/results"; LOG="$HERE/logs"; mkdir -p "$RES" "$LOG"
retry() { for i in 1 2 3; do "$@" && return 0; echo "RETRY $i: $*"; sleep 60; done; echo "GAVE UP: $*"; }
wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 1500 ]; do sleep 120; done; }
echo "resume $(date)  PAIRED_FID_FILE=$PAIRED_FID_FILE"
wait_gpu
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep_tabnet --variant trainmed -- \
  --fractions 0.05 --subsample-seeds 505 606 707 808 909 --out "$RES/tabnet.csv" >> "$LOG/tabnet.log" 2>&1 && echo "done tabnet 0.05 $(date)"
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep_tabnet --variant trainmed -- \
  --fractions 0.10 0.15 --subsample-seeds $D10 --out "$RES/tabnet.csv" >> "$LOG/tabnet.log" 2>&1 && echo "done tabnet $(date)"
retry $DF holdout_fill_check/run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel ltae_sparse_pixel \
  --fractions $FR --subsample-seeds $D10 --n-seeds 1 --out "$RES/ltae.csv" > "$LOG/ltae.log" 2>&1 && echo "done ltae $(date)"
for bb in tempcnn cnn_bilstm; do
  retry $DF sparse_gate_backbones/sweep_backbones.py --backbone $bb --fractions $FR --draws $D10 --out "$RES/$bb.csv" > "$LOG/$bb.log" 2>&1 && echo "done $bb $(date)"
done
echo "PAIRED SWEEP DONE $(date)"
