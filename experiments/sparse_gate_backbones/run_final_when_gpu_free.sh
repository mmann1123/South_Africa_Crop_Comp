#!/usr/bin/env bash
# Final GPU runs (2026-10-19), started only once the GPU has been idle (<1 GB used) for 10 minutes,
# so they do not compete with other GPU work:
#   1. XGBoost (field) at 10% and 15% of fields, 10 draws, both holdout feature sets
#   2. CNN-BiLSTM dense and sparse-gated at 100% (draw 42); finished rows are skipped
set -uo pipefail
cd "$(dirname "$0")"
DF=~/miniconda3/envs/deep_field/bin/python3
ML=~/miniconda3/envs/ml_field/bin/python
D10="42 101 202 303 404 505 606 707 808 909"
idle=0
while [ $idle -lt 20 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$used" ] && [ "$used" -lt 1000 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 30
done
echo "GPU idle since $(date -d '-10 min'); starting"
retry() { for i in 1 2 3; do "$@" && return 0; echo "RETRY $i: $*"; sleep 60; done; echo "GAVE UP: $*"; }
for v in orig trainmed; do
  retry $ML ../holdout_fill_check/run_sweep.py --script seed_sweep_classical --variant $v -- --models xgboost_field \
    --fractions 0.10 0.15 --subsample-seeds $D10 --out results/sweep_classical_$v.csv >> logs_final_xgb.log 2>&1
done
retry $DF sweep_backbones.py --backbone cnn_bilstm --fractions 1.0 --draws 42 >> logs_sweep_cnn_bilstm.log 2>&1
echo "FINAL RUNS DONE $(date)"
