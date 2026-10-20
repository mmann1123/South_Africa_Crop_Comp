#!/usr/bin/env bash
# Data-efficiency sweep scored on the September-filled holdout (see run_sweep.py).
# Same models, fractions and draws as experiments/field_reduction/results/seed_sweep_var.csv:
# fractions 0.25/0.50/0.75 x draws 42,101,202,303,404, plus the full training set (draw 42).
set -euo pipefail
cd "$(dirname "$0")"
ML=~/miniconda3/envs/ml_field/bin/python
DF=~/miniconda3/envs/deep_field/bin/python
DRAWS="42 101 202 303 404"
for v in orig trainmed; do
  $ML run_sweep.py --script seed_sweep_classical --variant $v -- --fractions 0.25 0.5 0.75 --subsample-seeds $DRAWS --out results/sweep_classical_$v.csv
  $ML run_sweep.py --script seed_sweep_classical --variant $v -- --fractions 1.0 --subsample-seeds 42 --out results/sweep_classical_$v.csv
done
$DF run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel ltae_sparse_pixel --fractions 0.25 0.5 0.75 --subsample-seeds $DRAWS --n-seeds 1 --out results/sweep_deep_trainmed.csv
$DF run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel ltae_sparse_pixel --fractions 1.0 --subsample-seeds 42 --n-seeds 1 --out results/sweep_deep_trainmed.csv
$DF run_sweep.py --script seed_sweep_tabnet --variant trainmed -- --fractions 0.25 0.5 0.75 --subsample-seeds $DRAWS --out results/sweep_tabnet_trainmed.csv
$DF run_sweep.py --script seed_sweep_tabnet --variant trainmed -- --fractions 1.0 --subsample-seeds 42 --out results/sweep_tabnet_trainmed.csv
echo "ALL SWEEPS DONE"
