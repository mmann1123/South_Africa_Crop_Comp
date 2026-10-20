#!/usr/bin/env bash
# Sparse-gate experiment: each backbone with the sparsemax gate and the softmax (dense) control,
# 5 seeds each with the backbone's published recipe. Variants whose holdout predictions already
# exist are skipped; a variant that crashes is retried once (a native segfault stopped the first
# tempcnn_softmax run on 2026-10-16; no Python error was raised).
set -uo pipefail
cd "$(dirname "$0")"
PY=~/miniconda3/envs/deep_field/bin/python3
for bb in tempcnn cnn_bilstm; do
  for g in sparsemax softmax; do
    if [ -f predictions/${bb}_${g}_trainmed.csv ]; then echo "skip $bb $g (done)"; continue; fi
    for attempt in 1 2; do
      if $PY train.py --backbone $bb --gate $g > logs_${bb}_${g}.log 2>&1; then echo "done $bb $g"; break; fi
      echo "FAILED $bb $g attempt $attempt (exit $?)"; cp logs_${bb}_${g}.log logs_${bb}_${g}_failed_$attempt.log
    done
  done
done
echo "ALL GATE RUNS DONE"
