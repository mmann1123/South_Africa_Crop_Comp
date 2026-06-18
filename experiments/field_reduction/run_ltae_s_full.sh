#!/bin/bash
# Queue + run the L-TAE-S full-data (fraction 1.0) training and OOS prediction.
# Waits for the currently-running GPU job to finish before starting, so the two
# jobs never contend for GPU memory.
set -u

PYTHON=/home/mmann1123/miniconda3/envs/deep_field/bin/python3
EXP=/home/mmann1123/Documents/github/South_Africa_Crop_Comp/experiments/field_reduction
OUT=$EXP/models/ltae_sparse_pixel/frac_1.00
WATCH_PID=1872269          # the 3D CNN (Multi_Channel_CNN.py) job to wait on
MIN_FREE_MIB=9000          # require this much free GPU memory before launching
LOG=$EXP/ltae_s_full.log

exec > "$LOG" 2>&1

echo "[$(date)] === L-TAE-S full-data (fraction=1.0) queued run ==="
echo "[$(date)] Waiting for GPU job PID $WATCH_PID to finish..."
while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] PID $WATCH_PID has exited."

echo "[$(date)] Waiting for >= ${MIN_FREE_MIB} MiB free GPU memory..."
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${FREE:-0}" -ge "$MIN_FREE_MIB" ]; then break; fi
  sleep 60
done
echo "[$(date)] GPU free=${FREE} MiB. Starting training."

cd "$EXP" || { echo "cd failed"; exit 1; }

echo "[$(date)] >>> TRAINING: train_ltae_sparse_pixel.py --fraction 1.0"
"$PYTHON" train_ltae_sparse_pixel.py --fraction 1.0 --output-dir "$OUT"
TRAIN_RC=$?
echo "[$(date)] Training exit code: $TRAIN_RC"

if [ "$TRAIN_RC" -eq 0 ]; then
  echo "[$(date)] >>> OOS PREDICTION: predict_oos.py --models ltae_sparse_pixel --fractions 1.0"
  "$PYTHON" predict_oos.py --models ltae_sparse_pixel --fractions 1.0
  echo "[$(date)] Prediction exit code: $?"
else
  echo "[$(date)] Skipping prediction because training failed."
fi

echo "[$(date)] === DONE ==="
