#!/usr/bin/env bash
# Month-dropout control for the sparse gate (2 x 2: dropout x gate) on TempCNN, CNN-BiLSTM and
# L-TAE, 5 seeds each with the published recipes. Dropout rate 0.2, whole months replaced by the
# scaled raw zero (month_dropout.py). Existing arms reused: published dense models,
# sparse_gate_backbones/{tempcnn,cnn_bilstm}_sparsemax, published L-TAE / L-TAE-S.
# "none" without dropout is retrained in this harness as a control for harness differences.
# Waits until the GPU has been idle (<1 GB) for 10 minutes; each step retried up to 3 times.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=~/miniconda3/envs/deep_field/bin/python3
SG=experiments/sparse_gate_backbones
FR=experiments/field_reduction
MD=experiments/missingness/month_dropout
idle=0
while [ $idle -lt 20 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$used" ] && [ "$used" -lt 1000 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 30
done
echo "GPU idle; starting $(date)"
retry() { for i in 1 2 3; do "$@" && return 0; echo "RETRY $i: $*"; sleep 60; done; echo "GAVE UP: $*"; return 1; }
sg() { [ -f $SG/predictions/$1_trainmed.csv ] && { echo "skip $1"; return 0; }; retry $PY $SG/train.py "${@:2}" > $MD/log_$1.log 2>&1 && echo "done $1"; }

# smoke test: 1 seed, 1 epoch, then the checkpoint must load as a plain TempCNN
retry $PY $SG/train.py --backbone tempcnn --gate none --month-dropout 0.2 --seeds 42 --max-epochs 1 --tag smoke > $MD/log_smoke.log 2>&1 || exit 1
$PY -c "
import sys, torch; sys.path.insert(0, '$SG'); import train as tr
sd = torch.load('$SG/models/tempcnn_none_md0.2_smoke/seed_42.pt', map_location='cpu')
ref = tr.TempCNN(tr.C, 5).state_dict()
assert list(sd) == list(ref) and all(sd[k].shape == ref[k].shape for k in ref), 'checkpoint keys differ'
print('smoke ok: checkpoint loads as a plain TempCNN')" || exit 1

# cheap arms first
sg tempcnn_none            --backbone tempcnn    --gate none
sg tempcnn_none_md0.2      --backbone tempcnn    --gate none      --month-dropout 0.2
sg cnn_bilstm_none         --backbone cnn_bilstm --gate none
sg cnn_bilstm_none_md0.2   --backbone cnn_bilstm --gate none      --month-dropout 0.2
for m in ltae ltae_sparse; do
  d=$MD/models/${m}_pixel_md0.2
  [ -f $d/label_encoder.joblib ] || retry $PY $FR/train_${m}_pixel.py --fraction 1.0 --output-dir $d --month-dropout 0.2 > $MD/log_${m}_md0.2.log 2>&1
done
# gated arms (about 5 h each)
sg tempcnn_sparsemax_md0.2    --backbone tempcnn    --gate sparsemax --month-dropout 0.2
sg cnn_bilstm_sparsemax_md0.2 --backbone cnn_bilstm --gate sparsemax --month-dropout 0.2

# L-TAE holdout predictions under zero and training-median fill
$PY - <<PYEOF
import os, sys
for p in ["deep_learn/src", "$FR"]:
    sys.path.insert(0, p)
import predict_oos as po
for m, fn in [("ltae", po.predict_ltae_pixel), ("ltae_sparse", po.predict_ltae_sparse_pixel)]:
    d = "$MD/models/" + m + "_pixel_md0.2"
    for fill, tag in [("zero", "orig"), ("train_median", "trainmed")]:
        os.environ["HOLDOUT_FILL"] = fill
        out = "$MD/predictions/" + m + "_pixel_md0.2_" + tag + ".csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fn(d, out)
        print("wrote", out, flush=True)
PYEOF
echo "MONTH DROPOUT DONE $(date)"
