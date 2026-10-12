#!/usr/bin/env bash
# Retrain every completed variant with checked MiniRocket features (features.checked_transform)
# into *_verify, and require every per-seed holdout CSV to be byte-identical to the
# original run. Also runs pixel MiniRocket-softmax for the first time (with checks).
# Writes results/verification.txt. Exit status is nonzero if any step fails.
set -uo pipefail
cd "$(dirname "$0")"
PY="$(conda info --base)/envs/ts_baselines/bin/python"
export PYTORCH_ALLOC_CONF=expandable_segments:True
fail=0
run() { echo "[$(date +%H:%M)] $*"; "$PY" "$@" || { echo "FAILED: $*"; fail=1; }; }

run train_gated.py --level pixel --gate softmax > logs/gated_softmax_pixel.log 2>&1
for lv in field pixel; do
    run train_l1.py --level "$lv" --out-tag verify --force > "logs/l1_${lv}_verify.log" 2>&1
done
run train_gated.py --level field --gate sparsemax --out-tag verify --force > logs/gated_sparsemax_field_verify.log 2>&1
run train_gated.py --level field --gate softmax --out-tag verify --force > logs/gated_softmax_field_verify.log 2>&1
run train_gated.py --level pixel --gate sparsemax --out-tag verify --force > logs/gated_sparsemax_pixel_verify.log 2>&1

{
  echo "Verification $(date -Iseconds): original runs vs retrains with checked features"
  for m in minirocket_l1_field minirocket_l1_pixel minirocket_s_field minirocket_softmax_field minirocket_s_pixel; do
      for s in 42 101 202 303 404; do
          a="results/predictions/${m}_seed${s}.csv"; b="results/predictions/${m}_verify_seed${s}.csv"
          if cmp -s "$a" "$b"; then echo "${m} seed ${s}: IDENTICAL"; else echo "${m} seed ${s}: DIFFERENT"; fail=1; fi
      done
  done
  echo "transform mismatches logged: $(cat results/transform_mismatches.log 2>/dev/null | wc -l)"
} | tee results/verification.txt
exit $fail
