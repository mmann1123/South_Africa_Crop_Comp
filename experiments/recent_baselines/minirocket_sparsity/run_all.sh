#!/usr/bin/env bash
# MiniRocket sparsity experiment: train 3 variants x 2 levels, reproducibility check,
# score, report, and verify that no published or completed-baseline file changed.
# Resumable: finished seeds are skipped.
#
#   bash experiments/recent_baselines/minirocket_sparsity/run_all.sh   (from the repo root)
set -euo pipefail
cd "$(dirname "$0")"
PY="$(conda info --base)/envs/ts_baselines/bin/python"
"$PY" -c "import aeon; assert aeon.__version__ == '1.6.0', aeon.__version__"
mkdir -p logs results

[ -f results/guard_before.txt ] || "$PY" guard_sparsity.py snapshot results/guard_before.txt

for level in field pixel; do
    for gate in sparsemax softmax; do
        "$PY" train_gated.py --level "$level" --gate "$gate" > "logs/gated_${gate}_${level}.log" 2>&1
    done
    "$PY" train_l1.py --level "$level" > "logs/l1_${level}.log" 2>&1
done

# Reproducibility: retrain field seed 42 of each variant from scratch; holdout CSV must be byte-identical.
{
  echo "Reproducibility check $(date -Iseconds), git $(git rev-parse --short HEAD)"
  for gate in sparsemax softmax; do
      "$PY" train_gated.py --level field --gate "$gate" --seeds 42 --out-tag repro --force > "logs/gated_${gate}_field_repro.log" 2>&1
  done
  "$PY" train_l1.py --level field --seeds 42 --out-tag repro --force > logs/l1_field_repro.log 2>&1
  for m in minirocket_s minirocket_softmax minirocket_l1; do
      if cmp -s "results/predictions/${m}_field_seed42.csv" "results/predictions/${m}_field_repro_seed42.csv"; then
          echo "${m} field seed 42: IDENTICAL"
      else
          echo "${m} field seed 42: DIFFERENT"
      fi
  done
} | tee results/reproducibility_check.txt

"$PY" score_sparsity.py 2>&1 | tee logs/score.log
"$PY" guard_sparsity.py snapshot results/guard_after.txt
"$PY" guard_sparsity.py compare results/guard_before.txt results/guard_after.txt | tee results/guard_check.txt
"$PY" make_report.py
