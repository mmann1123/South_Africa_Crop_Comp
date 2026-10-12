#!/usr/bin/env bash
# R2-Q3a recent baselines: train, check reproducibility, score, and verify that no
# published file changed. Resumable: finished seeds are skipped.
#
#   bash experiments/recent_baselines/run_all.sh          (from the repo root)
set -euo pipefail
cd "$(dirname "$0")"
PY="$(conda info --base)/envs/ts_baselines/bin/python"
[ -x "$PY" ] || { echo "ts_baselines env missing: run env/setup_env.sh" >&2; exit 1; }
"$PY" -c "import aeon; assert aeon.__version__ == '1.6.0', aeon.__version__"
mkdir -p logs results

[ -f results/guard_before.txt ] || "$PY" guard.py snapshot results/guard_before.txt

for level in field pixel; do
    "$PY" train_minirocket.py --level "$level" 2>&1 | tee "logs/minirocket_${level}.log"
    "$PY" train_timesnet.py   --level "$level" 2>&1 | tee "logs/timesnet_${level}.log"
done

# Reproducibility: retrain seed 42 of both field models from scratch and require a
# byte-identical holdout prediction file.
{
  echo "Reproducibility check $(date -Iseconds), git $(git rev-parse --short HEAD)"
  for m in minirocket timesnet; do
      "$PY" "train_${m}.py" --level field --seeds 42 --out-tag repro --force > "logs/${m}_field_repro.log" 2>&1
      if cmp -s "results/predictions/${m}_field_seed42.csv" "results/predictions/${m}_field_repro_seed42.csv"; then
          echo "${m} field seed 42: IDENTICAL"
      else
          echo "${m} field seed 42: DIFFERENT"
      fi
  done
} | tee results/reproducibility_check.txt

"$PY" score.py 2>&1 | tee logs/score.log
"$PY" make_report.py

"$PY" guard.py snapshot results/guard_after.txt
"$PY" guard.py compare results/guard_before.txt results/guard_after.txt | tee results/guard_check.txt
