#!/usr/bin/env bash
# Fill-value sweep: deep_field models first, then TimesNet / MiniRocket (ts_baselines).
cd "$(dirname "$0")/../../.."
~/miniconda3/envs/deep_field/bin/python3 experiments/missingness/fill_sweep.py --group deep
echo "deep group exit $?"
~/miniconda3/envs/ts_baselines/bin/python experiments/missingness/fill_sweep.py --group ts
echo "ts group exit $?"
echo "SWEEP DONE $(date)"
