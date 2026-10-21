#!/usr/bin/env bash
# Classical (xr_fresh) models under each fill: fill the raw series, rebuild the 174 xr_fresh
# features, combine to field level, run the published classical inference and xr_fresh TabNet.
# train_median is the existing "trainmed" variant; "orig" (no fill, NaN-aware features) is the
# published rule for these models.
set -uo pipefail
cd "$(dirname "$0")/../../.."
V="fill_zero fill_self_median fill_train_mean fill_interp"
FC=experiments/holdout_fill_check
OUT=experiments/missingness/fill_sweep/predictions
JAX_PLATFORMS=cpu ~/miniconda3/envs/xr_fresh/bin/python $FC/rebuild_xrfresh.py --variants $V || { echo "rebuild failed"; exit 1; }
~/miniconda3/envs/deep_field/bin/python3 $FC/combine_xrfresh.py --variants $V || { echo "combine failed"; exit 1; }
~/miniconda3/envs/ml_field/bin/python $FC/run_classical.py --variants $V --pred-dir $OUT || echo "classical exit $?"
~/miniconda3/envs/deep_field/bin/python3 $FC/run_deep_field.py --models tabnet_field --variants $V || echo "tabnet_field exit $?"
for v in $V; do cp $FC/predictions/tabnet_field_$v.csv $OUT/; done
echo "CLASSICAL DONE $(date)"
