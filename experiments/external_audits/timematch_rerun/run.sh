#!/usr/bin/env bash
# TimeMatch cloud-removal re-run, in order. Run from anywhere; steps are resumable.
# Env `ee` for Earth Engine and summaries, env `deep_field` (PyTorch, GPU) for model work.
# Data root: /mnt/bigdrive/external_audits/timematch_rerun (zip 74 GB, models zip, authors' clone at the pinned commit).
set -euo pipefail
cd "$(dirname "$0")/../../.."
EE=${EE_PY:-/home/mmann1123/miniconda3/envs/ee/bin/python}
DF=${DF_PY:-/home/mmann1123/miniconda3/envs/deep_field/bin/python}
D=experiments/external_audits/timematch_rerun
ROOT=/mnt/bigdrive/external_audits/timematch_rerun
export LD_LIBRARY_PATH=$(dirname "$EE")/../lib:${LD_LIBRARY_PATH:-}

# 0. inputs (manual, once): trained_models_and_results.zip -> $ROOT/models; git clone jnyborg/timematch -> $ROOT/repo (commit a652df9);
#    apply results/compat_fixes.patch to the clone; fetch timematch_data.zip with 00_parallel_download.py and extract
#    timematch_data/denmark/32VNH/2017 and timematch_data/france/30TXT/2017 to $ROOT/timematch_data/.
[ -f "$ROOT/timematch_data.zip" ] || $DF $D/00_parallel_download.py

$EE $D/01_prepare_meta.py                         # parcel tables, FR1 polygons (meta_fetch/ must hold the two tiles' meta/ folders)
$DF -I $D/03_reproduce.py                         # STEP 1 gate: authors' evaluation with released models
$EE $D/02_ee_parcel_cloud.py                      # Cloud Score+ per FR1 test parcel and date, block tables (Earth Engine)
$DF -I $D/04_extract_features.py                  # per parcel-date band medians and means
$DF -I $D/05_emulator.py                          # Cloud Score+ emulator for DK1 (no parcel polygons shipped)
for d in DK1_to_FR1 FR1_to_DK1; do
  $DF -I $D/06_run_conditions.py --direction $d --sweep -60 -48 -40 -32 -24 -16 -8 8 16 24 32 40 48 60
done
$EE $D/07_summarise.py
$EE $D/08_figure.py
LD_LIBRARY_PATH=$(dirname "$EE")/../lib $EE -m pytest experiments/external_audits/tests/test_timematch_rerun_lib.py -q
