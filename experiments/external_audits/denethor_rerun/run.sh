#!/usr/bin/env bash
# DENETHOR causal re-run, in order. Run from anywhere; every step resumes from its outputs.
# Env `deep_field` (PyTorch, GPU, sklearn). Data root: /mnt/bigdrive/external_audits/denethor_rerun
# (labels/ from the descriptive audit ../denethor/, authors' clone lukaskondmann/DENETHOR at 55573a8 in repo/).
set -euo pipefail
cd "$(dirname "$0")/../../.."
DF=${DF_PY:-/home/mmann1123/miniconda3/envs/deep_field/bin/python}
D=experiments/external_audits/denethor_rerun

$DF $D/01_extract_pf.py                                        # Planet Fusion SR + QA at 128 fixed pixels per field, 2 x 365 days (about 240 GB streamed)
$DF $D/02_train_eval.py --conds A0 A1 A2 --seeds 0 1 2 3 4 5 6 7 8 9   # PseLTae, about 2.2 min per run on an RTX 3080 Ti
$DF $D/03_s2_field_series.py                                   # field-mean Sentinel-2 series, 4 tile-years (BANDS.npy 19.9 GB each, deleted after)
$DF $D/04_rf.py                                                # random-forest conditions
$DF $D/05_summarise.py                                         # tables and figure
