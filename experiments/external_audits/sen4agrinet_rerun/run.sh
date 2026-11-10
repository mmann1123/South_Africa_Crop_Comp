#!/usr/bin/env bash
# Sen4AgriNet PAD causal re-run, in order. Run from anywhere; every step resumes from its outputs.
# Env `deep_field` (PyTorch, GPU). Data root: /mnt/bigdrive/external_audits/sen4agrinet_rerun
# (authors' clone Orion-AI-Lab/S4A-Models at ae5a376 in repo/). Needs ../sen4agrinet/results from the descriptive audit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
DF=${DF_PY:-/home/mmann1123/miniconda3/envs/deep_field/bin/python}
D=experiments/external_audits/sen4agrinet_rerun

$DF $D/00_fetch_weights.py                    # 9 PAD checkpoints streamed out of the authors' Dropbox bundle
$DF $D/01_fetch_patches.py                    # test-split NetCDFs, Scenarios 1-3, HF revision 4eb4415 (2,820 files, 79 GB)
$DF $D/02_run_conditions.py                   # STEP 1 gate: published inputs only, must match the README F1 to 0.02
$DF $D/02_run_conditions.py --full --workers 16   # all conditions, Scenarios 2 and 3 (about 11 h on an RTX 3080 Ti)
$DF -W ignore $D/03_summarise.py
$DF $D/04_figure.py
