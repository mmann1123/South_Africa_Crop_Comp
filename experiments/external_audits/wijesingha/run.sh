#!/usr/bin/env bash
# Wijesingha et al. 2024 audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/wijesingha/01_notebook_outputs.py
python -W ignore experiments/external_audits/wijesingha/02_sample_cropland.py
python -W ignore experiments/external_audits/wijesingha/03_observation_counts.py
python -W ignore experiments/external_audits/wijesingha/04_summarise.py
