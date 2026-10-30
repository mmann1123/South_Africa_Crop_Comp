#!/usr/bin/env bash
# Verhulst et al. 2024 audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/verhulst/01_digitise_fig7.py
python -W ignore experiments/external_audits/verhulst/02_forest_points.py
python -W ignore experiments/external_audits/verhulst/03_valid_dekads.py
python -W ignore experiments/external_audits/verhulst/04_summarise_gaps.py
python -W ignore experiments/external_audits/verhulst/05_correlate_oa.py
