#!/usr/bin/env bash
# Masolele et al. 2021 audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/masolele/01_points.py
python -W ignore experiments/external_audits/masolele/02_window_counts.py
python -W ignore experiments/external_audits/masolele/03_summarise.py
