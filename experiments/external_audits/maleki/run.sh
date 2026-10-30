#!/usr/bin/env bash
# Maleki and Rahdari 2026 audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/maleki/01_site_geometry.py
python -W ignore experiments/external_audits/maleki/02_sample_validity.py
python -W ignore experiments/external_audits/maleki/03_interp_metrics.py
python -W ignore experiments/external_audits/maleki/04_summarise.py
