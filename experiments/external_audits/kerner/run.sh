#!/usr/bin/env bash
# Kerner audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python experiments/external_audits/kerner/01_fetch_zenodo.py
python experiments/external_audits/kerner/02_hls_acquisitions.py
python experiments/external_audits/kerner/03_make_sample.py
python experiments/external_audits/kerner/04_ee_pixel_states.py
python experiments/external_audits/kerner/05_summarise.py
