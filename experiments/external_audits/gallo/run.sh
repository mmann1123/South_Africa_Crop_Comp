#!/usr/bin/env bash
# Gallo et al. 2023 audit, in order. Run from the repo root inside the `ee` conda env.
# Step 01 crawls about 4.6 million Kaggle file names (about 4 hours, resumable); 03 and 05 are throttled downloads.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/gallo/01_list_kaggle.py
python -W ignore experiments/external_audits/gallo/02_date_counts.py
python -W ignore experiments/external_audits/gallo/03_footprints.py
python -W ignore experiments/external_audits/gallo/04_ee_cloud_by_date.py
python -W ignore experiments/external_audits/gallo/05_pixel_check.py
python -W ignore experiments/external_audits/gallo/06_summarise.py
