#!/usr/bin/env bash
# TimeMatch audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python experiments/external_audits/timematch/01_fetch_footprints.py
python experiments/external_audits/timematch/02_ee_cloud_by_date.py
python experiments/external_audits/timematch/03_summarise.py
