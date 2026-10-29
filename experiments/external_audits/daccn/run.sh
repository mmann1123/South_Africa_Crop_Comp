#!/usr/bin/env bash
# DACCN audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python experiments/external_audits/daccn/01_regions.py
python experiments/external_audits/daccn/02_run_counters.py
python experiments/external_audits/daccn/03_summarise.py
python experiments/external_audits/daccn/04_snow_counters.py
python experiments/external_audits/daccn/05_snow_summarise.py
