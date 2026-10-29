#!/usr/bin/env bash
# Stoian et al. 2019 audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/stoian/01_digitise_fig19.py
python -W ignore experiments/external_audits/stoian/02_transcribe_table1.py
python -W ignore experiments/external_audits/stoian/03_scl_clear_counts.py
python -W ignore experiments/external_audits/stoian/04_l1c_effective_clear.py
python -W ignore experiments/external_audits/stoian/05_summarise.py
