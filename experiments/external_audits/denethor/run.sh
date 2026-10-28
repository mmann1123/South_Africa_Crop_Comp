#!/usr/bin/env bash
# DENETHOR audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python experiments/external_audits/denethor/01_fetch_masks.py
python experiments/external_audits/denethor/02_s2_clear_counts.py
python experiments/external_audits/denethor/03_pf_qa_synthetic.py
python experiments/external_audits/denethor/04_summarise.py
