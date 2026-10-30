#!/usr/bin/env bash
# Hao et al. 2020 audit, in order. Run from the repo root inside the `ee` conda env.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/hao/01_digitise_figs.py
python -W ignore experiments/external_audits/hao/02_points.py
python -W ignore experiments/external_audits/hao/03_ee_windows.py --stream hls
python -W ignore experiments/external_audits/hao/03_ee_windows.py --stream l8s2
python -W ignore experiments/external_audits/hao/04_summarise_windows.py
python -W ignore experiments/external_audits/hao/05_relate_oa.py
python -W ignore experiments/external_audits/hao/06_figure.py
