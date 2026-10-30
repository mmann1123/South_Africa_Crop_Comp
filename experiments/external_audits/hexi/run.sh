#!/usr/bin/env bash
# Mai et al. 2025 (Hexi Corridor) audit, in order. Run from the repo root inside the `ee` conda env.
# Run 02 for one year at a time (Earth Engine's concurrency limit); it is resumable.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -W ignore experiments/external_audits/hexi/01_points.py
python -W ignore experiments/external_audits/hexi/02_weekly_counts.py
python -W ignore experiments/external_audits/hexi/03_qa60_images.py
python -W ignore experiments/external_audits/hexi/04_summarise.py
python -W ignore experiments/external_audits/hexi/05_figure.py
