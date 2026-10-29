#!/usr/bin/env bash
# Run the external missingness audits end to end, then delete downloaded
# data (manifests kept). Pass --keep-data to skip the clean-up step.
# Requires: conda env `ee` (see env/), Earth Engine credentials
# (`earthengine authenticate --auth_mode=notebook`), ~20 GB free under /mnt/bigdrive/external_audits.
set -euo pipefail
cd "$(dirname "$0")/../.."
KEEP=0
for a in "$@"; do [[ "$a" == "--keep-data" ]] && KEEP=1; done
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ee
python -m pytest experiments/external_audits/tests -q
for paper in timematch denethor daccn kerner sen4agrinet stoian; do
  bash "experiments/external_audits/$paper/run.sh"
done
if [[ $KEEP -eq 0 ]]; then
  for paper in timematch denethor daccn kerner sen4agrinet stoian; do
    python "experiments/external_audits/$paper/99_cleanup.py"
  done
fi
echo
echo "== Findings =="
for paper in timematch denethor daccn kerner sen4agrinet stoian; do
  echo "--- $paper ---"
  awk '/^## Finding/{f=1;next} /^## /{f=0} f' "experiments/external_audits/$paper/README.md"
done
