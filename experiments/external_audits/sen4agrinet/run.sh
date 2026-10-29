#!/usr/bin/env bash
# Sen4AgriNet audit, in order. Run from the repo root inside the `ee` conda env.
# Step 02 streams the 13.7 GB Dropbox archive once (about 6 minutes) and keeps ~5 GB;
# step 04 reads ~650 NetCDF time axes from Hugging Face by range requests (about 20 minutes).
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -I experiments/external_audits/sen4agrinet/01_fetch_code.py
python -I experiments/external_audits/sen4agrinet/02_fetch_splits.py
python -I experiments/external_audits/sen4agrinet/03_split_composition.py
python -I experiments/external_audits/sen4agrinet/04_sample_time_stacks.py
python -I experiments/external_audits/sen4agrinet/05_fill_rates.py
python -I experiments/external_audits/sen4agrinet/06_oad_csv_crosscheck.py
python -I experiments/external_audits/sen4agrinet/07_stac_cloudcover.py
python -I experiments/external_audits/sen4agrinet/08_summarise.py
