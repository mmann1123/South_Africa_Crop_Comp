#!/usr/bin/env bash
# TimeSen2Crop re-run, in order. Run from anywhere; training resumes per run.
# Env `deep_field` (PyTorch, GPU). Data root: /mnt/bigdrive/external_audits/timesen2crop_rerun
set -euo pipefail
cd "$(dirname "$0")/../../.."
DF=${DF_PY:-/home/mmann1123/miniconda3/envs/deep_field/bin/python}
D=experiments/external_audits/timesen2crop_rerun
ROOT=/mnt/bigdrive/external_audits/timesen2crop_rerun
mkdir -p $ROOT
[ -f $ROOT/TimeSen2Crop.zip ] || curl -sL -o $ROOT/TimeSen2Crop.zip https://zenodo.org/api/records/4715631/files/TimeSen2Crop.zip/content
echo "b5b7aad3fef192e78252e11c9a0e5cb8  $ROOT/TimeSen2Crop.zip" | md5sum -c -

$DF $D/01_build_arrays.py                     # one array file per tile, read directly from the zip
$DF $D/02_composites.py                       # paper's monthly rule; empty-month table
$DF $D/03_tune.py                             # random search on the paper's ranges, chosen on 33TWN
$DF $D/04_train_eval.py                       # 40 runs, about 1 min each on an RTX 3080 Ti
$DF $D/05_testtime.py                         # month deletion, pattern imposition
$DF -W ignore $D/06_summarise.py              # applies the pass marks fixed in the design
$DF $D/07_exploratory_informative_month.py    # exploratory, outside the pass marks
$DF $D/08_figure.py
