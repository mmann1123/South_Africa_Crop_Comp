"""Loss / weighted-sampling ablation (reviewer comment 3.4), field-level retrains.

For each of three field-level models -- TabNet (field), L-TAE-S (field), and
CNN-BiLSTM (field, fragile-cluster contrast) -- retrain a 5-seed ensemble under
five imbalance-handling settings and score each on the spatial-transfer holdout
with the shared scorer (score_oos.score), so numbers are comparable to the paper.

Settings (one factor at a time around the published baseline):
    focal       + off   baseline (weighted focal gamma=2, no resampling)
    weighted_ce + off   class-weighted CE (removes focal focusing term)
    plain_ce    + off   unweighted CE (no imbalance handling)
    focal       + on    baseline loss + weighted sampling / oversampling
    plain_ce    + on    resampling as the sole imbalance handler

Trains via subprocess (each model's own training script, with toggles), then
predicts + scores in-process. Writes results/loss_sampler_ablation.csv.

Run with deep_field python (slow: 15 retrains x 5 seeds). Use --models to subset.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
FIELD_RED_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", "field_reduction"))
sys.path.insert(0, FIELD_RED_DIR)

sys.stdout.reconfigure(line_buffering=True)

import predict_oos as P            # noqa: E402
from score_oos import score        # noqa: E402
from config import DEEP_FIELD_PYTHON  # noqa: E402

PY = DEEP_FIELD_PYTHON
CNN_SCRIPT = os.path.join(REPO_ROOT, "deep_learn", "src", "Deep Learning",
                          "Pixel_Field_Level", "cnn_bilstm_field.py")
LTAES_SCRIPT = os.path.join(FIELD_RED_DIR, "train_ltae_sparse_field.py")
TABNET_SCRIPT = os.path.join(EXPERIMENT_DIR, "train_tabnet_field_variant.py")

MODELS_OUT = os.path.join(EXPERIMENT_DIR, "models")
PRED_DIR = os.path.join(EXPERIMENT_DIR, "results", "predictions")
RESULTS_CSV = os.path.join(EXPERIMENT_DIR, "results", "loss_sampler_ablation.csv")
os.makedirs(PRED_DIR, exist_ok=True)

VARIANTS = [
    ("focal", "off"),
    ("weighted_ce", "off"),
    ("plain_ce", "off"),
    ("focal", "on"),
    ("plain_ce", "on"),
]


def cmd_ltaes(vdir, loss, samp):
    return ([PY, LTAES_SCRIPT, "--fraction", "1.0", "--output-dir", vdir,
             "--loss", loss, "--sampler", samp], FIELD_RED_DIR)


def cmd_cnn(vdir, loss, samp):
    throwaway = os.path.join(vdir, "oos_pred.csv")
    return ([PY, CNN_SCRIPT, "--loss", loss, "--sampler", samp,
             "--output-dir", vdir, "--pred-csv", throwaway, "--no-report"], REPO_ROOT)


def cmd_tabnet(vdir, loss, samp):
    return ([PY, TABNET_SCRIPT, "--loss", loss, "--sampler", samp,
             "--output-dir", vdir], EXPERIMENT_DIR)


MODELS = {
    "TabNet (field)": ("tabnet_field", cmd_tabnet, P.predict_tabnet_field),
    "L-TAE-S (field)": ("ltaes_field", cmd_ltaes, P.predict_ltae_sparse_field),
    "CNN-BiLSTM (field)": ("cnn_bilstm_field", cmd_cnn, P.predict_cnn_bilstm_field),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=list(MODELS.keys()))
    args = parser.parse_args()

    rows = []
    for name in args.models:
        subdir, cmd_fn, predict_fn = MODELS[name]
        for loss, samp in VARIANTS:
            tag = f"{subdir}_{loss}_{samp}"
            vdir = os.path.join(MODELS_OUT, subdir, f"{loss}_{samp}")
            os.makedirs(vdir, exist_ok=True)
            cmd, cwd = cmd_fn(vdir, loss, samp)
            print(f"\n{'#' * 70}\n# {name}  loss={loss} sampler={samp}\n{'#' * 70}")
            subprocess.run(cmd, cwd=cwd, check=True)

            csv = os.path.join(PRED_DIR, f"{tag}.csv")
            predict_fn(vdir, csv)
            m = score(csv)
            rows.append(dict(model=name, loss=loss, sampler=samp,
                             f1_macro=m["f1_macro"], kappa=m["kappa"], xent=m["xent"]))

            # incremental save so partial progress survives a crash
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}\nResults -> {RESULTS_CSV}\n{'=' * 60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
