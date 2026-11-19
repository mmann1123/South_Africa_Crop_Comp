"""Re-score the ensemble-size and loss/sampler ablations on the training-median-filled holdout.

Re-inference only, from the saved checkpoints the two published ablations used
(ensemble_size_sweep.py, loss_sampler_ablation.py). The raw-sequence field models read the
holdout through missing.fill_missing, so HOLDOUT_FILL=train_median (set here before any import)
switches their fill; the xr_fresh TabNet (field) reads the field feature file, which is swapped
for the one rebuilt from the training-median-filled series
(holdout_fill_check/data/combined_test_features_trainmed.parquet). Predictions go to
results/predictions_filled/, tables to results/ensemble_size_sweep_filled.csv and
results/loss_sampler_ablation_filled.csv, each with the published (zero-fill) value alongside.
Run in the deep_field env:
  python experiments/ablations/rescore_filled.py
"""
import os
import sys

os.environ["HOLDOUT_FILL"] = "train_median"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import ensemble_size_sweep as ES  # noqa: E402
import loss_sampler_ablation as LS  # noqa: E402
import predict_oos as P  # noqa: E402
from score_oos import score  # noqa: E402

P.COMBINED_TEST_FEATURES_PATH = os.path.join(REPO, "experiments", "holdout_fill_check", "data",
                                             "combined_test_features_trainmed.parquet")
assert os.path.exists(P.COMBINED_TEST_FEATURES_PATH), P.COMBINED_TEST_FEATURES_PATH
PRED = os.path.join(HERE, "results", "predictions_filled")
os.makedirs(PRED, exist_ok=True)


def run(fn, model_dir, tag, seeds=None):
    csv = os.path.join(PRED, f"{tag}.csv")
    fn(model_dir, csv, seeds=seeds) if seeds is not None else fn(model_dir, csv)
    return score(csv)


def ensemble_size():
    pub = pd.read_csv(os.path.join(HERE, "results", "ensemble_size_sweep.csv"))
    rows = []
    for name, (fn, md) in ES.MODELS.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        m5 = run(fn, md, f"{safe}_5seed", ES.ALL_SEEDS)
        m3 = run(fn, md, f"{safe}_3seed", ES.ALL_SEEDS[:3])
        singles = [run(fn, md, f"{safe}_1seed_{s}", [s]) for s in ES.ALL_SEEDS]
        f1s = np.array([r["f1_macro"] for r in singles])
        ks = np.array([r["kappa"] for r in singles])
        rows.append(dict(model=name, n_seeds=1, seeds="42", f1_macro=singles[0]["f1_macro"], kappa=singles[0]["kappa"],
                         f1_macro_mean=f1s.mean(), f1_macro_std=f1s.std(), kappa_mean=ks.mean(), kappa_std=ks.std()))
        rows.append(dict(model=name, n_seeds=3, seeds="42,101,202", f1_macro=m3["f1_macro"], kappa=m3["kappa"]))
        rows.append(dict(model=name, n_seeds=5, seeds="42,101,202,303,404", f1_macro=m5["f1_macro"], kappa=m5["kappa"]))
    df = pd.DataFrame(rows).merge(pub[["model", "n_seeds", "f1_macro", "kappa", "f1_macro_mean", "f1_macro_std"]]
                                  .rename(columns=lambda c: c if c in ("model", "n_seeds") else f"{c}_published"),
                                  on=["model", "n_seeds"], how="left")
    df.to_csv(os.path.join(HERE, "results", "ensemble_size_sweep_filled.csv"), index=False)
    print("\n=== ensemble size, filled holdout ===")
    print(df.round(4).to_string(index=False))


def loss_sampler():
    pub = pd.read_csv(os.path.join(HERE, "results", "loss_sampler_ablation.csv"))
    rows = []
    for name, (subdir, _, predict_fn) in LS.MODELS.items():
        for loss, samp in LS.VARIANTS:
            vdir = os.path.join(LS.MODELS_OUT, subdir, f"{loss}_{samp}")
            m = run(predict_fn, vdir, f"{subdir}_{loss}_{samp}")
            rows.append(dict(model=name, loss=loss, sampler=samp, f1_macro=m["f1_macro"], kappa=m["kappa"], xent=m["xent"]))
    df = pd.DataFrame(rows).merge(pub.rename(columns={"f1_macro": "f1_macro_published", "kappa": "kappa_published", "xent": "xent_published"}),
                                  on=["model", "loss", "sampler"], how="left")
    df.to_csv(os.path.join(HERE, "results", "loss_sampler_ablation_filled.csv"), index=False)
    print("\n=== loss / sampler, filled holdout ===")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    ensemble_size()
    loss_sampler()
