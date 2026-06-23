"""Ensemble-size ablation (reviewer comment 3.4).

Re-inference only: reuses the saved per-seed checkpoints and the existing OOS
prediction + scoring pipeline (experiments/field_reduction/{predict_oos,score_oos}.py)
to measure spatial-transfer holdout F1-macro / Cohen kappa as a function of the
number of seeds in the ensemble (1 / 3 / 5), for three field-level models:

  - TabNet (field)       -- baseline retrain from loss_sampler_ablation (focal/off)
  - L-TAE-S (field)      -- sparse-attention model (field_reduction/.../frac_1.00/)
  - CNN-BiLSTM (field)   -- fragile-cluster contrast (deep_learn/src/models/)

The 5-seed rows reproduce the published configuration; for CNN-BiLSTM (field) the
5-seed result is validated against out_of_sample/scoring_results/model_comparison.csv
before the 1/3-seed rows are trusted (see VERIFY below). (The flagship pixel-level
TabNet ensemble's 5-seed transfer score, 0.60, is reported in the main results
table; its old checkpoints predate the current pytorch_tabnet and cannot be
re-loaded for re-inference, so the field-level TabNet retrain stands in here.)

For the 1-seed setting we report seed 42 as the point estimate plus mean+/-std
across all five single-seed members. 3-seed uses the prefix [42,101,202].

Run with the deep_field python (torch + pytorch_tabnet):
    python experiments/ablations/ensemble_size_sweep.py
"""

import os
import sys

import numpy as np
import pandas as pd

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIELD_RED_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", "field_reduction"))
sys.path.insert(0, FIELD_RED_DIR)

sys.stdout.reconfigure(line_buffering=True)

import predict_oos as P            # noqa: E402
from score_oos import score        # noqa: E402
from config import MODEL_DIR        # noqa: E402

OUT_DIR = os.path.join(EXPERIMENT_DIR, "results")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

ALL_SEEDS = [42, 101, 202, 303, 404]

LTAE_S_FIELD_DIR = os.path.join(FIELD_RED_DIR, "models", "ltae_sparse_field", "frac_1.00")
# TabNet field baseline (focal/off) produced by loss_sampler_ablation.py
TABNET_FIELD_DIR = os.path.join(EXPERIMENT_DIR, "models", "tabnet_field", "focal_off")

# Published OOS anchors (out_of_sample/scoring_results/model_comparison.csv).
# None where no published OOS number exists (these are fresh/new references).
ANCHORS = {
    "TabNet (field)": None,
    "CNN-BiLSTM (field)": (0.5060, 0.3256),
    "L-TAE-S (field)": None,
}

MODELS = {
    "TabNet (field)": (P.predict_tabnet_field, TABNET_FIELD_DIR),
    "L-TAE-S (field)": (P.predict_ltae_sparse_field, LTAE_S_FIELD_DIR),
    "CNN-BiLSTM (field)": (P.predict_cnn_bilstm_field, MODEL_DIR),
}


def run(fn, model_dir, seeds, tag):
    csv = os.path.join(PRED_DIR, f"{tag}.csv")
    fn(model_dir, csv, seeds=seeds)
    return score(csv)


def main():
    rows = []
    for name, (fn, md) in MODELS.items():
        print(f"\n{'=' * 60}\n{name}  (model_dir={md})\n{'=' * 60}")
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")

        # --- 5-seed (published config) ---
        m5 = run(fn, md, ALL_SEEDS, f"{safe}_5seed")

        # VERIFY: 5-seed must reproduce the published OOS anchor (if any)
        anchor = ANCHORS.get(name)
        if anchor is not None:
            f1_a, k_a = anchor
            df1 = abs(m5["f1_macro"] - f1_a)
            dk = abs(m5["kappa"] - k_a)
            flag = "OK" if (df1 <= 0.005 and dk <= 0.005) else "*** MISMATCH ***"
            print(f"[VERIFY] {name} 5-seed vs anchor: "
                  f"F1 {m5['f1_macro']:.4f} vs {f1_a:.4f} (d={df1:.4f}), "
                  f"kappa {m5['kappa']:.4f} vs {k_a:.4f} (d={dk:.4f})  {flag}")

        # --- 3-seed (prefix) ---
        m3 = run(fn, md, ALL_SEEDS[:3], f"{safe}_3seed")

        # --- 1-seed: all five singletons ---
        singles = [run(fn, md, [s], f"{safe}_1seed_{s}") for s in ALL_SEEDS]
        f1s = np.array([r["f1_macro"] for r in singles])
        ks = np.array([r["kappa"] for r in singles])
        seed42 = singles[0]  # ALL_SEEDS[0] == 42

        rows.append(dict(model=name, n_seeds=1, seeds="42",
                         f1_macro=seed42["f1_macro"], kappa=seed42["kappa"],
                         f1_macro_mean=f1s.mean(), f1_macro_std=f1s.std(),
                         kappa_mean=ks.mean(), kappa_std=ks.std()))
        rows.append(dict(model=name, n_seeds=3, seeds="42,101,202",
                         f1_macro=m3["f1_macro"], kappa=m3["kappa"]))
        rows.append(dict(model=name, n_seeds=5, seeds="42,101,202,303,404",
                         f1_macro=m5["f1_macro"], kappa=m5["kappa"]))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "ensemble_size_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n{'=' * 60}\nResults -> {out_csv}\n{'=' * 60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
