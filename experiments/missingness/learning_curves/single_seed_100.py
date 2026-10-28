"""100% points of the learning curves as single models, matching the sweep protocol (one training
seed per point), instead of one sweep run. Each of the 5 seeds behind a full-data model is scored
on its own on the training-median-filled holdout.

  TempCNN, CNN-BiLSTM (dense and gated): the harness checkpoints in
      experiments/sparse_gate_backbones/models/{bb}_{none,sparsemax} (the sweep trains the same
      harness), predicted here
  L-TAE-S: field_reduction/models/ltae_sparse_pixel/frac_1.00, predicted here
  L-TAE, pixel TabNet: per-seed files already in holdout_fill_check/predictions/
  XGBoost (field), logistic regression: deterministic single models; the sweep value stands

Check: averaging the seeds' logits and voting per field must reproduce the existing 5-seed
ensemble predictions (at most 2 fields may differ, from AMP).
Writes learning_curves/predictions_100/ and results/learning_curve_100pct.csv.
Run in the deep_field env: python experiments/missingness/learning_curves/single_seed_100.py
"""
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SG = os.path.join(REPO, "experiments", "sparse_gate_backbones")
FR = os.path.join(REPO, "experiments", "field_reduction")
FILL = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
for p in (SG, FR, os.path.join(REPO, "deep_learn", "src"), os.path.join(REPO, "out_of_sample"),
          os.path.join(HERE, "..")):
    sys.path.insert(0, p)
import train as tr  # noqa: E402
import models_arch as ma  # noqa: E402
import predict_oos as po  # noqa: E402
from attribution import load_model  # noqa: E402
from compare_predictions import load_ground_truth  # noqa: E402
from missing import fill_missing  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

OUT = os.path.join(HERE, "predictions_100")
TRAINMED = tr.HOLDOUT["trainmed"]


def vote(fids, pred, le):
    s = pd.DataFrame({"fid": fids, "p": pred}).groupby("fid")["p"].agg(lambda x: Counter(x).most_common(1)[0][0])
    return pd.Series(le.inverse_transform(s.values), index=s.index, name="crop_name")


def harness(bb, arm):
    r = tr.RECIPE[bb]
    art = joblib.load(f"{SG}/models/{bb}_{arm}/artifacts.joblib")
    X, fids = tr.load_holdout(TRAINMED, art, r)
    per, logits = {}, []
    for s in r["seeds"]:
        lg, _ = tr.logits_of(load_model(bb, arm, len(art["le"].classes_), s), X, r["batch"])
        logits.append(lg)
        per[s] = vote(fids, lg.argmax(1).numpy(), art["le"])
        torch.cuda.empty_cache()
    ens = vote(fids, torch.stack(logits).mean(0).argmax(1).numpy(), art["le"])
    return per, ens, f"{SG}/predictions/{bb}_{arm}_trainmed.csv"


def ltae_s():
    d = f"{FR}/models/ltae_sparse_pixel/frac_1.00"
    cols, scaler, le = (joblib.load(f"{d}/{n}.joblib") for n in ("feature_columns", "scaler", "label_encoder"))
    df = pd.read_parquet(TRAINMED)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df[cols] = fill_missing(df, cols, published="zero")
    X = np.nan_to_num(scaler.transform(df[cols].values).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X = torch.tensor(X.reshape(-1, ma.T_SEQ, ma.N_BANDS))
    fids = df["fid"].values
    per, logits = {}, []
    for s in po.SEEDS_ENSEMBLE:
        m = ma.LTAESparse(in_channels=ma.N_BANDS, num_classes=len(le.classes_), **po._ltae_s_arch(d)).to(tr.device)
        m.load_state_dict(torch.load(f"{d}/ltae_sparse_seed_{s}.pt", map_location=tr.device, weights_only=True))
        m.eval()
        out = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in range(0, len(X), 2048):
                out.append(m(X[i:i + 2048].to(tr.device))[0].float().cpu())
        lg = torch.cat(out)
        logits.append(lg)
        per[s] = vote(fids, lg.argmax(1).numpy(), le)
    ens = vote(fids, torch.stack(logits).mean(0).argmax(1).numpy(), le)
    return per, ens, f"{FILL}/ltae_s_trainmed.csv"


def from_files(stem, seeds):
    rd = lambda p: pd.read_csv(p).drop_duplicates("fid").set_index("fid")["crop_name"]  # noqa: E731
    return {s: rd(f"{FILL}/{stem}_trainmed_seed{s}.csv") for s in seeds}, rd(f"{FILL}/{stem}_trainmed.csv"), None


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    cl = sorted(gt.unique()); ci = {c: i for i, c in enumerate(cl)}; K = len(cl)

    def f1(p):
        idx = gt.index.intersection(p.index)
        return metrics(confusion(gt.reindex(idx).map(ci).values, p.reindex(idx).map(ci).values, K))[0]

    runs = {"TempCNN": lambda: harness("tempcnn", "none"), "TempCNN, gated": lambda: harness("tempcnn", "sparsemax"),
            "CNN-BiLSTM": lambda: harness("cnn_bilstm", "none"),
            "CNN-BiLSTM, gated": lambda: harness("cnn_bilstm", "sparsemax"),
            "L-TAE-S": ltae_s, "L-TAE": lambda: from_files("ltae", [42, 101, 202, 303, 404]),
            "TabNet (pixel)": lambda: from_files("tabnet", [42, 101, 202, 303, 404])}
    rows = []
    for name, fn in runs.items():
        per, ens, ref = fn()
        if ref is not None:  # predicted here: the seed average must reproduce the saved ensemble
            saved = pd.read_csv(ref).drop_duplicates("fid").set_index("fid")["crop_name"]
            n = int((ens.reindex(saved.index) != saved).sum())
            assert n <= 2, f"{name}: seed-averaged ensemble differs from {ref} on {n} fields"
        for s, p in per.items():
            p.reset_index().to_csv(f"{OUT}/{name.replace(', ', '_').replace(' ', '_')}_seed{s}.csv", index=False)
            rows.append({"model": name, "seed": s, "f1m_single": f1(p), "f1m_ensemble": f1(ens)})
        print(name, [round(r["f1m_single"], 3) for r in rows if r["model"] == name], "ensemble", round(f1(ens), 3), flush=True)
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(HERE, "..", "results", "learning_curve_100pct.csv"), index=False)


if __name__ == "__main__":
    main()
