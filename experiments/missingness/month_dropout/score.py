"""Score the month-dropout x sparse-gate 2 x 2 on the holdout, under zero and training-median
September fill. Per cell: macro-F1 with a percentile field-bootstrap 95% CI. Paired contrasts on
the same bootstrap draws (B = 10,000, seed 42): the dropout effect within each gate arm, the gate
effect within each dropout arm, and the interaction (dropout effect with the gate minus without),
each with an exact McNemar test on field correctness (none for the interaction).

Cells:
  TempCNN / CNN-BiLSTM: dense and sparsemax arms retrained in sparse_gate_backbones/train.py;
    dense without dropout is the harness control ({bb}_none), the published dense model is
    scored alongside as a check that the harness reproduces it.
  L-TAE: dense = L-TAE, gated = L-TAE-S; no-dropout arms are the published checkpoints
    re-scored in holdout_fill_check, dropout arms trained by run.sh.
Writes results/cells.csv and results/contrasts.csv. Run in the deep_field env:
  python experiments/missingness/month_dropout/score.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

SG = os.path.join(REPO, "experiments", "sparse_gate_backbones", "predictions")
FILL = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
PRED = os.path.join(HERE, "predictions")
B, SEED = 10_000, 42
FILLS = {"zero": "orig", "train_median": "trainmed"}


def cells(bb, tag):
    """(dropout, gate) -> prediction file for one backbone and fill tag."""
    if bb == "ltae":
        return {(0, "dense"): f"{FILL}/ltae_{tag}.csv", (0, "sparse"): f"{FILL}/ltae_s_{tag}.csv",
                (1, "dense"): f"{PRED}/ltae_pixel_md0.2_{tag}.csv",
                (1, "sparse"): f"{PRED}/ltae_sparse_pixel_md0.2_{tag}.csv",
                "published dense": f"{FILL}/ltae_{tag}.csv"}
    return {(0, "dense"): f"{SG}/{bb}_none_{tag}.csv", (0, "sparse"): f"{SG}/{bb}_sparsemax_{tag}.csv",
            (1, "dense"): f"{SG}/{bb}_none_md0.2_{tag}.csv", (1, "sparse"): f"{SG}/{bb}_sparsemax_md0.2_{tag}.csv",
            "published dense": f"{FILL}/{bb}_{tag}.csv"}


def rd(p):
    return pd.read_csv(p).drop_duplicates("fid").set_index("fid")["crop_name"]


if __name__ == "__main__":
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    cl = sorted(gt.unique()); ci = {c: i for i, c in enumerate(cl)}; K = len(cl)
    rng = np.random.default_rng(SEED)
    crow, prow = [], []
    for bb in ["tempcnn", "cnn_bilstm", "ltae"]:
        for fill, tag in FILLS.items():
            files = cells(bb, tag)
            preds = {k: rd(f) for k, f in files.items()}
            idx = gt.index
            for p in preds.values():
                idx = idx.intersection(p.index)
            assert len(idx) > 0.99 * len(gt), f"{bb} {fill}: {len(idx)} of {len(gt)} fields predicted"
            yt = gt.reindex(idx).map(ci).values
            yp = {k: p.reindex(idx).map(ci).values for k, p in preds.items()}
            N = len(idx)
            W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(float) for _ in range(B)]
            f = {k: metrics(confusion(yt, y, K))[0] for k, y in yp.items()}
            bs = {k: np.array([metrics(confusion(yt, y, K, w))[0] for w in W]) for k, y in yp.items()}
            for k in yp:
                lo, hi = np.percentile(bs[k], [2.5, 97.5])
                crow.append({"backbone": bb, "fill": fill,
                             "dropout": k[0] if isinstance(k, tuple) else 0,
                             "gate": k[1] if isinstance(k, tuple) else "dense (published)",
                             "f1m": f[k], "ci_lo": lo, "ci_hi": hi, "file": os.path.relpath(files[k], REPO)})

            def contrast(name, a, b, sub=None):
                d = bs[a] - bs[b] - (bs[sub[0]] - bs[sub[1]] if sub else 0)
                est = f[a] - f[b] - (f[sub[0]] - f[sub[1]] if sub else 0)
                lo, hi = np.percentile(d, [2.5, 97.5])
                r = {"backbone": bb, "fill": fill, "contrast": name, "delta": est, "ci_lo": lo, "ci_hi": hi,
                     "significant": bool(lo > 0 or hi < 0)}
                if not sub:
                    ca, cb = yp[a] == yt, yp[b] == yt
                    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                    r.update(a_only_right=n01, b_only_right=n10,
                             mcnemar_p=binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)
                prow.append(r)

            contrast("dropout | dense", (1, "dense"), (0, "dense"))
            contrast("dropout | sparse", (1, "sparse"), (0, "sparse"))
            contrast("gate | no dropout", (0, "sparse"), (0, "dense"))
            contrast("gate | dropout", (1, "sparse"), (1, "dense"))
            contrast("interaction (dropout x gate)", (1, "sparse"), (0, "sparse"), sub=((1, "dense"), (0, "dense")))
            contrast("harness dense vs published dense", (0, "dense"), "published dense")
            print(f"{bb:10s} {fill:12s} " + "  ".join(
                f"{'md' if k[0] else '--'}/{k[1]} {f[k]:.3f}" for k in f if isinstance(k, tuple)), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    c, p = pd.DataFrame(crow).round(4), pd.DataFrame(prow).round(4)
    c.to_csv(os.path.join(HERE, "results", "cells.csv"), index=False)
    p.to_csv(os.path.join(HERE, "results", "contrasts.csv"), index=False)
    pd.set_option("display.width", 220)
    print(p.drop(columns=["significant"]).to_string(index=False))
