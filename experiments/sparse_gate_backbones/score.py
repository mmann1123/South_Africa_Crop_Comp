"""Score the gated backbones on the holdout: each variant's macro-F1 under both September fills,
and paired field bootstraps (B=10,000, seed 42) of sparsemax vs softmax control and of each
gated variant vs the published dense model. Run in the deep_field env."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

FILL = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
cl = sorted(gt.unique()); ci = {c: i for i, c in enumerate(cl)}; K = len(cl)


def rd(p):
    return pd.read_csv(p).drop_duplicates("fid").set_index("fid")["crop_name"]


def pair(a, b, rng):
    j = pd.concat([gt.rename("t"), a.rename("a"), b.rename("b")], axis=1, join="inner")
    yt, ya, yb = (j[c].map(ci).values for c in "tab")
    N = len(j)
    pa, pb = metrics(confusion(yt, ya, K)), metrics(confusion(yt, yb, K))
    d = np.empty((10_000, 3))
    for i in range(len(d)):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        d[i] = metrics(confusion(yt, ya, K, w)) - metrics(confusion(yt, yb, K, w))
    lo, hi = np.percentile(d[:, 0], [2.5, 97.5])
    return pa[0], pb[0], pa[0] - pb[0], lo, hi


rows = []
rng = np.random.default_rng(42)
for bb in ["tempcnn", "cnn_bilstm"]:
    for fill in ["trainmed", "orig"]:
        v = {g: rd(os.path.join(HERE, "predictions", f"{bb}_{g}_{fill}.csv")) for g in ["sparsemax", "softmax"]}
        v["dense (published)"] = rd(os.path.join(FILL, f"{bb}_{fill}.csv"))
        for a, b in [("sparsemax", "softmax"), ("sparsemax", "dense (published)"), ("softmax", "dense (published)")]:
            fa, fb, dlt, lo, hi = pair(v[a], v[b], rng)
            rows.append({"backbone": bb, "fill": fill, "A": a, "B": b, "F1m_A": fa, "F1m_B": fb,
                         "delta": dlt, "ci_lo": lo, "ci_hi": hi, "significant": bool(lo > 0 or hi < 0)})
df = pd.DataFrame(rows).round(3)
df.to_csv(os.path.join(HERE, "results.csv"), index=False)
print(df.to_string(index=False))
