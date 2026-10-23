"""Tree + deep ensembles on the holdout (Shwartz-Ziv & Armon 2022: XGBoost + deep ensembles beat
either alone). Hard majority vote over saved holdout predictions, under the training-median and
zero September fills. No holdout information is used to pick members or break ties: members are
whole model families or all combinations, and vote ties go to the member with the higher published
in-region macro-F1 (Table 1), then alphabetical.

  deep: raw-sequence pixel models (CNN-BiLSTM, TempCNN, L-TAE, Transformer, TimesNet)
  tree: XGBoost/LightGBM field, Random Forest/LightGBM/XGBoost pixel (xr_fresh features)

1. Family ensembles: deep-5, tree-5, mixed-10, with paired field bootstraps (B = 10,000, seed 42)
   of mixed vs deep and mixed vs tree.
2. Every 3-member ensemble: deep-only (10), tree-only (10), mixed 2+1 and 1+2 (100); gain over the
   mean and over the best of its own members.
3. Error diversity: mean pairwise disagreement and double-fault rate (both wrong) within each
   family and across families.
Writes results/ensemble_family.csv, ensemble_triples.csv, ensemble_diversity.csv.
Run in the deep_field env: python experiments/missingness/ensemble.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

PRED = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
B, SEED = 10_000, 42
FILLS = {"train_median": "trainmed", "zero": "orig"}
# name -> (file stem, family, published in-region macro-F1)
MEMBERS = {
    "CNN-BiLSTM, pixel": ("cnn_bilstm", "deep", 0.72),
    "TempCNN, pixel": ("tempcnn", "deep", 0.77),
    "L-TAE, pixel": ("ltae", "deep", 0.78),
    "Transformer, pixel": ("transformer", "deep", 0.78),
    "TimesNet, pixel": ("timesnet", "deep", 0.76),
    "XGBoost, field": ("xgboost", "tree", 0.70),
    "LightGBM, field": ("lgbm", "tree", 0.68),
    "Random Forest, pixel": ("base_rf", "tree", 0.65),
    "LightGBM, pixel": ("base_lgbm", "tree", 0.62),
    "XGBoost, pixel": ("base_xgb", "tree", 0.64),
}
PRIORITY = sorted(MEMBERS, key=lambda m: (-MEMBERS[m][2], m))  # tie-break order


def vote(Y, names, K):
    """Majority vote over the columns of Y (fields x members); ties to the earliest in PRIORITY."""
    order = sorted(range(len(names)), key=lambda i: PRIORITY.index(names[i]))
    Y = Y[:, order]
    counts = np.zeros((len(Y), K), int)
    for j in range(Y.shape[1]):
        counts[np.arange(len(Y)), Y[:, j]] += 1
    top = counts == counts.max(1, keepdims=True)
    out = Y[:, 0].copy()
    for j in range(Y.shape[1] - 1, -1, -1):  # the highest-priority tied member wins
        hit = top[np.arange(len(Y)), Y[:, j]]
        out[hit] = Y[hit, j]
    return out


if __name__ == "__main__":
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    cl = sorted(gt.unique()); ci = {c: i for i, c in enumerate(cl)}; K = len(cl)
    f1 = lambda yt, yp, w=None: metrics(confusion(yt, yp, K, w))[0]  # noqa: E731
    rng = np.random.default_rng(SEED)
    fam_rows, tri_rows, div_rows = [], [], []
    for fill, tag in FILLS.items():
        preds = {m: pd.read_csv(f"{PRED}/{s}_{tag}.csv").drop_duplicates("fid").set_index("fid")["crop_name"]
                 for m, (s, _, _) in MEMBERS.items()}
        idx = gt.index
        for p in preds.values():
            idx = idx.intersection(p.index)
        yt = gt.reindex(idx).map(ci).values
        Y = {m: p.reindex(idx).map(ci).values for m, p in preds.items()}
        assert not any(np.isnan(v.astype(float)).any() for v in Y.values()), "unknown class label"
        single = {m: f1(yt, y) for m, y in Y.items()}
        fam = {f: [m for m in MEMBERS if MEMBERS[m][1] == f] for f in ("deep", "tree")}

        # 1. family ensembles
        ens = {"deep-5": fam["deep"], "tree-5": fam["tree"], "mixed-10": list(MEMBERS)}
        ey = {e: vote(np.column_stack([Y[m] for m in ms]), ms, K) for e, ms in ens.items()}
        N = len(idx)
        W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(float) for _ in range(B)]
        bs = {e: np.array([f1(yt, y, w) for w in W]) for e, y in ey.items()}
        for e, y in ey.items():
            lo, hi = np.percentile(bs[e], [2.5, 97.5])
            fam_rows.append({"fill": fill, "ensemble": e, "f1m": f1(yt, y), "ci_lo": lo, "ci_hi": hi,
                             "best_member": max(single[m] for m in ens[e]),
                             "mean_member": np.mean([single[m] for m in ens[e]])})
        for a, b in [("mixed-10", "deep-5"), ("mixed-10", "tree-5"), ("deep-5", "tree-5")]:
            d = bs[a] - bs[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            fam_rows.append({"fill": fill, "ensemble": f"{a} minus {b}", "f1m": f1(yt, ey[a]) - f1(yt, ey[b]),
                             "ci_lo": lo, "ci_hi": hi})

        # 2. every 3-member ensemble
        for trio in itertools.combinations(MEMBERS, 3):
            nd = sum(MEMBERS[m][1] == "deep" for m in trio)
            kind = {3: "deep-only", 0: "tree-only", 2: "mixed 2 deep + 1 tree", 1: "mixed 1 deep + 2 tree"}[nd]
            s = f1(yt, vote(np.column_stack([Y[m] for m in trio]), list(trio), K))
            tri_rows.append({"fill": fill, "kind": kind, "members": " | ".join(trio), "f1m": s,
                             "gain_vs_mean": s - np.mean([single[m] for m in trio]),
                             "gain_vs_best": s - max(single[m] for m in trio)})

        # 3. diversity
        for a, b in itertools.combinations(MEMBERS, 2):
            fa, fb = MEMBERS[a][1], MEMBERS[b][1]
            div_rows.append({"fill": fill, "pair": "within deep" if fa == fb == "deep" else
                             "within tree" if fa == fb == "tree" else "deep x tree",
                             "a": a, "b": b, "disagreement": np.mean(Y[a] != Y[b]),
                             "double_fault": np.mean((Y[a] != yt) & (Y[b] != yt))})
        for m in MEMBERS:
            fam_rows.append({"fill": fill, "ensemble": f"single: {m}", "f1m": single[m]})

    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    fr, tr, dv = (pd.DataFrame(r).round(4) for r in (fam_rows, tri_rows, div_rows))
    fr.to_csv(f"{res}/ensemble_family.csv", index=False)
    tr.to_csv(f"{res}/ensemble_triples.csv", index=False)
    dv.to_csv(f"{res}/ensemble_diversity.csv", index=False)
    pd.set_option("display.width", 200)
    print(fr.to_string(index=False))
    print(tr.groupby(["fill", "kind"])[["f1m", "gain_vs_mean", "gain_vs_best"]]
          .agg(["mean", "min", "max"]).round(3).to_string())
    print((tr.assign(pos=tr.gain_vs_best > 0).groupby(["fill", "kind"]).pos.mean()).round(2).to_string())
    print(dv.groupby(["fill", "pair"])[["disagreement", "double_fault"]].mean().round(3).to_string())
