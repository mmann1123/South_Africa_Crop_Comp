"""Month-by-month missing-month injection: is the September result specific to September?

For each of the ten months in turn, all six bands of that month are blanked for every pixel, on
the holdout and on the in-region test fields (the 20% FID split every raw-sequence model shares,
train_test_split(fids, 0.2, random_state=42)). The injected month then receives one treatment:
  zero         raw 0 before standardization (the published rule)
  train_median the training-tile median of each column
  deletion     masked out of attention (step_deletion.py; attention models only)
Missing values that occur naturally (September in 41.5% of holdout pixels, scattered elsewhere)
are filled with the training median in every condition, so each row isolates the injected month.
A "none" row (no injection) is the reference.

Models (5-seed ensembles, AMP, field majority vote): the TempCNN and CNN-BiLSTM 2 x 2 arms
(dense / gated x month dropout, sparse_gate_backbones/models), and L-TAE, L-TAE-S, the
Transformer and the month-dropout L-TAE / L-TAE-S (loaders from step_deletion.py).

To keep this inference-only and fast, at most 50 pixels per field are used (fixed draw, seed 42),
identical for every model and condition; absolute scores differ slightly from all-pixel scores,
comparisons across conditions do not. Writes results/month_injection.csv and
figures/month_injection.{pdf,png}. Run in the deep_field env:
  python experiments/missingness/month_injection.py
"""
import os
import sys
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import step_deletion as sd  # noqa: E402  (also puts deep_learn/src, out_of_sample, field_reduction on the path)
from attribution import load_model, SG, tr  # noqa: E402
from config import MERGED_DL_PATH, MERGED_DL_TEST_PATH  # noqa: E402
from compare_predictions import load_ground_truth  # noqa: E402
from missing import train_stats  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

MONTHS = tr.MONTHS
CAP, SEED, BS = 50, 42, 8192
ARMS = {"none": "dense", "none_md0.2": "dense + dropout", "sparsemax": "gated", "sparsemax_md0.2": "gated + dropout"}
ATTN = {"ltae": "L-TAE", "ltae_s": "L-TAE-S", "transformer": "Transformer",
        "ltae_md0.2": "L-TAE + dropout", "ltae_s_md0.2": "L-TAE-S + dropout"}


def cap_rows(fids, rng):
    s = pd.Series(np.arange(len(fids)), index=fids)
    keep = [g.values if len(g) <= CAP else rng.choice(g.values, CAP, replace=False) for _, g in s.groupby(level=0)]
    return np.sort(np.concatenate(keep))


def datasets():
    rng = np.random.default_rng(SEED)
    tr_fid = pq.read_table(MERGED_DL_PATH, columns=["fid"]).to_pandas()["fid"]
    _, test_fids = train_test_split(tr_fid.unique(), test_size=0.2, random_state=42)
    idx = np.where(tr_fid.isin(test_fids).values)[0]
    train = pd.read_parquet(MERGED_DL_PATH).iloc[idx].reset_index(drop=True)
    train = train.iloc[cap_rows(train["fid"].values, rng)].reset_index(drop=True)
    lab_in = train.groupby("fid")["crop_name"].agg(lambda x: x.mode()[0])
    hold = pd.read_parquet(MERGED_DL_TEST_PATH)
    hold = hold.iloc[cap_rows(hold["fid"].values, rng)].reset_index(drop=True)
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    return {"in-region": (train, lab_in), "holdout": (hold, gt)}


class Spec:
    """Everything needed to turn a raw frame into model inputs and run the 5-seed ensemble."""

    def __init__(self, cols, scaler, le, to_seq, models, fwd, fwd_masked=None):
        self.cols, self.scaler, self.le, self.to_seq = cols, scaler, le, to_seq
        self.models, self.fwd, self.fwd_masked = models, fwd, fwd_masked
        names = to_seq(np.array(cols, dtype=object)[None])[0]  # (T, C) column names
        self.slot = {m: next(t for t in range(names.shape[0]) if names[t, 0].endswith("_" + m)) for m in MONTHS}
        self.median = train_stats("median", cols).values


def harness_spec(bb, arm):
    r = tr.RECIPE[bb]
    art = joblib.load(f"{SG}/models/{bb}_{arm}/artifacts.joblib")
    models = [load_model(bb, arm, len(art["le"].classes_), s) for s in r["seeds"]]
    return Spec(art["cols"], art["scaler"], art["le"], lambda X: tr.to_seq(X, r["layout"]), models, lambda m, x: m(x))


def attn_spec(name):
    loader, masked, logits_only, _ = sd.MODELS[name]
    (cols, scaler, le), (make, path) = loader()
    models = []
    for s in sd.SEEDS:
        m = make(s, len(le.classes_)).to(sd.device)
        m.load_state_dict(torch.load(path.format(s), map_location=sd.device, weights_only=True))
        models.append(m.eval())
    return Spec(cols, scaler, le, lambda X: X.reshape(-1, sd.T, sd.C), models,
                lambda m, x: sd.plain(m, x, logits_only), masked)


def predict(spec, df, month, treatment):
    raw = df[spec.cols].to_numpy(dtype=np.float64, copy=True)
    seq_nan = spec.to_seq(np.isnan(raw))  # natural gaps, (N, T, C)
    raw = np.where(np.isnan(raw), spec.median[None], raw)  # natural gaps: training median
    obs = np.ones(seq_nan.shape[:2], bool)
    if month is not None:
        t = spec.slot[month]
        cols_m = [i for i, c in enumerate(spec.cols) if c.endswith("_" + month)]
        if treatment == "zero":
            raw[:, cols_m] = 0.0
        else:  # train_median, and the placeholder value for deletion
            raw[:, cols_m] = spec.median[cols_m]
        if treatment == "deletion":
            obs[:, t] = False
    X = raw if spec.scaler is None else spec.scaler.transform(raw)
    X = torch.from_numpy(spec.to_seq(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)))
    ob = torch.from_numpy(obs)
    logits = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for m in spec.models:
            out = []
            for i in range(0, len(X), BS):
                xb, obb = X[i:i + BS].to(sd.device), ob[i:i + BS].to(sd.device)
                out.append((spec.fwd_masked(m, xb, obb) if treatment == "deletion" else spec.fwd(m, xb)).float().cpu())
            logits = logits + torch.cat(out)
    p = (logits / len(spec.models)).argmax(1).numpy()
    v = pd.DataFrame({"fid": df["fid"].values, "p": p}).groupby("fid")["p"].agg(lambda x: Counter(x).most_common(1)[0][0])
    return pd.Series(spec.le.inverse_transform(v.values), index=v.index)


def f1(pred, labels):
    cl = sorted(labels.unique()); ci = {c: i for i, c in enumerate(cl)}
    idx = labels.index.intersection(pred.index)
    return metrics(confusion(labels.reindex(idx).map(ci).values, pred.reindex(idx).map(ci).values, len(cl)))[0]


def main():
    data = datasets()
    specs = [(f"{'TempCNN' if bb == 'tempcnn' else 'CNN-BiLSTM'}, {lab}", lambda bb=bb, a=a: harness_spec(bb, a))
             for bb in ("tempcnn", "cnn_bilstm") for a, lab in ARMS.items()]
    specs += [(lab, lambda n=n: attn_spec(n)) for n, lab in ATTN.items()]
    out = os.path.join(HERE, "results", "month_injection.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["model"] for r in rows}
    for name, make in specs:
        if name in done:
            print(f"{name}: cached", flush=True)
            continue
        spec = make()
        treatments = ["zero", "train_median"] + (["deletion"] if spec.fwd_masked else [])
        new = []
        for dname, (df, labels) in data.items():
            base = predict(spec, df, None, None)
            new.append({"model": name, "data": dname, "month": "none", "treatment": "none",
                        "f1m": f1(base, labels), "fields_changed": 0})
            for m in MONTHS:
                for t in treatments:
                    p = predict(spec, df, m, t)
                    new.append({"model": name, "data": dname, "month": m, "treatment": t, "f1m": f1(p, labels),
                                "fields_changed": int((p.reindex(base.index) != base).sum())})
        rows += new
        pd.DataFrame(rows).round(4).to_csv(out, index=False)
        z = pd.DataFrame(new)
        print(name, z[z.treatment != "none"].groupby(["data", "treatment"]).f1m.mean().round(3).to_dict(),
              "base", z[z.treatment == "none"].set_index("data").f1m.round(3).to_dict(), flush=True)
        del spec; torch.cuda.empty_cache()
    figure(pd.DataFrame(rows))


def figure(d):
    sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
    from figstyle import apply_style
    apply_style()
    base = d[d.treatment == "none"].set_index(["model", "data"]).f1m
    d = d[d.treatment != "none"].copy()
    d["drop"] = d.apply(lambda r: r.f1m - base[(r.model, r.data)], axis=1)
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey="row")
    col = {"zero": "#d55e00", "train_median": "#0072b2", "deletion": "#009e73"}
    x = np.arange(len(MONTHS))
    groups = [("dense, no dropout", [m for m in models if m.endswith("dense") or m in ("L-TAE", "Transformer")]),
              ("gated, no dropout", [m for m in models if m.endswith(", gated") or m == "L-TAE-S"]),
              ("with month dropout", [m for m in models if "dropout" in m])]
    for i, dname in enumerate(["holdout", "in-region"]):
        for j, (title, ms) in enumerate(groups):
            ax = axes[i, j]
            for t, c in col.items():
                g = d[(d.data == dname) & (d.treatment == t) & d.model.isin(ms)]
                if g.empty:
                    continue
                piv = g.pivot_table(index="month", columns="model", values="drop").reindex(MONTHS)
                ax.plot(x, piv.mean(1), color=c, marker="o", ms=3.5, lw=1.5, label=t.replace("_", " "))
                ax.fill_between(x, piv.min(1), piv.max(1), color=c, alpha=0.12, lw=0)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(x, [m[:3] for m in MONTHS], fontsize=7, rotation=90)
            ax.set_title(f"({chr(97 + 3 * i + j)}) {dname}: {title} (n={len(ms)})", fontsize=9.5)
        axes[i, 0].set_ylabel("Change in macro-F1 vs no injection", fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"month_injection.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
