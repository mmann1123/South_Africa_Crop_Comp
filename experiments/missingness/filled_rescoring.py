"""Re-score the consistent-fill (training-median) holdout predictions saved by
experiments/holdout_fill_check for the four cheap items the paper plan still needed:

  1. paired field bootstrap (B = 10,000, seed 42) and exact McNemar for named model pairs on
     the filled predictions: top dense pixel models vs TabNet, XGBoost (field) and logistic
     regression; the gate pairs; field vs pixel for every architecture run at both levels;
  2. per-class F1 with independent field-bootstrap CIs for all 27 models, and the per-class
     figure for four models (figures/per_class_f1_filled.pdf);
  3. the cereal merge (wheat+barley; all cereals) on the filled predictions;
  4. per-seed macro-F1 and its SD for the models whose five seed members were saved under both
     fills (CNN-BiLSTM, TempCNN, L-TAE pixel and field, TabNet pixel).

Reads experiments/holdout_fill_check/predictions/<key>_trainmed[_seed*].csv. Writes
results/paired_bootstrap_filled.csv, results/per_class_filled.csv, results/cereal_merge_filled.csv,
results/per_seed_sd_filled.csv. Run in the deep_field env:
  python experiments/missingness/filled_rescoring.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "holdout_fill_check"))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
import score as sc  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
B, SEED = 10_000, 42
CLASSES = ["Lucerne/Medics", "Canola", "Barley", "Wheat", "Small grain grazing"]

PAIRS = [
    # top filled dense pixel models vs the three reference models
    ("CNN-BiLSTM, pixel", "TabNet (raw), pixel"), ("TempCNN, pixel", "TabNet (raw), pixel"),
    ("L-TAE, pixel", "TabNet (raw), pixel"), ("L-TAE-S, pixel", "TabNet (raw), pixel"),
    ("CNN-BiLSTM, pixel", "XGBoost, field"), ("TempCNN, pixel", "XGBoost, field"),
    ("L-TAE, pixel", "XGBoost, field"),
    ("CNN-BiLSTM, pixel", "Logistic Regression, pixel"), ("TempCNN, pixel", "Logistic Regression, pixel"),
    ("L-TAE, pixel", "Logistic Regression, pixel"), ("Transformer, pixel", "Logistic Regression, pixel"),
    ("TabNet (raw), pixel", "Logistic Regression, pixel"), ("XGBoost, field", "Logistic Regression, pixel"),
    ("MiniRocket, pixel", "Logistic Regression, pixel"),
    # the published headline claims, re-tested on filled inputs
    ("TabNet (raw), pixel", "CNN-BiLSTM, pixel"), ("L-TAE-S, pixel", "L-TAE, pixel"),
    ("L-TAE-S, field", "L-TAE, field"), ("Transformer, pixel", "L-TAE, pixel"),
    ("L-TAE-S, pixel", "Transformer, pixel"),
    # field vs pixel, every architecture run at both levels
    ("TempCNN, field", "TempCNN, pixel"), ("L-TAE, field", "L-TAE, pixel"),
    ("L-TAE-S, field", "L-TAE-S, pixel"), ("CNN-BiLSTM, field", "CNN-BiLSTM, pixel"),
    ("TimesNet, field", "TimesNet, pixel"), ("MiniRocket, field", "MiniRocket, pixel"),
    ("TabNet (raw, field-avg), field", "TabNet (raw), pixel"), ("XGBoost, field", "XGBoost, pixel"),
    ("LightGBM, field", "LightGBM, pixel"),
]
FIG_MODELS = [("CNN-BiLSTM, pixel", "CNN-BiLSTM"), ("L-TAE, pixel", "L-TAE"),
              ("XGBoost, field", "XGBoost (field)"), ("TabNet (raw), pixel", "TabNet")]
SCHEMES = {"5-class": {}, "wheat+barley": {"Barley": "Wheat/Barley", "Wheat": "Wheat/Barley"},
           "all cereals": {"Barley": "Cereals", "Wheat": "Cereals", "Small grain grazing": "Cereals"}}
SEED_MODELS = {"CNN-BiLSTM, pixel": "cnn_bilstm", "TempCNN, pixel": "tempcnn", "L-TAE, pixel": "ltae",
               "L-TAE, field": "ltae_field", "TabNet (raw), pixel": "tabnet"}


def per_class_f1(cm):
    cm = cm.astype(float)
    tp, denom = np.diag(cm), cm.sum(0) + cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * tp / denom, 0.0)


def paired(gt, preds, cidx, K, rng):
    rows = []
    for a, b in PAIRS:
        m = pd.concat([gt.rename("t"), preds[a].rename("a"), preds[b].rename("b")], axis=1, join="inner")
        yt, ya, yb = (m[c].map(cidx).values for c in ("t", "a", "b"))
        N = len(m)
        pa, pb = metrics(confusion(yt, ya, K)), metrics(confusion(yt, yb, K))
        d = np.empty((B, 3))
        for i in range(B):
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
            d[i] = metrics(confusion(yt, ya, K, w)) - metrics(confusion(yt, yb, K, w))
        ca, cb = ya == yt, yb == yt
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_mcn = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        for j, met in enumerate(["macro_f1", "kappa", "accuracy"]):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            rows.append({"Model A": a, "Model B": b, "metric": met, "A": pa[j], "B": pb[j], "Delta": pa[j] - pb[j],
                         "Delta_lo": lo, "Delta_hi": hi,
                         "p_two_sided": min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean())),
                         "significant_95": bool(lo > 0 or hi < 0),
                         "A_only_correct": n10, "B_only_correct": n01, "mcnemar_p": p_mcn})
    return pd.DataFrame(rows)


def per_class(gt, preds, classes, cidx, K, rng):
    rows = []
    for name, p in preds.items():
        m = pd.concat([gt.rename("t"), p.rename("p")], axis=1, join="inner")
        yt, yp = m["t"].map(cidx).values, m["p"].map(cidx).values
        N = len(m)
        point = per_class_f1(confusion(yt, yp, K))
        boot = np.empty((B, K))
        for i in range(B):
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
            boot[i] = per_class_f1(confusion(yt, yp, K, w))
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for k, c in enumerate(classes):
            rows.append({"model": name, "class": c, "f1": point[k], "f1_lo": lo[k], "f1_hi": hi[k],
                         "support": int((yt == k).sum())})
    return pd.DataFrame(rows)


def per_class_figure(pc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import apply_style, color_for_model
    apply_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    x, w = np.arange(len(CLASSES)), 0.2
    for i, (name, label) in enumerate(FIG_MODELS):
        d = pc[pc["model"] == name].set_index("class").reindex(CLASSES)
        yerr = np.array([(d["f1"] - d["f1_lo"]).clip(lower=0), (d["f1_hi"] - d["f1"]).clip(lower=0)])
        ax.bar(x + (i - 1.5) * w, d["f1"], w, label=label, color=color_for_model(label),
               edgecolor="white", linewidth=0.5, yerr=yerr,
               error_kw=dict(elinewidth=0.9, capsize=1.8, ecolor="0.3"))
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Out-of-region F1 (September filled)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"per_class_f1_filled.{ext}"), dpi=200)
    plt.close(fig)


def merge_cm(cm, mapping):
    cm = cm.rename(index=lambda c: mapping.get(c, c), columns=lambda c: mapping.get(c, c))
    cm = cm.groupby(level=0).sum().T.groupby(level=0).sum().T
    labels = sorted(set(cm.index) | set(cm.columns))
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def cereal(gt, preds):
    rows = []
    for name, p in preds.items():
        m = pd.concat([gt.rename("t"), p.rename("p")], axis=1, join="inner")
        cm = pd.crosstab(m["t"], m["p"])
        for scheme, mapping in SCHEMES.items():
            mc = merge_cm(cm, mapping)
            f1m, k, acc = metrics(mc.values)
            pc = dict(zip(mc.index, per_class_f1(mc.values)))
            rows.append({"model": name, "scheme": scheme, "macro_f1": f1m, "kappa": k, "accuracy": acc,
                         "f1_merged_class": pc.get("Cereals", pc.get("Wheat/Barley", np.nan))})
    return pd.DataFrame(rows)


def per_seed(gt, cidx, K):
    rows = []
    for name, key in SEED_MODELS.items():
        for variant in ("orig", "trainmed"):
            files = sorted(glob.glob(os.path.join(sc.PRED, f"{key}_{variant}_seed*.csv")))
            f1s = []
            for f in files:
                m = pd.concat([gt.rename("t"), sc.read(f).rename("p")], axis=1, join="inner")
                f1s.append(metrics(confusion(m["t"].map(cidx).values, m["p"].map(cidx).values, K))[0])
            m = pd.concat([gt.rename("t"), sc.read(os.path.join(sc.PRED, f"{key}_{variant}.csv")).rename("p")], axis=1, join="inner")
            ens = metrics(confusion(m["t"].map(cidx).values, m["p"].map(cidx).values, K))[0]
            rows.append({"model": name, "fill": {"orig": "published", "trainmed": "training median"}[variant],
                         "n_seeds": len(f1s), "seed_mean_f1m": np.mean(f1s), "seed_sd_f1m": np.std(f1s, ddof=1),
                         "seed_min_f1m": np.min(f1s), "seed_max_f1m": np.max(f1s), "ensemble_f1m": ens})
    return pd.DataFrame(rows)


def main():
    gt = sc.load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    classes = sorted(gt.unique())
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    preds = {name: sc.read(os.path.join(sc.PRED, f"{key}_trainmed.csv")) for name, (_, key) in sc.MODELS.items()}
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    pd.set_option("display.width", 220)

    pb = paired(gt, preds, cidx, K, np.random.default_rng(SEED))
    pb.round(4).to_csv(os.path.join(RES, "paired_bootstrap_filled.csv"), index=False)
    print("=== paired bootstrap, filled predictions (macro-F1) ===")
    print(pb[pb.metric == "macro_f1"].drop(columns="metric").round(3).to_string(index=False))

    pc = per_class(gt, preds, classes, cidx, K, np.random.default_rng(SEED))
    pc.round(4).to_csv(os.path.join(RES, "per_class_filled.csv"), index=False)
    per_class_figure(pc)
    print("\n=== per-class F1, filled (figure models) ===")
    print(pc[pc.model.isin([m for m, _ in FIG_MODELS])].pivot(index="model", columns="class", values="f1")[CLASSES].round(3).to_string())

    ce = cereal(gt, preds)
    ce.round(4).to_csv(os.path.join(RES, "cereal_merge_filled.csv"), index=False)
    piv = ce.pivot(index="model", columns="scheme", values="macro_f1")[list(SCHEMES)]
    piv["gain_all_cereals"] = piv["all cereals"] - piv["5-class"]
    print("\n=== cereal merge, filled (macro-F1) ===")
    print(piv.sort_values("5-class", ascending=False).round(3).to_string())

    ps = per_seed(gt, cidx, K)
    ps.round(4).to_csv(os.path.join(RES, "per_seed_sd_filled.csv"), index=False)
    print("\n=== per-seed SD ===")
    print(ps.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
