#!/usr/bin/env python
"""Per-class out-of-sample F1 for selected models (grouped bars).
Source: out_of_sample/scoring_results/per_class_<Model>.csv
"""
import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figstyle import apply_style, color_for_model

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCORES = os.path.join(REPO, "out_of_sample", "scoring_results")
OUT = os.path.join(REPO, "writeup", "figures", "per_class_f1.pdf")

# Per-class 95% bootstrap CIs: index by (Model, class) -> (f1_lo, f1_hi).
_ci_path = os.path.join(SCORES, "bootstrap_per_class_ci.csv")
PCCI = pd.read_csv(_ci_path).set_index(["Model", "class"]) if os.path.exists(_ci_path) else None


def class_yerr(model_name, point_vals):
    """Asymmetric (2, n_class) CI half-widths aligned to CLASSES; zeros if absent."""
    lo, hi = [], []
    for c, pt in zip(CLASSES, point_vals):
        if PCCI is not None and (model_name, c) in PCCI.index and np.isfinite(pt):
            lo.append(max(0.0, pt - PCCI.loc[(model_name, c), "f1_lo"]))
            hi.append(max(0.0, PCCI.loc[(model_name, c), "f1_hi"] - pt))
        else:
            lo.append(0.0); hi.append(0.0)
    return np.array([lo, hi])

MODELS = [
    ("TabNet (pixel)", "TabNet"),
    ("L-TAE-S (pixel)", "L-TAE-S"),
    ("L-TAE (pixel)", "L-TAE"),
    ("CNN-BiLSTM (pixel)", "CNN-BiLSTM"),
]
CLASSES = ["Lucerne/Medics", "Canola", "Barley", "Wheat", "Small grain grazing"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


files = {}
for p in glob.glob(os.path.join(SCORES, "per_class_*.csv")):
    files[norm(os.path.basename(p)[len("per_class_"):-len(".csv")])] = p

fig, ax = plt.subplots(figsize=(10.0, 5.8))
x = np.arange(len(CLASSES))
w = 0.2
for i, (csvname, label) in enumerate(MODELS):
    path = files.get(norm(csvname))
    if path is None:
        print("WARNING: no per-class file for", csvname); continue
    d = pd.read_csv(path).set_index("class")
    vals = [d.loc[c, "f1-score"] if c in d.index else np.nan for c in CLASSES]
    ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color_for_model(label),
           edgecolor="white", linewidth=0.5,
           yerr=class_yerr(csvname, vals),
           error_kw=dict(elinewidth=0.9, capsize=1.8, ecolor="0.3"))

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=15, ha="right")
ax.set_ylabel("Spatial-transfer F1")
ax.set_title("Per-class spatial-transfer performance")
ax.set_ylim(0, 1)
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2, loc="upper right")
fig.savefig(OUT)
print("wrote", OUT)
