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

MODELS = [
    ("TabNet (pixel)", "TabNet"),
    ("XGBoost (field)", "XGBoost"),
    ("Base LR (pixel)", "Logistic Regression"),
    ("CNN-BiLSTM (field)", "CNN-BiLSTM"),
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
           edgecolor="white", linewidth=0.5)

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
