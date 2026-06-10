#!/usr/bin/env python
"""Deliverable A / Fig: per-class out-of-sample F1 for selected models.

Grouped bars showing where the holdout gap concentrates (Wheat / Barley /
Small grain grazing) vs. the easy classes (Lucerne/Medics, Canola).
Source: out_of_sample/scoring_results/per_class_<Model>.csv
"""
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCORES = os.path.join(REPO, "out_of_sample", "scoring_results")
OUT = os.path.join(REPO, "writeup", "per_class_f1.pdf")

# Display name -> color. Best OOS, a robust tree/field model, and the in-region hero that collapses.
MODELS = [
    ("TabNet (pixel)", "#4c72b0"),
    ("XGBoost (field)", "#8172b3"),
    ("Base LR (pixel)", "#55a868"),
    ("CNN-BiLSTM (pixel)", "#c44e52"),
]
CLASSES = ["Lucerne/Medics", "Canola", "Barley", "Wheat", "Small grain grazing"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Build a normalized lookup of available per-class files.
files = {}
for p in glob.glob(os.path.join(SCORES, "per_class_*.csv")):
    key = norm(os.path.basename(p)[len("per_class_"):-len(".csv")])
    files[key] = p

fig, ax = plt.subplots(figsize=(7.6, 4.8))
x = np.arange(len(CLASSES))
w = 0.2
for i, (name, color) in enumerate(MODELS):
    path = files.get(norm(name))
    if path is None:
        print("WARNING: no per-class file for", name)
        continue
    d = pd.read_csv(path).set_index("class")
    vals = [d.loc[c, "f1-score"] if c in d.index else np.nan for c in CLASSES]
    ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=color)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Out-of-sample F1")
ax.set_title("Per-class holdout performance")
ax.set_ylim(0, 1)
ax.grid(axis="y", color="0.9")
ax.legend(fontsize=8, frameon=False, ncol=2)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
