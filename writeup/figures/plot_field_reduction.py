#!/usr/bin/env python
"""Deliverable A / Fig: out-of-sample F1-macro vs. training-set size (field reduction).

One line per model across training fractions 1.00 / 0.75 / 0.50 / 0.25.
Source: experiments/field_reduction/results/field_reduction_results.csv
"""
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "experiments", "field_reduction", "results", "field_reduction_results.csv")
OUT = os.path.join(REPO, "writeup", "field_reduction.pdf")

df = pd.read_csv(SRC)

# Curated set telling the data-efficiency / feature-selection story.
MODELS = [
    ("TabNet (pixel)", "#4c72b0", "o", "-"),
    ("L-TAE (pixel)", "#55a868", "s", "-"),
    ("XGBoost (field)", "#8172b3", "^", "-"),
    ("Base LR (pixel)", "#c44e52", "D", "-"),
    ("LassoNet (pixel)", "#937860", "v", "--"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.6))
for name, color, marker, ls in MODELS:
    sub = df[df["Model"] == name].sort_values("Fraction")
    if sub.empty:
        print("WARNING: no rows for", name)
        continue
    ax.plot(sub["Fraction"], sub["OOS F1 (macro)"], color=color, marker=marker,
            ls=ls, lw=1.8, ms=6, label=name)

ax.set_xlabel("Fraction of training fields retained")
ax.set_ylabel("Out-of-sample F1 (macro)")
ax.set_title("Data efficiency under spatial holdout")
ax.set_xticks([0.25, 0.50, 0.75, 1.00])
ax.invert_xaxis()
ax.grid(color="0.9")
ax.legend(frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
