#!/usr/bin/env python
"""Out-of-sample F1-macro vs training-set size (field reduction).
Source: experiments/field_reduction/results/field_reduction_results.csv
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import apply_style, color_for_model

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "experiments", "field_reduction", "results", "field_reduction_results.csv")
OUT = os.path.join(REPO, "writeup", "figures", "field_reduction.pdf")

df = pd.read_csv(SRC)

# (csv name, display label, marker)
MODELS = [
    ("TabNet (pixel)", "TabNet", "o"),
    ("L-TAE (pixel)", "L-TAE", "s"),
    ("XGBoost (field)", "XGBoost", "^"),
    ("Base LR (pixel)", "Logistic Regression", "D"),
    ("LassoNet (pixel)", "LassoNet", "v"),
]

fig, ax = plt.subplots(figsize=(9.0, 6.0))
for name, label, marker in MODELS:
    sub = df[df["Model"] == name].sort_values("Fraction")
    if sub.empty:
        print("WARNING: no rows for", name); continue
    ls = "--" if label == "LassoNet" else "-"
    ax.plot(sub["Fraction"], sub["OOS F1 (macro)"], color=color_for_model(label),
            marker=marker, ls=ls, label=label)

ax.set_xlabel("Fraction of training fields retained")
ax.set_ylabel("Spatial-transfer F1 (macro)")
ax.set_title("Data efficiency under spatial transfer")
ax.set_xticks([0.25, 0.50, 0.75, 1.00])
ax.invert_xaxis()
ax.grid(True)
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.savefig(OUT)
print("wrote", OUT)
