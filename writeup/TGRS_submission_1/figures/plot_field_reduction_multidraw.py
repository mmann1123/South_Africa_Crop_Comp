#!/usr/bin/env python
"""Data efficiency under spatial transfer -- multi-draw version of field_reduction.pdf.

Same five models and styling as plot_field_reduction.py, but each point is the MEAN over 5
independent subsample draws (single training seed) instead of the single seed-42 draw used in
the original. Averaging over draws removes the spurious L-TAE-S "dip" at the 50% fraction, which
was an unlucky draw on the one draw-sensitive model. The 100% point is also single-seed so the
whole curve is on one ensemble level (~0.05 below the 5-seed Table I numbers; the band-width /
robustness, not the absolute height, is the point of this figure).

Source: experiments/field_reduction/results/seed_sweep_var.csv
Set SHOW_ERR=True to draw +/-1 s.d. (over draws) error bars.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import apply_style, color_for_model

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "experiments", "field_reduction", "results", "seed_sweep_var.csv")
OUT = os.path.join(REPO, "writeup", "figures", "field_reduction_multidraw.pdf")

SHOW_ERR = True  # flip to True for +/-1 s.d. over draws

# (csv model key, display label, marker) -- matches plot_field_reduction.py
MODELS = [
    ("tabnet_pixel",      "TabNet",              "o"),
    ("ltae_pixel",        "L-TAE",               "s"),
    ("ltae_sparse_pixel", "L-TAE-S",             "v"),
    ("xgboost_field",     "XGBoost",             "^"),
    ("base_lr_pixel",     "Logistic Regression", "D"),
]

df = pd.read_csv(SRC)
fig, ax = plt.subplots(figsize=(9.0, 6.0))
for key, label, marker in MODELS:
    sub = df[df["model"] == key]
    if sub.empty:
        print("WARNING: no rows for", key); continue
    g = sub.groupby("fraction")["oos_f1_macro"].agg(["mean", "std"]).reset_index().sort_values("fraction")
    ls = "--" if label == "L-TAE-S" else "-"
    if SHOW_ERR:
        ax.errorbar(g["fraction"], g["mean"], yerr=g["std"].fillna(0.0),
                    color=color_for_model(label), marker=marker, ls=ls, label=label,
                    capsize=3, elinewidth=1.0)
    else:
        ax.plot(g["fraction"], g["mean"], color=color_for_model(label), marker=marker, ls=ls, label=label)

ax.set_xlabel("Fraction of training fields retained")
ax.set_ylabel("Spatial-transfer F1 (macro)")
ax.set_title("Data efficiency under spatial transfer")
ax.set_xticks([0.25, 0.50, 0.75, 1.00])
ax.invert_xaxis()
ax.grid(True)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize="small")
fig.savefig(OUT)
print("wrote", OUT)
