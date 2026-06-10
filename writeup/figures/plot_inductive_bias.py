#!/usr/bin/env python
"""Deliverable A / Fig: generalization gap (Delta) grouped by inductive-bias family.

Empirical backbone of the thesis: sparse/tree-like and low-capacity models transfer
(small |Delta|); dense temporal/patch deep nets and synthetic augmentation overfit.
Source: out_of_sample/scoring_results/f1_macro_train_vs_oos.csv

Caveats encoded explicitly: TabNet-field variants are excluded from the tree family
(aggregation confound) and L-TAE Field is flagged separately.
"""
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "out_of_sample", "scoring_results", "f1_macro_train_vs_oos.csv")
OUT = os.path.join(REPO, "writeup", "inductive_bias_gap.pdf")

df = pd.read_csv(SRC).dropna(subset=["Train F1 (macro)", "OOS F1 (macro)"]).copy()
df["Delta"] = df["OOS F1 (macro)"] - df["Train F1 (macro)"]

FAMILIES = {
    "Tree-based\n(RF/XGB/LGBM/TabNet-px)": (
        ["Base RF (pixel)", "Base XGBoost (pixel)", "Base LightGBM (pixel)",
         "XGBoost (field)", "LightGBM (field)", "TabNet (pixel)"], "#4c72b0"),
    "Linear\n(LogReg)": (["Base LR (pixel)"], "#55a868"),
    "Dense temporal /\npatch DL": (
        ["CNN-BiLSTM (pixel)", "TempCNN (pixel)", "L-TAE (pixel)", "3D CNN (patch)"], "#c44e52"),
    "Synthetic\naugmentation": (["SMOTE Stacked (field)"], "#937860"),
}

fig, ax = plt.subplots(figsize=(8.6, 5.4))
xpos = np.arange(len(FAMILIES))
means = []
for i, (fam, (models, color)) in enumerate(FAMILIES.items()):
    vals = df[df["Model"].isin(models)]["Delta"].values
    m = float(np.mean(vals)) if len(vals) else np.nan
    means.append(m)
    ax.bar(i, m, 0.6, color=color, alpha=0.55, zorder=1)
    jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.25
    ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, edgecolor="k",
               s=38, zorder=3, linewidth=0.4)

ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(list(FAMILIES.keys()), fontsize=11)
ax.set_ylabel(r"$\Delta$ F1-macro (OOS $-$ in-region)")
ax.set_title("Generalization gap by inductive-bias family")
ax.grid(axis="y", color="0.9")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
for fam, m in zip(FAMILIES, means):
    print(f"{fam.splitlines()[0]:24s} mean Delta = {m:+.3f}")
print("Note: TabNet Field/Temporal Field excluded from tree family; L-TAE Field (Delta -0.072) flagged separately in text.")
