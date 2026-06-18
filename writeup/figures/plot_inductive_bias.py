#!/usr/bin/env python
"""Generalization gap (Delta) grouped by inductive-bias family.
Source: out_of_sample/scoring_results/f1_macro_train_vs_oos.csv
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figstyle import apply_style, FAMILY_COLORS

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "out_of_sample", "scoring_results", "f1_macro_train_vs_oos.csv")
OUT = os.path.join(REPO, "writeup", "figures", "inductive_bias_gap.pdf")

df = pd.read_csv(SRC).dropna(subset=["Train F1 (macro)", "OOS F1 (macro)"]).copy()
df["Delta"] = df["OOS F1 (macro)"] - df["Train F1 (macro)"]

FAMILIES = {
    "Tree-based\n(RF / XGB / LGBM / TabNet-px)": (
        ["Base RF (pixel)", "Base XGBoost (pixel)", "Base LightGBM (pixel)",
         "XGBoost (field)", "LightGBM (field)", "TabNet (pixel)"], FAMILY_COLORS["tree"]),
    "Linear\n(LogReg)": (["Base LR (pixel)"], FAMILY_COLORS["linear"]),
    "Dense temporal /\npatch DL": (
        ["CNN-BiLSTM (pixel)", "TempCNN (pixel)", "L-TAE (pixel)", "Transformer (pixel)",
         "3D CNN (patch)", "Multi-Ch CNN (patch)"],
        FAMILY_COLORS["dense"]),
    "Sparse-attention\n(L-TAE-S)": (["L-TAE-S (pixel)"], FAMILY_COLORS["sparse"]),
    "Synthetic\naugmentation": (["SMOTE Stacked (field)"], FAMILY_COLORS["aug"]),
}

fig, ax = plt.subplots(figsize=(9.0, 5.8))
means = []
for i, (fam, (models, color)) in enumerate(FAMILIES.items()):
    vals = df[df["Model"].isin(models)]["Delta"].values
    m = float(np.mean(vals)) if len(vals) else np.nan
    means.append(m)
    ax.bar(i, m, 0.6, color=color, alpha=0.55, zorder=1)
    jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.22
    ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, edgecolor="black",
               s=70, zorder=3, linewidth=0.6)

ax.axhline(0, color="0.4", lw=1.0)
ax.set_xticks(range(len(FAMILIES)))
ax.set_xticklabels(list(FAMILIES.keys()))
ax.set_ylabel(r"$\Delta$ F1-macro (ST $-$ in-region)")
ax.set_title("Spatial-transfer gap by inductive-bias family")
ax.grid(axis="y")
ax.set_axisbelow(True)
fig.savefig(OUT)
print("wrote", OUT)
for fam, m in zip(FAMILIES, means):
    print(f"{fam.splitlines()[0]:24s} mean Delta = {m:+.3f}")
