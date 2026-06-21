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
    "Tree-based": (
        ["Base RF (pixel)", "Base XGBoost (pixel)", "Base LightGBM (pixel)",
         "XGBoost (field)", "LightGBM (field)", "TabNet (pixel)"], FAMILY_COLORS["tree"]),
    "Linear": (["Base LR (pixel)"], FAMILY_COLORS["linear"]),
    "Dense temporal / patch DL": (
        ["CNN-BiLSTM (pixel)", "TempCNN (pixel)", "L-TAE (pixel)", "Transformer (pixel)",
         "3D CNN (patch)", "Multi-Ch CNN (patch)"],
        FAMILY_COLORS["dense"]),
    "Sparse-attention": (["L-TAE-S (pixel)"], FAMILY_COLORS["sparse"]),
    "Synthetic augmentation": (["SMOTE Stacked (field)"], FAMILY_COLORS["aug"]),
}

SHORT_NAMES = {
    "Base RF (pixel)": "RF",
    "Base XGBoost (pixel)": "XGBoost (px)",
    "Base LightGBM (pixel)": "LightGBM (px)",
    "XGBoost (field)": "XGBoost (fd)",
    "LightGBM (field)": "LightGBM (fd)",
    "TabNet (pixel)": "TabNet",
    "Base LR (pixel)": "LogReg",
    "CNN-BiLSTM (pixel)": "CNN-BiLSTM",
    "TempCNN (pixel)": "TempCNN",
    "L-TAE (pixel)": "L-TAE",
    "Transformer (pixel)": "Transformer",
    "3D CNN (patch)": "3D CNN",
    "Multi-Ch CNN (patch)": "Multi-Ch CNN",
    "L-TAE-S (pixel)": "L-TAE-S",
    "SMOTE Stacked (field)": "SMOTE Stacked",
}

fig, ax = plt.subplots(figsize=(9.5, 6.2))
means = []
for i, (fam, (models, color)) in enumerate(FAMILIES.items()):
    sub = df[df["Model"].isin(models)][["Model", "Delta"]].reset_index(drop=True)
    vals = sub["Delta"].values
    names = sub["Model"].tolist()
    m = float(np.mean(vals)) if len(vals) else np.nan
    means.append(m)
    ax.bar(i, m, 0.6, color=color, alpha=0.55, zorder=1)
    jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.22
    xpos = np.full(len(vals), i) + jitter
    ax.scatter(xpos, vals, color=color, edgecolor="black",
               s=70, zorder=3, linewidth=0.6)

    if len(vals) == 0:
        continue
    best_idx = int(np.argmax(vals))    # least negative Δ
    worst_idx = int(np.argmin(vals))   # most negative Δ
    label_indices = [best_idx] if best_idx == worst_idx else [best_idx, worst_idx]
    for idx in label_indices:
        ax.annotate(
            SHORT_NAMES.get(names[idx], names[idx]),
            xy=(xpos[idx], vals[idx]),
            xytext=(9, 0), textcoords="offset points",
            ha="left", va="center", fontsize=10, color="0.2",
            zorder=4,
        )

ax.axhline(0, color="0.4", lw=1.0)
ax.set_xticks(range(len(FAMILIES)))
ax.set_xticklabels(list(FAMILIES.keys()), rotation=30, ha="right")
ax.set_ylabel(r"$\Delta$ F1-macro (ST $-$ in-region)")
ax.set_title("Spatial-transfer gap by inductive-bias family")
ax.grid(axis="y")
ax.set_axisbelow(True)
fig.savefig(OUT)
print("wrote", OUT)
for fam, m in zip(FAMILIES, means):
    print(f"{fam.splitlines()[0]:24s} mean Delta = {m:+.3f}")
