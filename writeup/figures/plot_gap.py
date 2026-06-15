#!/usr/bin/env python
"""In-region vs spatial-holdout F1-macro gap (headline figure), dumbbell chart.
Source: out_of_sample/scoring_results/f1_macro_train_vs_oos.csv
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import apply_style, COND_INREGION, COND_HOLDOUT

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "out_of_sample", "scoring_results", "f1_macro_train_vs_oos.csv")
OUT = os.path.join(REPO, "writeup", "figures", "train_vs_oos_gap.pdf")

df = pd.read_csv(SRC).dropna(subset=["Train F1 (macro)", "OOS F1 (macro)"]).copy()
df["Delta"] = df["OOS F1 (macro)"] - df["Train F1 (macro)"]
df = df.sort_values("Delta")

fig, ax = plt.subplots(figsize=(9.5, 8.5))
y = range(len(df))
for yi, (_, r) in zip(y, df.iterrows()):
    ax.plot([r["Train F1 (macro)"], r["OOS F1 (macro)"]], [yi, yi],
            color="0.75", lw=2.2, zorder=1)
ax.scatter(df["Train F1 (macro)"], list(y), color=COND_INREGION, s=90, zorder=3,
           label="In-region (field-wise validation)", edgecolor="white", linewidth=0.6)
ax.scatter(df["OOS F1 (macro)"], list(y), color=COND_HOLDOUT, s=90, zorder=3,
           label="Spatial transfer (ST)", edgecolor="white", linewidth=0.6)
ax.set_yticks(list(y))
ax.set_yticklabels(df["Model"])
ax.set_xlabel("F1 (macro)")
ax.set_title("In-region validation vs. spatial transfer")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
ax.grid(axis="x")
ax.set_axisbelow(True)
fig.savefig(OUT)
print("wrote", OUT)
