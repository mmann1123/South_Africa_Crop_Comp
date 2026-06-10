#!/usr/bin/env python
"""Deliverable A / Fig: in-region vs spatial-holdout F1-macro gap (the headline figure).

Dumbbell chart: for each model, the in-region (k-fold/validation) F1-macro and the
true out-of-sample holdout F1-macro, connected by a line, sorted by the drop (Delta).
Source: out_of_sample/scoring_results/f1_macro_train_vs_oos.csv
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "out_of_sample", "scoring_results", "f1_macro_train_vs_oos.csv")
OUT = os.path.join(REPO, "writeup", "train_vs_oos_gap.pdf")

df = pd.read_csv(SRC)
df = df.dropna(subset=["Train F1 (macro)", "OOS F1 (macro)"]).copy()
df["Delta"] = df["OOS F1 (macro)"] - df["Train F1 (macro)"]
df = df.sort_values("Delta")  # most-overfit first (most negative)

fig, ax = plt.subplots(figsize=(7.2, 6.4))
y = range(len(df))
for yi, (_, r) in zip(y, df.iterrows()):
    ax.plot([r["Train F1 (macro)"], r["OOS F1 (macro)"]], [yi, yi],
            color="0.7", lw=1.6, zorder=1)
ax.scatter(df["Train F1 (macro)"], list(y), color="#c44e52", s=42, zorder=2,
           label="In-region (field-wise k-fold)")
ax.scatter(df["OOS F1 (macro)"], list(y), color="#4c72b0", s=42, zorder=2,
           label="True spatial holdout")
ax.set_yticks(list(y))
ax.set_yticklabels(df["Model"], fontsize=8)
ax.set_xlabel("F1 (macro)")
ax.set_title("In-region validation vs. spatial-holdout generalization")
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.grid(axis="x", color="0.9")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
print(df[["Model", "Train F1 (macro)", "OOS F1 (macro)", "Delta"]].to_string(index=False))
