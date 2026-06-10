#!/usr/bin/env python
"""Deliverable A / Fig: pixel vs field-level OOS F1-macro, paired per architecture.

Shows field-level aggregation as a variance-reduction lever for dense temporal
nets (L-TAE, TempCNN) and the TabNet exception (field hurts).
Source: out_of_sample/scoring_results/f1_macro_train_vs_oos.csv
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
OUT = os.path.join(REPO, "writeup", "pixel_vs_field.pdf")

df = pd.read_csv(SRC).set_index("Model")

# (architecture label, pixel-variant row, field-variant row)
PAIRS = [
    ("L-TAE", "L-TAE (pixel)", "L-TAE Field (field)"),
    ("TempCNN", "TempCNN (pixel)", "TempCNN Field (field)"),
    ("TabNet", "TabNet (pixel)", "TabNet Field (field)"),
    ("XGBoost", "Base XGBoost (pixel)", "XGBoost (field)"),
    ("LightGBM", "Base LightGBM (pixel)", "LightGBM (field)"),
]

labels, pix, fld = [], [], []
for arch, pname, fname in PAIRS:
    labels.append(arch)
    pix.append(df.loc[pname, "OOS F1 (macro)"] if pname in df.index else np.nan)
    fld.append(df.loc[fname, "OOS F1 (macro)"] if fname in df.index else np.nan)

x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.bar(x - w / 2, pix, w, label="Pixel level", color="#4c72b0")
ax.bar(x + w / 2, fld, w, label="Field level (aggregated)", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Out-of-sample F1 (macro)")
ax.set_title("Pixel vs. field-level aggregation")
ax.set_ylim(0, 0.7)
ax.grid(axis="y", color="0.9")
ax.legend(frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
for arch, p, f in zip(labels, pix, fld):
    print(f"{arch:10s} pixel={p:.3f} field={f:.3f}")
