#!/usr/bin/env python
"""Pixel vs field-level OOS F1-macro, paired per architecture.
Source: out_of_sample/scoring_results/f1_macro_train_vs_oos.csv
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figstyle import apply_style, COND_PIXEL, COND_FIELD

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "out_of_sample", "scoring_results", "f1_macro_train_vs_oos.csv")
OUT = os.path.join(REPO, "writeup", "figures", "pixel_vs_field.pdf")

df = pd.read_csv(SRC).set_index("Model")

# (architecture label, pixel-variant row, field-variant row)
PAIRS = [
    ("L-TAE", "L-TAE (pixel)", "L-TAE Field (field)"),
    ("TempCNN", "TempCNN (pixel)", "TempCNN Field (field)"),
    ("TabNet", "TabNet (pixel)", "TabNet Temporal Field (field)"),
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
fig, ax = plt.subplots(figsize=(9.0, 5.8))
ax.bar(x - w / 2, pix, w, label="Pixel level", color=COND_PIXEL,
       edgecolor="white", linewidth=0.5)
ax.bar(x + w / 2, fld, w, label="Field level (aggregated)", color=COND_FIELD,
       edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Out-of-sample F1 (macro)")
ax.set_title("Pixel vs. field-level aggregation")
ax.set_ylim(0, 0.72)
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.savefig(OUT)
print("wrote", OUT)
for a, p, f in zip(labels, pix, fld):
    print(f"{a:10s} pixel={p:.3f} field={f:.3f}")
