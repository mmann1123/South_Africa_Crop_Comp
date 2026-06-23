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
CI = os.path.join(REPO, "out_of_sample", "scoring_results", "bootstrap_ci.csv")
OUT = os.path.join(REPO, "writeup", "figures", "pixel_vs_field.pdf")

df = pd.read_csv(SRC).set_index("Model")
ci = pd.read_csv(CI).set_index("Model")


def yerr_for(name):
    """Asymmetric [lower, upper] 95% bootstrap CI half-widths, or [0,0] if absent."""
    if name in ci.index and name in df.index:
        pt = df.loc[name, "OOS F1 (macro)"]
        return [max(0.0, pt - ci.loc[name, "F1_lo"]), max(0.0, ci.loc[name, "F1_hi"] - pt)]
    return [0.0, 0.0]

# (architecture label, pixel-variant row, field-variant row)
PAIRS = [
    ("L-TAE", "L-TAE (pixel)", "L-TAE Field (field)"),
    ("L-TAE-S", "L-TAE-S (pixel)", "L-TAE-S Field (field)"),
    ("TempCNN", "TempCNN (pixel)", "TempCNN Field (field)"),
    ("CNN-BiLSTM", "CNN-BiLSTM (pixel)", "CNN-BiLSTM Field (field)"),
    ("TabNet", "TabNet (pixel)", "TabNet Temporal Field (field)"),
    ("XGBoost", "Base XGBoost (pixel)", "XGBoost (field)"),
    ("LightGBM", "Base LightGBM (pixel)", "LightGBM (field)"),
]

labels, pix, fld = [], [], []
pix_err, fld_err = [], []
for arch, pname, fname in PAIRS:
    labels.append(arch)
    pix.append(df.loc[pname, "OOS F1 (macro)"] if pname in df.index else np.nan)
    fld.append(df.loc[fname, "OOS F1 (macro)"] if fname in df.index else np.nan)
    pix_err.append(yerr_for(pname))
    fld_err.append(yerr_for(fname))

# transpose to matplotlib's expected (2, N) shape: row0=lower, row1=upper
pix_err = np.array(pix_err).T
fld_err = np.array(fld_err).T

x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(9.0, 5.8))
ax.bar(x - w / 2, pix, w, label="Pixel level", color=COND_PIXEL,
       edgecolor="white", linewidth=0.5,
       yerr=pix_err, error_kw=dict(elinewidth=1.1, capsize=2.5, ecolor="0.25"))
ax.bar(x + w / 2, fld, w, label="Field level (aggregated)", color=COND_FIELD,
       edgecolor="white", linewidth=0.5,
       yerr=fld_err, error_kw=dict(elinewidth=1.1, capsize=2.5, ecolor="0.25"))
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Spatial-transfer F1 (macro)")
ax.set_title("Pixel vs. field-level aggregation")
ax.set_ylim(0, 0.72)
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.savefig(OUT)
print("wrote", OUT)
for a, p, f in zip(labels, pix, fld):
    print(f"{a:10s} pixel={p:.3f} field={f:.3f}")
