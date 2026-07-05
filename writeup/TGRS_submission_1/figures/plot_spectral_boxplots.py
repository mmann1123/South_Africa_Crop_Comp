#!/usr/bin/env python
"""Rebuild the spectral boxplots (all crops, and wheat vs barley).

Each pixel's value for a band is the mean across the available monthly observations;
every band is then z-scored across pixels so the six bands (which sit on very
different native scales) are comparable on one axis and cross-crop overlap is visible.
Source: data/merged_dl_train.parquet
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import apply_style, CROP_COLORS

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "data", "merged_dl_train.parquet")
OUT = os.path.join(REPO, "writeup", "figures")

BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
CROP_ORDER = ["Lucerne/Medics", "Wheat", "Barley", "Canola", "Small grain grazing"]
N_PER_CROP = 8000
RNG = np.random.default_rng(42)

# Read crop label + all band-month columns, build one mean-per-band value per pixel.
cols = pd.read_parquet(SRC, columns=None).columns
band_cols = {b: [c for c in cols if re.match(rf"^{re.escape(b)}_[A-Z]", c)] for b in BANDS}
need = ["crop_name"] + [c for cc in band_cols.values() for c in cc]
df = pd.read_parquet(SRC, columns=need)

feat = pd.DataFrame({"crop_name": df["crop_name"].values})
for b in BANDS:
    feat[b] = df[band_cols[b]].mean(axis=1).astype("float32")
del df

# z-score each band across all pixels
for b in BANDS:
    feat[b] = (feat[b] - feat[b].mean()) / feat[b].std()

# sample per crop for tractable, legible boxplots
sample = (feat.groupby("crop_name", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), N_PER_CROP), random_state=42)))


def boxplot(crops, fname, title):
    n = len(crops)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for j, crop in enumerate(crops):
        sub = sample[sample["crop_name"] == crop]
        data = [sub[b].values for b in BANDS]
        pos = [i + (j - (n - 1) / 2) * width for i in range(len(BANDS))]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.2))
        for box in bp["boxes"]:
            box.set(facecolor=CROP_COLORS[crop], edgecolor="black", linewidth=0.6, alpha=0.9)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color="0.4", linewidth=0.8)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS)
    ax.set_xlabel("Spectral band / index")
    ax.set_ylabel("Standardized intensity (z-score)")
    ax.set_title(title)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CROP_COLORS[c], edgecolor="black",
                             linewidth=0.6, alpha=0.9) for c in crops]
    ax.legend(handles, crops, frameon=False, ncol=min(len(crops), 3),
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(os.path.join(OUT, fname))
    print("wrote", fname)


boxplot(CROP_ORDER, "all_crops_boxplot.pdf",
        "Standardized spectral intensities by crop type")
boxplot(["Wheat", "Barley"], "wheat_barley_boxplot.pdf",
        "Wheat vs. barley: standardized spectral intensities")
