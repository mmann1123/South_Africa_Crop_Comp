"""Mechanism figure for the main text: share of September-missing holdout pixels whose 5-seed
ensemble prediction changes between zero fill and training-median fill, for the month-dropout x
gate arms of TempCNN and CNN-BiLSTM (results/attribution_september.csv, from attribution.py), next
to each arm's holdout macro-F1 under zero fill (month_dropout/results/cells.csv).
Writes figures/flip_rate.{pdf,png}. Run: python experiments/missingness/flip_rate_figure.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
from figstyle import apply_style  # noqa: E402

ARMS = ["dense", "gated", "dense + dropout", "gated + dropout"]
CELL = {"dense": (0, "dense"), "gated": (0, "sparse"), "dense + dropout": (1, "dense"), "gated + dropout": (1, "sparse")}
BB = {"tempcnn": "TempCNN", "cnn_bilstm": "CNN-BiLSTM"}

if __name__ == "__main__":
    a = pd.read_csv(os.path.join(HERE, "results", "attribution_september.csv"))
    a = a[a.pixels == "missing"].drop_duplicates(["backbone", "arm"]).set_index(["backbone", "arm"])["pred_flip_zero_vs_median"]
    c = pd.read_csv(os.path.join(HERE, "month_dropout", "results", "cells.csv"))
    c = c[(c.fill == "zero") & c.gate.isin(["dense", "sparse"])].set_index(["backbone", "dropout", "gate"])
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(ARMS)); w = 0.38
    col = {"tempcnn": "#d55e00", "cnn_bilstm": "#0072b2"}
    for k, bb in enumerate(BB):
        axes[0].bar(x + (k - 0.5) * w, [100 * a[(bb, arm)] for arm in ARMS], w, color=col[bb], label=BB[bb])
        f = [c.loc[(bb, *CELL[arm])] for arm in ARMS]
        axes[1].bar(x + (k - 0.5) * w, [r.f1m for r in f], w, color=col[bb], label=BB[bb],
                    yerr=[[r.f1m - r.ci_lo for r in f], [r.ci_hi - r.f1m for r in f]], capsize=2)
    axes[0].set_ylabel("Pixel predictions changed by the fill (%)", fontsize=9)
    axes[0].set_title("(a) Sensitivity to a zero-filled September", fontsize=10)
    axes[1].set_ylabel("Holdout macro-F1, zero fill", fontsize=9)
    axes[1].set_ylim(0.45, 0.70)
    axes[1].set_title("(b) Accuracy with the zero-filled September", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [s.replace(" + ", "\n+ ") for s in ARMS], fontsize=8.5)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"flip_rate.{ext}"), dpi=300)
