#!/usr/bin/env python
"""One figure: macro F1 under conditions a to e' per direction (fold means, thin bars =
range over folds; random-removal controls show the range of the 10 draws averaged over folds)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from experiments.external_audits.common import plotting  # noqa: E402

RES = HERE / "results"
LAB = {"a": "as\npublished", "b": "cloud\nremoved", "e": "random\nremoval", "c": "shift", "d": "shift +\ncloud\nremoved", "e_shift": "shift +\nrandom\nremoval"}
ORDER = ["a", "b", "e", "c", "d", "e_shift"]
TITLE = {"DK1_to_FR1": "DK1 to FR1 (target cloud: Cloud Score+ per parcel)", "FR1_to_DK1": "FR1 to DK1 (target cloud: emulated Cloud Score+)"}


def main() -> None:
    plotting.theme()
    ft = pd.read_csv(RES / "conditions_by_fold.csv")
    ef = pd.read_csv(RES / "cloud_effect_by_fold.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, d in zip(axes, ["DK1_to_FR1", "FR1_to_DK1"]):
        src = "csplus_parcel" if d == "DK1_to_FR1" else "emulator"
        g = ft[(ft.direction == d) & ((ft.cloud_source == src) | (ft.cloud_source == "none"))]
        g = g[(g.threshold == 0.6) | g.threshold.isna()]
        for x, c in enumerate(ORDER):
            v = g[g.condition == c].sort_values("fold")["macro_f1"].values * 100
            col = plotting.SERIES[1] if c in ("b", "d") else (plotting.INK_SECONDARY if c in ("e", "e_shift") else plotting.SERIES[0])
            ax.bar(x, v.mean(), color=col, width=0.62)
            ax.plot([x, x], [v.min(), v.max()], color=plotting.INK, lw=1.0)
            ax.text(x, v.mean() + 1.2, f"{v.mean():.1f}", ha="center", fontsize=8)
        ub = ef[ef.direction == d]["upper"].mean() * 100
        ax.axhline(ub, color=plotting.SERIES[2], ls="--", lw=1.2)
        ax.text(5.45, ub + 1, f"target-trained {ub:.1f}", ha="right", fontsize=8, color=plotting.SERIES[2])
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LAB[c] for c in ORDER], fontsize=7.5)
        ax.set_title(TITLE[d], fontsize=9)
    axes[0].set_ylabel("macro F1 (%), mean of 3 folds")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    plotting.save(fig, HERE / "figures" / "timematch_rerun_conditions")


if __name__ == "__main__":
    main()
