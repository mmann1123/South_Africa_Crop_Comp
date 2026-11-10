#!/usr/bin/env python
"""Figure, LSTM only (TempCNN missed the reproduction gate): (a) what deleting a
whole month costs on each 33UVP year, under the published zero fill, with the
months each year actually lacks marked; (b, exploratory) a never-zero-trained
model with the month missing in 40.6 percent of pixels: cost with a median fill,
and the extra cost of zero fill on top. Means with 95 percent t intervals over 5 seeds."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.timesen2crop_rerun import tslib as tl  # noqa: E402

PAPER = "timesen2crop_rerun"
RES = paths.results_dir(PAPER)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
SURFACE = "#fcfcfb"


def ci(g: pd.Series) -> float:
    return stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))


def dots(ax, df, col, color, label, dx):
    a = df.groupby("month")[col].agg(["mean", ci]).reindex(tl.MONTHS)
    x = np.arange(12) + dx
    ax.errorbar(x, a["mean"], yerr=a["ci"], fmt="o", ms=6, lw=1.5, color=color, label=label, mec=SURFACE, mew=1, capsize=0)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True, facecolor=SURFACE)
    m = pd.read_csv(RES / "month_deletion.csv").query("model == 'lstm' and train_fill == 'Z'")
    m["cost"] = 100 * (m["f1m_asis"] - m["f1m_blanked"])
    seedm = m.groupby(["tile", "month", "seed"])["cost"].mean().reset_index()
    dots(ax[0], seedm[seedm["tile"] == tl.TEST_TILE], "cost", "#2a78d6", "33UVP 2018", -0.12)
    dots(ax[0], seedm[seedm["tile"] == tl.YEAR_TILE], "cost", "#eb6834", "33UVP 2019", 0.12)
    ax[0].set_title("(a) Deleting one month, published zero fill", loc="left", fontsize=11)
    e = pd.read_csv(RES / "exploratory_month_zero_vs_median.csv").query("model == 'lstm'")
    e["median_cost"] = 100 * (e["f1m_base"] - e["f1m_median"])
    e["extra_zero"] = 100 * (e["f1m_median"] - e["f1m_zero"])
    s = e.groupby(["month", "seed"])[["median_cost", "extra_zero"]].mean().reset_index()
    dots(ax[1], s, "median_cost", "#1baf7a", "Month lost, median fill", -0.12)
    dots(ax[1], s, "extra_zero", "#eda100", "Extra cost of zero fill", 0.12)
    ax[1].set_title("(b) Exploratory: never-zero model, month missing in 40.6%", loc="left", fontsize=11)
    for a in ax:
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xticks(range(12), tl.MONTHS)
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_facecolor(SURFACE)
        a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0), labelcolor=INK)
    ax[0].set_ylabel("Macro-F1 cost (points)")
    fig.tight_layout()
    out = paths.figures_dir(PAPER) / "timesen2crop_rerun.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
