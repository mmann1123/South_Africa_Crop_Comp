#!/usr/bin/env python
"""Per continent: share of point-windows whose six-month median composite is empty (no valid observation),
share of points with any empty window, valid observations per window; per continent and window; and a
figure of the empty share by window."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.masolele import catalogue as cat, masolele_lib as lib  # noqa: E402

RES = paths.results_dir(cat.PAPER)
FIG = paths.figures_dir(cat.PAPER)
WINDOWS = lib.window_ids(cat.YEARS, cat.HALVES)
ORDER = list(cat.CONTINENTS)


def main() -> None:
    with RunLog("03_summarise", {}, paper=cat.PAPER) as log:
        df = pd.read_csv(RES / "window_counts.csv.gz")
        n0 = len(df)
        df = df.dropna(subset=lib.count_columns(WINDOWS))
        log.count("points_unsampled_dropped", n0 - len(df))
        lf = lib.long_form(df, WINDOWS)
        cs = lib.continent_summary(lf).set_index("continent").loc[ORDER].reset_index()
        cs["empty_share_ratio_max_over_min"] = lib.max_pairwise_ratio(dict(zip(cs["continent"], cs["empty_share"])))
        write_csv(cs, RES / "continent_summary.csv", "03_summarise", ["window_counts.csv.gz"])
        ws = lib.window_summary(lf)
        write_csv(ws, RES / "window_summary.csv", "03_summarise", ["window_counts.csv.gz"])
        print(cs.round(3).to_string())

        plotting.theme()
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for i, c in enumerate(ORDER):
            d = ws[ws["continent"] == c].sort_values("window")
            ax[0].plot(range(len(d)), d["empty_share"], "-o", ms=3, color=plotting.SERIES[i], label=c)
            ax[1].plot(range(len(d)), d["mean_obs"], "-o", ms=3, color=plotting.SERIES[i], label=c)
        for a, yl in zip(ax, ["share of points with an empty composite", "mean valid observations per composite"]):
            a.set_xticks(range(len(WINDOWS)), [w.replace("h1", " Jan-Jun").replace("h2", " Jul-Dec") for w in WINDOWS], rotation=60, ha="right", fontsize=7)
            a.set_ylabel(yl)
            a.axvline(6.5, color=plotting.INK_SECONDARY, lw=0.8, ls=":")
        ax[0].text(6.6, 0.005, "Landsat 7 SLC-off (31 May 2003)", fontsize=7, color=plotting.INK_SECONDARY)
        ax[0].legend(frameon=False, fontsize=8)
        ax[0].set_ylim(0, None)
        ax[1].set_ylim(0, None)
        fig.suptitle("Landsat 5/7 six-month composites at tropical 2001-2005 loss pixels (scene cloud < 50 %, QA clear)", fontsize=9)
        plotting.save(fig, FIG / "masolele_empty_composites.png")
        log.count("points", int(df.shape[0]))


if __name__ == "__main__":
    main()
