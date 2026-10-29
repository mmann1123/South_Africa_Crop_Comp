#!/usr/bin/env python
"""DACCN snow check summaries: region-year table, SCL/MODIS cross-check, transfer pairs, by-step figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.daccn import regions, snowrule  # noqa: E402

PAPER = "daccn"
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
ORDER = [r[0] for r in regions.REGIONS]
SRC = RES / "daccn_snow_counters.csv"


def figure(summary: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True, sharey=True)
    steps = np.arange(1, 23)
    for ax, region in zip(axes.ravel(), ORDER):
        for year, color in ((2018, plotting.SERIES[0]), (2019, plotting.SERIES[1])):
            for variant, ls in (("A", "-"), ("B", "--")):
                r = summary[(summary["region"] == region) & (summary["year"] == year) & (summary["variant"] == variant)]
                if r.empty:
                    continue
                z = np.array([float(v) for v in r["share_snow_by_step"].iloc[0].split(";")])
                ax.plot(steps, z, color=color, ls=ls, label=f"{year} variant {variant}")
        ax.set_title(region)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 6, 11, 16, 22])
    axes[1, 1].set_xlabel("10-day composite step (1 = DOY 90, 22 = DOY 300)")
    fig.supylabel("share of valid composites with snow", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return plotting.save(fig, out)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError("daccn_snow_counters.csv not found; run 04_snow_counters.py")
    with RunLog("05_snow_summarise", {}, paper=PAPER) as log:
        df = pd.read_csv(SRC)
        log.count("rows_in", int(len(df)))
        summary = snowrule.region_year_snow(df)
        cross = snowrule.crosscheck(df[df["variant"] == "B"])  # variants differ only at the edge steps; B is the DACCN-stated window
        pairs = snowrule.pairs_snow(summary, regions.SPATIAL_PAIRS)
        write_csv(summary, RES / "daccn_snow_summary.csv", "05_snow_summarise", [str(SRC)])
        write_csv(cross, RES / "daccn_snow_crosscheck.csv", "05_snow_summarise", [str(SRC)])
        write_csv(pairs, RES / "daccn_snow_pairs.csv", "05_snow_summarise", [str(SRC)])
        log.add("figure", str(figure(summary, FIG / "daccn_snow_by_step.png")))
        pd.set_option("display.width", 250)
        print(summary.drop(columns=["share_snow_by_step"]).round(3).to_string(index=False))
        print(cross.round(3).to_string(index=False))
        print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
