#!/usr/bin/env python
"""DACCN summary tables, transfer-pair table and the zero-by-step figure."""
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
from experiments.external_audits.daccn import dcsummary, regions  # noqa: E402

PAPER = "daccn"
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
ORDER = [r[0] for r in regions.REGIONS]


def load_counters() -> tuple[pd.DataFrame, Path]:
    for p in (RES / "daccn_counters.csv", RES / "daccn_counters.csv.gz"):
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError("daccn_counters.csv(.gz) not found; run 02_run_counters.py")


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
                z = np.array([float(v) for v in r["share_zero_by_step"].iloc[0].split(";")])
                ax.plot(steps, z, color=color, ls=ls, label=f"{year} variant {variant}")
        ax.set_title(region)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 6, 11, 16, 22])
    for ax in axes[1]:
        ax.set_xlabel("10-day step (1 = DOY 90, 22 = DOY 300)")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of pixels zero-filled")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return plotting.save(fig, out)


def main() -> None:
    with RunLog("03_summarise", {}, paper=PAPER) as log:
        df, src = load_counters()
        log.count("rows_in", int(len(df)))
        summary = dcsummary.region_year_summary(df)
        pairs = dcsummary.pairs_table(df, regions.SPATIAL_PAIRS)
        write_csv(summary, RES / "daccn_summary.csv", "03_summarise", [str(src)])
        write_csv(pairs, RES / "daccn_pairs.csv", "03_summarise", [str(src)])
        log.add("figure", str(figure(summary, FIG / "daccn_zero_by_step.png")))
        cols = ["region", "year", "variant", "n_points", "mean_n_valid", "share_any_zero", "mean_n_zero", "share_zero_step01", "share_zero_step22"]
        print(summary[cols].round(3).to_string(index=False))
        print(pairs[["kind", "source", "target", "variant", "source_share_any_zero", "target_share_any_zero", "difference"]].round(3).to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
