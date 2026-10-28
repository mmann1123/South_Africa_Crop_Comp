#!/usr/bin/env python
"""Per-tile and monthly cloud summaries plus the four-panel figure for TimeMatch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch import tmsummary  # noqa: E402

PAPER = "timematch"
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
ORDER = ["AT1", "DK1", "FR1", "FR2"]


def figure(df: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["label"] = d["tile"].map(tmsummary.LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for ax, label in zip(axes.ravel(), ORDER):
        g = d[d["label"] == label].sort_values("date")
        gi = g[g["n_images"] > 0]
        ax.plot(g["date"], g["tile_cloudy_pct"] / 100, color=plotting.INK_SECONDARY, lw=0.8, alpha=0.6, label="tile-level cloudy % (metadata)")
        ax.scatter(gi["date"], gi["frac_cloudy_csplus"], s=16, color=plotting.SERIES[0], label="fields cloudy, Cloud Score+ (cs_cdf < 0.6)", zorder=3)
        ax.scatter(gi["date"], gi["frac_cloudy_s2cp40"], s=16, facecolors="none", edgecolors=plotting.SERIES[1], lw=0.9, label="fields cloudy, s2cloudless ≥ 40 %", zorder=3)
        ax.axhline(0.5, color=plotting.GRID, lw=0.8, ls="--")
        tile = g["tile"].iloc[0]
        ax.set_title(f"{label} ({tile}), {len(g)} shipped dates, 2017")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction cloudy over cropland")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return plotting.save(fig, out)


def main() -> None:
    src = RES / "cloud_by_date.csv"
    with RunLog("03_summarise", {}, paper=PAPER) as log:
        df = pd.read_csv(src)
        log.count("rows_in", int(len(df)))
        log.count("dates_without_images", int((df["n_images"] == 0).sum()))
        log.count("dates_failed", int((df["n_images"] < 0).sum()))
        tiles = tmsummary.tile_summary(df[df["n_images"] >= 0])
        monthly = tmsummary.monthly_summary(df[df["n_images"] >= 0])
        write_csv(tiles, RES / "tile_summary.csv", "03_summarise", [str(src)])
        write_csv(monthly, RES / "monthly_summary.csv", "03_summarise", [str(src)])
        log.add("figure", str(figure(df[df["n_images"] >= 0], FIG / "timematch_cloud_by_date.png")))
        cols = ["label", "n_dates", "mean_tile_cloudy_pct", "mean_field_cloudy_csplus", "dates_field_over50_csplus", "expected_cloudy_obs_per_series_csplus", "expected_clear_obs_per_series_csplus", "mean_field_cloudy_s2cp40", "expected_cloudy_obs_per_series_s2cp40"]
        print(tiles[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
