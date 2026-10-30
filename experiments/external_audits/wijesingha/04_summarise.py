#!/usr/bin/env python
"""Dropped-sample share and valid-observation counts per region-year, train versus test years,
sensor mix, and the figure."""
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
from experiments.external_audits.wijesingha import catalogue, wijesingha_lib as lib  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
BM = catalogue.BIMONTH_STARTS


def year_table(d: pd.DataFrame) -> pd.DataFrame:
    d = lib.dropna_flags(d, BM)
    rows = [{"region": r, "year": int(y), "period": "test" if y in catalogue.TEST_YEARS else "train", **lib.summarise_group(g, BM)} for (r, y), g in d.groupby(["region", "year"])]
    return pd.DataFrame(rows)


def sensor_mix(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (r, y), g in tab.groupby(["region", "year"]):
        row = {"region": r, "year": y}
        for s in lib.SENSOR_ORDER:
            row[f"mean_obs_{s}"] = float(np.mean([g[f"mean_{s}_m{m:02d}"].iloc[0] for m in BM]))
        rows.append(row)
    return pd.DataFrame(rows)


def figure(tab: pd.DataFrame, mix: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    years = catalogue.YEARS
    x = np.arange(len(years))
    for k, (reg, color) in enumerate(zip(("NH", "WE"), (plotting.SERIES[0], plotting.SERIES[1]))):
        t = tab[tab["region"] == reg].set_index("year").loc[years]
        axes[0].bar(x + (k - 0.5) * 0.38, 100 * t["dropped_share"], 0.36, color=color, label={"NH": "North Hesse", "WE": "Weser-Ems"}[reg], zorder=3)
    axes[0].axvspan(-0.6, 2.5, color=plotting.GRID, alpha=0.6, zorder=0)
    axes[0].text(1, axes[0].get_ylim()[1] * 0.93, "paper's temporal\ntest years", ha="center", va="top", fontsize=7, color=plotting.INK_SECONDARY)
    axes[0].set_xticks(x, years, rotation=45)
    axes[0].set_ylabel("cropland points removed by dropna() (%)")
    axes[0].set_title("(a) at least one empty bimonthly median")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.8))
    bottom = np.zeros(len(years))
    avg = mix.groupby("year")[[f"mean_obs_{s}" for s in lib.SENSOR_ORDER]].mean().loc[years]
    for s, color in zip(lib.SENSOR_ORDER, (plotting.SERIES[3], plotting.SERIES[4], plotting.SERIES[2])):
        axes[1].bar(x, avg[f"mean_obs_{s}"], 0.7, bottom=bottom, color=color, label={"L5": "Landsat 5", "L7": "Landsat 7", "L8": "Landsat 8"}[s], zorder=3)
        bottom += avg[f"mean_obs_{s}"].to_numpy()
    axes[1].axvspan(-0.6, 2.5, color=plotting.GRID, alpha=0.6, zorder=0)
    axes[1].set_xticks(x, years, rotation=45)
    axes[1].set_ylabel("valid observations per bimonth\n(mean of regions and bimonths)")
    axes[1].set_title("(b) by sensor")
    axes[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return plotting.save(fig, out)


def main() -> None:
    with RunLog("04_summarise", {}, paper=PAPER) as log:
        d = pd.read_csv(RES / "obs_counts_points.csv.gz")
        tab = year_table(d)
        mix = sensor_mix(tab)
        contrast = lib.period_contrast(tab, catalogue.TRAIN_YEARS, catalogue.TEST_YEARS)
        # Spatial scenario: same years, the other region (mean over training years, 2013-2018).
        sp = tab[tab["year"].isin(catalogue.TRAIN_YEARS)].groupby("region")["dropped_share"].mean()
        contrast["dropped_same_years_other_region"] = contrast["region"].map({"NH": sp["WE"], "WE": sp["NH"]})
        contrast["paper_nh_rf_mf1_loss"] = np.where(contrast["region"] == "NH", catalogue.PAPER_NH_RF_LOSS, np.nan)
        inputs = [str(RES / "obs_counts_points.csv.gz")]
        write_csv(tab.round(4), RES / "year_summary.csv", "04_summarise", inputs)
        write_csv(mix.round(4), RES / "sensor_mix.csv", "04_summarise", inputs)
        write_csv(contrast.round(4), RES / "period_contrast.csv", "04_summarise", inputs)
        log.add("figure", str(figure(tab, mix, FIG / "wijesingha_dropped_share.png")))
        pd.set_option("display.width", 250)
        show = ["region", "year", "dropped_share"] + [f"empty_share_m{m:02d}" for m in BM] + [f"mean_obs_m{m:02d}" for m in BM]
        print(tab[show].round(3).to_string(index=False))
        print(mix.round(2).to_string(index=False))
        print(contrast.round(3).to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
