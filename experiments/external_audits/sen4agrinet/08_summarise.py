#!/usr/bin/env python
"""Summary tables and the figure: per-month missing share by country-year, fill
shares per scenario and window (train vs test), observations per patch-year."""
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
from experiments.external_audits.sen4agrinet import fillstats  # noqa: E402

PAPER = "sen4agrinet"
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOMAINS = [("Catalonia", 2019), ("Catalonia", 2020), ("France", 2019)]


def population(comp: pd.DataFrame) -> pd.DataFrame:
    c = comp[comp["scenario"].isin([2, 3]) & comp["split"].isin(["train", "test"])]
    return c.groupby(["country", "year", "tile"])["n_patches"].sum().reset_index()


def figure(cym: pd.DataFrame, fs: pd.DataFrame, pf: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    colors = {("Catalonia", 2019): plotting.SERIES[0], ("Catalonia", 2020): plotting.SERIES[2], ("France", 2019): plotting.SERIES[1]}
    ax.axvspan(3.5, 9.5, color=plotting.GRID, alpha=0.55, lw=0, zorder=0)
    ax.hlines(-0.06, 3.6, 9.4, color=plotting.INK_SECONDARY, lw=2)
    ax.hlines(-0.12, 4.6, 10.4, color=plotting.INK_SECONDARY, lw=2, ls="--")
    ax.text(9.6, -0.06, "PAD window (Apr-Sep)", va="center", fontsize=7.5, color=plotting.INK_SECONDARY)
    ax.text(10.6, -0.12, "OAD as coded (May-Nov)", va="center", fontsize=7.5, color=plotting.INK_SECONDARY)
    for dom in DOMAINS:
        g = cym[(cym["country"] == dom[0]) & (cym["year"] == dom[1])].sort_values("month")
        ax.plot(range(12), g["share_missing"].values, marker="o", color=colors[dom], label=f"{dom[0]} {dom[1]}")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS)
    ax.set_ylim(-0.17, 1.2)
    ax.set_xlim(-0.4, 12.6)
    ax.set_ylabel("share of patches with no acquisition")
    ax.set_title("(a) Months with no Sentinel-2 acquisition, per patch (these months are filled by the loader)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax2 = fig.add_subplot(gs[1, 0])
    labels, xs = [], []
    pos = 0
    for win, wl in (("PAD_Apr-Sep", "PAD"), ("OAD_May-Nov", "OAD")):
        for scen in (2, 3):
            tr = fs[(fs["scenario"] == scen) & (fs["split"] == "train") & (fs["window"] == win)].iloc[0]
            te = fs[(fs["scenario"] == scen) & (fs["split"] == "test") & (fs["window"] == win)].iloc[0]
            for j, (r, c) in enumerate(((tr, plotting.SERIES[0]), (te, plotting.SERIES[1]))):
                x = pos + j * 0.8
                ax2.bar(x, r["share_filled"] - r["share_extrapolated"], 0.7, color=c)
                ax2.bar(x, r["share_extrapolated"], 0.7, bottom=r["share_filled"] - r["share_extrapolated"], color=c, hatch="////", edgecolor=plotting.SURFACE, lw=0)
                ax2.errorbar(x, r["share_filled"], yerr=[[r["share_filled"] - r["share_filled_ci_low"]], [r["share_filled_ci_high"] - r["share_filled"]]], color=plotting.INK, lw=0.8, capsize=2)
            labels.append(f"{wl}\nS{scen}")
            xs.append(pos + 0.4)
            pos += 2.2
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("share of patch-months filled")
    ax2.set_title("(b) Filled share, train vs test")
    from matplotlib.patches import Patch

    ax2.legend(handles=[Patch(color=plotting.SERIES[0], label="train"), Patch(color=plotting.SERIES[1], label="test"), Patch(facecolor="white", edgecolor=plotting.INK_SECONDARY, hatch="////", label="extrapolated")], loc="upper left", fontsize=7)

    ax3 = fig.add_subplot(gs[1, 1])
    u = pf.drop_duplicates("stem")
    data = [u[(u["country"] == c) & (u["year"] == y)]["n_obs"].values for c, y in DOMAINS]
    parts = ax3.violinplot(data, positions=range(3), showmedians=True, widths=0.8)
    for body, dom in zip(parts["bodies"], DOMAINS):
        body.set_facecolor(colors[dom])
        body.set_edgecolor(colors[dom])
        body.set_alpha(0.55)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(plotting.INK_SECONDARY)
        parts[key].set_linewidth(0.9)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels([f"{c[:3]}\n{y}" if c == "Catalonia" else f"FR\n{y}" for c, y in DOMAINS])
    ax3.set_ylabel("acquisitions per patch-year")
    ax3.set_title("(c) Observations per patch-year")
    return plotting.save(fig, out)


def main() -> None:
    with RunLog("08_summarise", {}, paper=PAPER) as log:
        comp = pd.read_csv(RES / "split_composition.csv")
        fs = pd.read_csv(RES / "fill_by_split.csv")
        pf = pd.read_csv(RES / "patch_fill.csv")
        mm_ty = pd.read_csv(RES / "month_missing_by_tile_year.csv")
        cym = fillstats.combine_tile_years(mm_ty, population(comp))
        write_csv(cym, RES / "month_missing_by_country_year.csv", "08_summarise", ["month_missing_by_tile_year.csv", "split_composition.csv"])
        cc_path = RES / "oad_csv_crosscheck.csv"
        # headline comparison table: this rebuild against the earlier scratch-script values
        earlier = pd.DataFrame(
            [
                (2, "train", "PAD_Apr-Sep", 0.098), (2, "test", "PAD_Apr-Sep", 0.149), (3, "train", "PAD_Apr-Sep", 0.158), (3, "test", "PAD_Apr-Sep", 0.150),
                (2, "train", "OAD_May-Nov", 0.014), (2, "test", "OAD_May-Nov", 0.198), (3, "train", "OAD_May-Nov", 0.207), (3, "test", "OAD_May-Nov", 0.000),
            ],
            columns=["scenario", "split", "window", "earlier_share_filled"],
        )
        cmp_ = fs.merge(earlier, on=["scenario", "split", "window"])
        cmp_["difference"] = cmp_["share_filled"] - cmp_["earlier_share_filled"]
        write_csv(cmp_[["scenario", "split", "window", "share_filled", "share_filled_ci_low", "share_filled_ci_high", "earlier_share_filled", "difference"]], RES / "rebuild_vs_earlier.csv", "08_summarise", ["fill_by_split.csv"])
        stac_path = RES / "stac_granules_by_month.csv"
        if stac_path.exists():
            st = pd.read_csv(stac_path).merge(mm_ty, on=["country", "year", "tile", "month"], how="left")
            st["patch_month_mostly_missing"] = st["share_missing"] >= 0.5
            st["stac_no_clear_granule"] = st["n_cloud_lt10"] == 0
            write_csv(st, RES / "stac_vs_patch_presence.csv", "08_summarise", ["stac_granules_by_month.csv", "month_missing_by_tile_year.csv"])
            agree = (st["patch_month_mostly_missing"] == st["stac_no_clear_granule"]).mean()
            log.add("stac_agreement_share_of_tile_year_months", float(agree))
            log.add("stac_tile_year_months", int(len(st)))
        figure(cym, fs, pf, FIG / "sen4agrinet_fill_by_month.png")
        log.count("figure", 1)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
