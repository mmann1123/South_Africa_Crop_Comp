#!/usr/bin/env python
"""DENETHOR summary tables and the two-panel figure."""
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
from experiments.external_audits.denethor import dnsummary  # noqa: E402

PAPER = "denethor"
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
ORDER = ["s2-utm-33N-18E-242N-2018", "s2-utm-33N-17E-243N-2018", "s2-utm-33N-18E-242N-2019", "s2-utm-33N-17E-243N-2019"]
SHORT = {"s2-utm-33N-18E-242N-2018": "18E-242N 2018 (train)", "s2-utm-33N-17E-243N-2018": "17E-243N 2018", "s2-utm-33N-18E-242N-2019": "18E-242N 2019", "s2-utm-33N-17E-243N-2019": "17E-243N 2019 (test)"}
MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]


def figure(s: pd.DataFrame, monthly: pd.DataFrame | None, out: Path) -> Path:
    plotting.theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    data = [s.loc[s["tile_year"] == ty, "n_clear_s2cp40_aprOct"].values for ty in ORDER]
    colors = [plotting.SERIES[0], plotting.SERIES[2], plotting.SERIES[3], plotting.SERIES[1]]
    parts = ax1.violinplot(data, positions=range(4), showmedians=True, widths=0.8)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.55)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(plotting.INK_SECONDARY)
        parts[key].set_linewidth(0.9)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([SHORT[t].replace(" (", "\n(") for t in ORDER], fontsize=7.5)
    ax1.set_ylabel("clear Sentinel-2 acquisitions per field, Apr to Oct")
    ax1.set_title("(a) Clear observations, s2cloudless < 40 %")
    if monthly is not None:
        for (tile, year), g in monthly.groupby(["tile", "year"]):
            m = g.groupby("month")["share_days_mostly_synthetic"].mean()
            m = m.reindex(range(4, 11))
            label = f"{tile} {year} ({'train' if year == 2018 else 'test'})"
            color = plotting.SERIES[0] if year == 2018 else plotting.SERIES[1]
            ax2.plot(MONTHS, m.values, marker="o", color=color, label=label)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("share of days mostly synthetic (PF-QA > 50 %)")
        ax2.set_title("(b) Planet Fusion inputs synthesised")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    return plotting.save(fig, out)


def main() -> None:
    src = RES / "s2_field_summary.csv"
    with RunLog("04_summarise", {}, paper=PAPER) as log:
        s = pd.read_csv(src)
        tys = dnsummary.tile_year_summary(s)
        t22 = dnsummary.two_by_two(s)
        marg = dnsummary.margins(t22)
        marg_lit = dnsummary.margins(dnsummary.two_by_two(s), col="median_n_clear_literal40")
        monthly_path = RES / "pfqa_field_month.csv"
        monthly = pd.read_csv(monthly_path) if monthly_path.exists() else None
        if monthly is not None:
            season = monthly[(monthly["month"] >= 4) & (monthly["month"] <= 10)]
            pf = season.groupby(["tile", "year"]).agg(mean_share_days_mostly_synthetic_season=("share_days_mostly_synthetic", "mean"), mean_percent_synthetic=("mean_percent_synthetic", "mean"), mean_days_to_real=("mean_days_to_real", "mean"), n_fields=("field_id", "nunique")).reset_index()
            pf["by_month"] = [";".join(f"{v:.3f}" for v in season[(season["tile"] == t) & (season["year"] == y)].groupby("month")["share_days_mostly_synthetic"].mean().reindex(range(4, 11)).values) for t, y in zip(pf["tile"], pf["year"])]
            tys = tys.merge(pf, on=["tile", "year"], how="left", suffixes=("", "_pf"))
            log.add("pfqa_season", pf.to_dict("records"))
        write_csv(tys, RES / "denethor_summary.csv", "04_summarise", [str(src), str(monthly_path)])
        t22["margin"] = ""
        write_csv(t22, RES / "denethor_2x2.csv", "04_summarise", [str(src)])
        log.add("margins_s2cp40", marg)
        log.add("margins_literal40", marg_lit)
        log.add("figure", str(figure(s, monthly, FIG / "denethor_clear_and_synthetic.png")))
        print(tys[["tile_year", "n_fields", "median_n_clear_s2cp40", "median_n_clear_literal40", "median_n_clear_s2cp40_aprOct", "share_fields_under_20_clear_aprOct_s2cp40", "median_longest_gap_days"]].round(3).to_string(index=False))
        print("margins (s2cp40):", marg)
        print("margins (literal40):", marg_lit)
        if monthly is not None:
            print(pf.round(3).to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
