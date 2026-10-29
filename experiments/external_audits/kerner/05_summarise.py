#!/usr/bin/env python
"""Kerner audit summary: for each year x basis (fixed DOY / pixel growth stage) x
stage x class, the share of model inputs with no clear HLS observation within
+-8 days, the median clear count, and the exact-day state; plus 2019 minus
training differences and the figure."""
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
from experiments.external_audits.kerner import kernerlib as kl  # noqa: E402
from experiments.external_audits.kerner import kernersummary as ks  # noqa: E402

PAPER = "kerner"
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
YEAR_COLOR = {2017: plotting.SERIES[0], 2018: plotting.SERIES[1], 2019: plotting.SERIES[2]}


def build_inputs(sample: pd.DataFrame, pix: pd.DataFrame, acq: pd.DataFrame, log: RunLog):
    sample = sample.merge(pix, on="pid", how="inner", validate="1:1")
    # One index-to-calendar offset for all years: the one that puts the released
    # phenology DOYs on HLS acquisition days most often, pooled over years.
    offs = []
    for year, g in sample.groupby("year"):
        a = acq[acq["year"] == year]
        days = set(a.loc[a["spatial_coverage_pct"] > 0, "doy"])  # only acquisitions observed over part of the tile can match
        _, share = kl.best_offset(g[list(kl.GROWTH_STAGES)].to_numpy(), days, offsets=(-1, 0, 1))
        offs.append({"year": year, "match_share_offsetm1": share[-1], "match_share_offset0": share[0], "match_share_offset1": share[1], "n_acquisition_days": len(days), "n_pixels": len(g)})
    offs = pd.DataFrame(offs)
    off = int(max((-1, 0, 1), key=lambda o: offs[f"match_share_offset{'m1' if o == -1 else o}"].mean()))
    offs["offset_used_all_years"] = off
    parts = []
    for year, g in sample.groupby("year"):
        a = acq[acq["year"] == year].reset_index(drop=True)
        states = ks.decode_states(g["states"].tolist())
        assert states.shape[1] == len(a), f"{year}: {states.shape[1]} state digits vs {len(a)} acquisitions"
        parts.append(ks.input_table(g.reset_index(drop=True), states, a["doy"].to_numpy(), int(year), off))
    log.add("doy_offsets", offs.to_dict("records"))
    return pd.concat(parts, ignore_index=True), offs


def figure(summary: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharey="row")
    metrics = [("share_zero_clear", "no clear observation within +-8 days"), ("share_exact_cloudy", "flagged on the exact input day\n(pale: flagged or not observed)")]
    for r, (m, ylabel) in enumerate(metrics):
        for c, (basis, stages, title) in enumerate((("fixed", kl.FIXED_STAGES, "fixed DOYs"), ("growth", kl.GROWTH_STAGES, "pixel growth-stage DOYs"))):
            ax = axes[r, c]
            for j, year in enumerate((2017, 2018, 2019)):
                ys, lo, hi = [], [], []
                for s in stages:
                    row = summary[(summary["year"] == year) & (summary["basis"] == basis) & (summary["stage"] == s) & (summary["class"] == "all")].iloc[0]
                    ys.append(row[m]); lo.append(row[m] - row[m + "_lo"]); hi.append(row[m + "_hi"] - row[m])
                x = np.arange(3) + (j - 1) * 0.25
                if m == "share_exact_cloudy":  # pale bar: not clear on the exact day (flagged or no HLS observation); solid: flagged
                    nc = [summary[(summary["year"] == year) & (summary["basis"] == basis) & (summary["stage"] == s_) & (summary["class"] == "all")].iloc[0]["share_exact_not_clear"] for s_ in stages]
                    ax.bar(x, nc, width=0.24, color=YEAR_COLOR[year], alpha=0.3)
                ax.bar(x, ys, width=0.24, color=YEAR_COLOR[year], label=str(year) + (" (test)" if year == 2019 else " (train)"))
                ax.errorbar(x, ys, yerr=[lo, hi], fmt="none", ecolor=plotting.INK_SECONDARY, elinewidth=0.8, capsize=1.5)
            ax.set_xticks(range(3))
            ax.set_xticklabels(stages)
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"share of inputs\n{ylabel}", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return plotting.save(fig, out)


def main(force: bool = False, n_boot: int = 1000) -> None:
    out_sum = RES / "kerner_summary.csv"
    if out_sum.exists() and not force:
        print("summary already present; nothing to do (use --force to redo)")
        return
    with RunLog("05_summarise", {"force": force, "n_boot": n_boot}, paper=PAPER) as log:
        sample = pd.read_csv(RES / "kerner_sample.csv")
        pix = pd.read_csv(RES / "kerner_pixel_states.csv", dtype={"states": str})
        acq = pd.read_csv(RES / "kerner_hls_acquisitions.csv")
        log.count("pixels_sampled", int(len(sample)))
        log.count("pixels_with_states", int(len(pix)))
        inputs, offs = build_inputs(sample, pix, acq, log)
        summary = ks.aggregate(inputs, n_boot=n_boot)
        diff = ks.test_minus_train(inputs, n_boot=n_boot)
        srcs = [str(RES / n) for n in ("kerner_sample.csv", "kerner_pixel_states.csv", "kerner_hls_acquisitions.csv")]
        write_csv(offs, RES / "kerner_doy_offset.csv", "05_summarise", srcs)
        write_csv(summary, out_sum, "05_summarise", srcs)
        write_csv(diff, RES / "kerner_test_minus_train.csv", "05_summarise", srcs)
        log.add("figure", str(figure(summary, FIG / "kerner_fill_shares.png")))
        pd.set_option("display.width", 250)
        print(offs.round(3).to_string(index=False))
        cols = ["year", "basis", "stage", "class", "n_pixels", "share_zero_clear", "median_n_clear", "share_exact_cloudy", "share_exact_no_acq"]
        print(summary[cols].round(3).to_string(index=False))
        print(diff.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    main(**vars(ap.parse_args()))
