#!/usr/bin/env python
"""Join the paper's per-direction OA (Tables 4 and 5) to the measured per-site interpolation share and
valid-observation counts; write the direction table and the figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.maleki import catalogue as C, interplib as il  # noqa: E402

RES = paths.results_dir(C.PAPER)
FIG = paths.figures_dir(C.PAPER)


def paper_table() -> pd.DataFrame:
    rows = []
    for (src, tgt), algs in C.TABLE4.items():
        d_src = il.drops(algs, C.TABLE5[src])
        d_tgt = il.drops(algs, C.TABLE5[tgt])
        oa = [x for a in algs.values() for x in a]
        rows.append({"train": src, "test": tgt, "n_configs": len(oa), "oa_mean": round(float(np.mean(oa)), 1), "oa_best": max(oa),
                     "drop_vs_source_baseline_mean": round(float(np.mean(d_src)), 1), "drop_vs_source_baseline_min": min(d_src), "drop_vs_source_baseline_max": max(d_src),
                     "drop_vs_target_baseline_mean": round(float(np.mean(d_tgt)), 1), "drop_vs_target_baseline_min": min(d_tgt), "drop_vs_target_baseline_max": max(d_tgt)})
    return pd.DataFrame(rows)


def join(paper: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    s = summ[(summ["mask"] == "paper_masks") & (summ["stratum"] == "all")].set_index("site")
    out = []
    for _, r in paper.iterrows():
        a, b = s.loc[r["train"]], s.loc[r["test"]]
        row = r.to_dict()
        for k in ("n_valid_median", "share_not_observed_mean", "longest_unobserved_steps_median", "longest_gap_days_median"):
            row[f"source_{k}"], row[f"target_{k}"] = a[k], b[k]
        row["source_more_interpolated"] = bool(a["share_not_observed_mean"] > b["share_not_observed_mean"])
        out.append(row)
    return pd.DataFrame(out)


def figure(summ: pd.DataFrame, monthly: pd.DataFrame, px: pd.DataFrame, paper: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plotting.theme()
    col = {"Camargue": plotting.SERIES[1], "Champagne": plotting.SERIES[0]}
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for site, g in px.groupby("site"):
        ax[0].hist(g["n_valid"], bins=np.arange(0, g["n_valid"].max() + 3, 2), alpha=0.6, color=col[site], label=f"{site} (median {int(g['n_valid'].median())})", density=True)
    ax[0].set_xlabel("valid acquisitions per pixel, 2021")
    ax[0].set_ylabel("share of pixels")
    ax[0].set_title("(a) after the paper's masks")
    ax[0].legend()
    for site, g in monthly.groupby("site"):
        ax[1].plot(g["month"], g["valid_mean"], marker="o", color=col[site], label=site)
    ax[1].set_xticks(range(1, 13))
    ax[1].set_xlabel("month")
    ax[1].set_ylabel("mean valid acquisitions per pixel")
    ax[1].set_title("(b) by month")
    x = np.arange(2)
    w = 0.38
    sa = summ[(summ["mask"] == "paper_masks") & (summ["stratum"] == "all")].set_index("site")
    vals = [100 * sa.loc[s, "share_not_observed_mean"] for s in C.SITES]
    ax[2].bar(x[:1], vals[:1], w * 2, color=col["Camargue"])
    ax[2].bar(x[1:], vals[1:], w * 2, color=col["Champagne"])
    for xi, v, (lo, hi) in zip(x, vals, [(100 * sa.loc[s, "share_not_observed_mean_lo"], 100 * sa.loc[s, "share_not_observed_mean_hi"]) for s in C.SITES]):
        ax[2].errorbar(xi, v, yerr=[[v - lo], [hi - v]], color=plotting.INK, capsize=3, lw=1)
        ax[2].text(xi, hi + 1.5, f"{v:.0f}%", ha="center")
    ax[2].set_xticks(x, [f"{s}\nas source" for s in C.SITES])
    ax[2].set_ylabel("grid steps not within +-3 d of a valid date (%)")
    ax[2].set_title("(c) share of the 60 steps interpolated")
    ax2 = ax[2].twinx()
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    drop = paper.set_index("train")["drop_vs_source_baseline_mean"]
    ax2.plot(x, [drop["Camargue"], drop["Champagne"]], "D", color=plotting.INK, label="paper's mean OA drop as source")
    ax2.set_ylabel("paper: mean OA drop vs source baseline (points)")
    ax2.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout()
    plotting.save(fig, FIG / "maleki_valid_obs_interpolation.png")


def main(force: bool = False) -> None:
    out = RES / "maleki_direction_table.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("04_summarise", {"force": force}, paper=C.PAPER) as log:
        paper = paper_table()
        write_csv(paper, RES / "paper_transfer_oa.csv", "04_summarise", ["Maleki and Rahdari 2026, Tables 4 and 5 (transcribed in catalogue.py)"])
        summ = pd.read_csv(RES / "interp_summary.csv")
        table = join(paper, summ)
        write_csv(table, out, "04_summarise", [str(RES / "interp_summary.csv"), str(RES / "paper_transfer_oa.csv")])
        figure(summ, pd.read_csv(RES / "monthly_valid.csv"), pd.read_csv(RES / "pixel_metrics.csv"), paper)
        log.count("directions", len(table))
        print(table.T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
