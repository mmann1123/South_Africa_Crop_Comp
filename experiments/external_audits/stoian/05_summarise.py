#!/usr/bin/env python
"""Join the clear-count, L1C, digitised kappa and Table 1 results; correlations, partial
correlations, the no-mountain and leave-one-out variants, and the figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.stoian import catalogue, corr  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
TILES = catalogue.TILES
MEASURES = {
    "scl_median_year": "SCL median, year",
    "scl_median_year_frpart": "SCL median, year, T30TYN French part",
    "scl_median_marOct": "SCL median, Mar to Oct",
    "scl_p10_year": "SCL 10th percentile, year",
    "neg_gap_marOct": "Longest Mar to Oct gap (sign flipped)",
    "l1c_year": "L1C effective clear, year",
    "l1c_marOct": "L1C effective clear, Mar to Oct",
}
TARGETS = {"fg_drop": "FG-UNET drop", "rf_drop": "RF drop", "fg_10p1": "FG-UNET 10+1 kappa", "rf_10p1": "RF 10+1 kappa"}


def build_table() -> pd.DataFrame:
    scl = pd.read_csv(RES / "scl_tile_summary.csv").set_index("tile")
    fr = pd.read_csv(RES / "scl_t30tyn_french_part.csv").set_index("tile")
    l1 = pd.read_csv(RES / "l1c_tile_summary.csv").set_index("tile")
    k = pd.read_csv(RES / "fig19_kappa_digitised_wide.csv").set_index("tile")
    t1 = pd.read_csv(RES / "table1_class_areas.csv")
    d = pd.DataFrame(index=TILES)
    d["region"] = pd.Series(catalogue.REGION)
    d["scl_median_year"] = scl["median_clear_year"]
    d["scl_median_marOct"] = scl["median_clear_marOct"]
    d["scl_p10_year"] = scl["p10_clear_year"]
    d["scl_gap_marOct_days"] = scl["median_longest_gap_marOct_days"]
    d["neg_gap_marOct"] = -d["scl_gap_marOct_days"]
    d["scl_median_year_frpart"] = d["scl_median_year"]
    d.loc["T30TYN", "scl_median_year_frpart"] = fr.loc["T30TYN", "median_clear_year"]
    d["l1c_year"] = l1["l1c_eff_clear_year"]
    d["l1c_marOct"] = l1["l1c_eff_clear_marOct"]
    d["fg_11"], d["fg_10p1"], d["rf_11"], d["rf_10p1"] = k["fg_unet_11"], k["fg_unet_10p1"], k["rf_11"], k["rf_10p1"]
    d["fg_drop"] = d["fg_11"] - d["fg_10p1"]
    d["rf_drop"] = d["rf_11"] - d["rf_10p1"]
    d["mountain"] = [t in catalogue.MOUNTAIN for t in TILES]
    d["composition_jsd"] = corr.composition_index(t1, TILES)
    d["mountain_class_share"] = corr.share_of_classes(t1, TILES, ["BRO", "NGL", "GPS"])
    return d


def correlation_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, mlabel in MEASURES.items():
        for tgt, tlabel in TARGETS.items():
            rows.append({"measure": mlabel, "measure_key": m, "target": tlabel, "target_key": tgt, "subset": "all 11 tiles", **corr.rank_and_linear(d[m], d[tgt])})
    nm = d[~d["mountain"]]
    for m in ("scl_median_year", "scl_median_marOct"):
        for tgt in ("fg_drop", "rf_drop"):
            rows.append({"measure": MEASURES[m], "measure_key": m, "target": TARGETS[tgt], "target_key": tgt, "subset": "9 tiles, T30TYN and T31TGK removed", **corr.rank_and_linear(nm[m], nm[tgt])})
    return pd.DataFrame(rows)


def extras(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lo, hi = corr.leave_one_out_range(d["scl_median_marOct"], d["fg_drop"])
    rows.append({"quantity": "leave-one-out Spearman rho range, SCL median Mar to Oct vs FG drop", "value": f"{lo:.2f} to {hi:.2f}"})
    lo, hi = corr.leave_one_out_range(d["scl_median_year"], d["fg_drop"])
    rows.append({"quantity": "leave-one-out Spearman rho range, SCL median year vs FG drop", "value": f"{lo:.2f} to {hi:.2f}"})
    for tgt in ("fg_drop", "rf_drop"):
        rows.append({"quantity": f"Spearman rho, composition JSD vs {tgt}", "value": f"{corr.rank_and_linear(d['composition_jsd'], d[tgt])['spearman_rho']:.2f}"})
        rows.append({"quantity": f"partial Spearman rho, SCL median Mar to Oct vs {tgt} | composition JSD", "value": f"{corr.partial_spearman(d['scl_median_marOct'], d[tgt], d['composition_jsd']):.2f}"})
        rows.append({"quantity": f"partial Spearman rho, SCL median year vs {tgt} | composition JSD", "value": f"{corr.partial_spearman(d['scl_median_year'], d[tgt], d['composition_jsd']):.2f}"})
        rows.append({"quantity": f"partial Spearman rho, composition JSD vs {tgt} | SCL median Mar to Oct", "value": f"{corr.partial_spearman(d['composition_jsd'], d[tgt], d['scl_median_marOct']):.2f}"})
        rows.append({"quantity": f"partial Spearman rho, composition JSD vs {tgt} | SCL median year", "value": f"{corr.partial_spearman(d['composition_jsd'], d[tgt], d['scl_median_year']):.2f}"})
        rows.append({"quantity": f"OLS R2, {tgt} ~ SCL median Mar to Oct + composition JSD", "value": f"{corr.ols_r2(d[tgt], d['scl_median_marOct'], d['composition_jsd']):.2f}"})
    rows.append({"quantity": "Spearman rho, SCL median Mar to Oct vs composition JSD", "value": f"{corr.rank_and_linear(d['scl_median_marOct'], d['composition_jsd'])['spearman_rho']:.2f}"})
    for tgt in ("fg_10p1", "rf_10p1"):
        rows.append({"quantity": f"Pearson r, mountain-class share (BRO+NGL+GPS) vs {tgt} (held-out kappa)", "value": f"{corr.rank_and_linear(d['mountain_class_share'], d[tgt])['pearson_r']:.2f}"})
    return pd.DataFrame(rows)


def figure(d: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True)
    for ax, tgt, title in zip(axes, ("fg_drop", "rf_drop"), ("(a) FG-UNET", "(b) Random forest")):
        for mtn, color, label in ((False, plotting.SERIES[0], "other tiles"), (True, plotting.SERIES[1], "mountain tiles (authors)")):
            s = d[d["mountain"] == mtn]
            ax.scatter(s["scl_median_marOct"], s[tgt], color=color, s=28, label=label, zorder=3)
            for t, r in s.iterrows():
                ax.annotate(t[1:], (r["scl_median_marOct"], r[tgt]), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=plotting.INK_SECONDARY)
        rr = corr.rank_and_linear(d["scl_median_marOct"], d[tgt])
        ax.set_title(f"{title}\nSpearman rho = {rr['spearman_rho']:.2f} (n = 11)", fontsize=9)
        ax.set_xlabel("median clear dates per pixel,\nMar to Oct 2016 (Sen2Cor SCL)")
        ax.axhline(0, color=plotting.GRID, lw=0.8)
    axes[0].set_ylabel("kappa drop when tile held out")
    axes[1].legend(loc="center right")
    axes[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    fig.tight_layout()
    return plotting.save(fig, out)


def main() -> None:
    with RunLog("05_summarise", {}, paper=PAPER) as log:
        d = build_table()
        ct = correlation_table(d)
        ex = extras(d)
        inputs = [str(RES / f) for f in ("scl_tile_summary.csv", "scl_t30tyn_french_part.csv", "l1c_tile_summary.csv", "fig19_kappa_digitised_wide.csv", "table1_class_areas.csv")]
        write_csv(d.reset_index(names="tile").round(4), RES / "stoian_tile_table.csv", "05_summarise", inputs)
        write_csv(ct.round(4), RES / "stoian_correlations.csv", "05_summarise", inputs)
        write_csv(ex, RES / "stoian_extras.csv", "05_summarise", inputs)
        log.add("figure", str(figure(d, FIG / "stoian_clear_vs_kappa_drop.png")))
        pd.set_option("display.width", 250)
        print(d.drop(columns="region").round(3).to_string())
        print(ct[["measure", "target", "subset", "n", "spearman_rho", "spearman_p", "pearson_r"]].round(2).to_string(index=False))
        print(ex.to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
