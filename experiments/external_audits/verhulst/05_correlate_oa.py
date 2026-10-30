#!/usr/bin/env python
"""Join the 20 digitised cross-year OA losses (Figure 7) to the per-year valid-dekad measures and
correlate them (Spearman; n = 20 per algorithm is suggestive only). Also the validation-year level
(n = 5) using the bar losses printed in the paper's Figure 7b-d, and the figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.verhulst import catalogue, verhulstlib as vl  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
YEARS = catalogue.YEARS
ALGOS = ["RF", "SVM", "MLP"]
# Mean loss per validation year printed in the text for the extreme years (RF, SVM, MLP): checks the digitising.
TEXT_LOSSES = {("RF", 2022): 2.81, ("RF", 2021): 8.12, ("SVM", 2019): 3.19, ("SVM", 2018): 14.92, ("MLP", 2022): 2.30, ("MLP", 2020): 9.03}


def scenario_table(oa: pd.DataFrame, summ: pd.DataFrame, variant: str) -> pd.DataFrame:
    s = summ[summ["variant"] == variant].set_index("year")
    rows = []
    for algo in ALGOS:
        a = oa[oa["algo"] == algo]
        d = {(int(r.training_year), int(r.validation_year)): r.oa for r in a.itertuples()}  # key (train, test)
        for sc in vl.scenarios(d, YEARS):
            tr, te = sc["train_year"], sc["test_year"]
            rows.append(
                {
                    "algo": algo, "variant": variant, **sc,
                    "valid_train": s.loc[tr, "valid_dekads_mean"], "valid_test": s.loc[te, "valid_dekads_mean"],
                    "valid_jj_train": s.loc[tr, "valid_dekads_jan_jun_mean"], "valid_jj_test": s.loc[te, "valid_dekads_jan_jun_mean"],
                    "gap_jj_train": s.loc[tr, "longest_gap_jan_jun_mean"], "gap_jj_test": s.loc[te, "longest_gap_jan_jun_mean"],
                    "filled_train": s.loc[tr, "filled_share"], "filled_test": s.loc[te, "filled_share"],
                }
            )
    t = pd.DataFrame(rows)
    t["diff_valid"] = t["valid_train"] - t["valid_test"]  # positive: the test year is gappier
    t["abs_diff_valid"] = t["diff_valid"].abs()
    t["diff_valid_jj"] = t["valid_jj_train"] - t["valid_jj_test"]
    t["diff_gap_jj"] = t["gap_jj_test"] - t["gap_jj_train"]
    t["diff_filled"] = t["filled_test"] - t["filled_train"]
    return t


def correlations(t: pd.DataFrame, variant: str) -> pd.DataFrame:
    mean3 = t.groupby(["train_year", "test_year"], as_index=False).agg({c: "mean" for c in ["oa_loss", "oa_same_year", "diff_valid", "abs_diff_valid", "diff_valid_jj", "diff_gap_jj", "diff_filled", "valid_test", "valid_jj_test"]})
    mean3.insert(0, "algo", "mean3")
    out = []
    for algo, g in list(t.groupby("algo")) + [("mean3", mean3)]:
        for pred in ["diff_valid", "abs_diff_valid", "diff_valid_jj", "diff_gap_jj", "diff_filled", "valid_test", "valid_jj_test", "oa_same_year"]:
            out.append({"variant": variant, "algo": algo, "target": "oa_loss", "predictor": pred, **vl.spearman(g[pred], g["oa_loss"])})
    return pd.DataFrame(out)


def year_level(bars: pd.DataFrame, summ: pd.DataFrame, variant: str) -> pd.DataFrame:
    s = summ[summ["variant"] == variant].set_index("year")
    b = bars.copy()
    b["valid_test"] = b["validation_year"].map(s["valid_dekads_mean"])
    b["valid_jj_test"] = b["validation_year"].map(s["valid_dekads_jan_jun_mean"])
    out = []
    for algo, g in list(b.groupby("algo")) + [("mean3", b.groupby("validation_year", as_index=False).mean(numeric_only=True))]:
        for pred in ["valid_test", "valid_jj_test"]:
            out.append({"variant": variant, "algo": algo, "level": "validation_year", "predictor": pred, **vl.spearman(g[pred], g["mean_loss_bars"])})
    return pd.DataFrame(out)


def figure(summ: pd.DataFrame, dek: pd.DataFrame, t: pd.DataFrame, path: Path) -> None:
    plotting.theme()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    w = 0.27
    for i, (v, lab) in enumerate([("none", "no dilation"), ("d1", "dilated, about 1 px"), ("d3", "dilated, about 3 px")]):
        s = summ[summ["variant"] == v].set_index("year").loc[YEARS]
        ax[0].bar(np.arange(5) + (i - 1) * w, s["valid_dekads_mean"], w, yerr=s["valid_dekads_se"], color=plotting.SERIES[i], label=lab, error_kw={"lw": 0.8})
    ax[0].set_xticks(range(5), YEARS)
    ax[0].set_ylabel("Valid dekads per forest pixel (of 36)")
    ax[0].set_title("Valid dekads per year")
    ax[0].legend(loc="lower right")
    d = dek[dek["variant"] == catalogue.PRIMARY]
    for i, y in enumerate(YEARS):
        g = d[d["year"] == y].drop_duplicates("month").sort_values("month")
        ax[1].plot(g["month"], g["month_valid_share"], marker="o", color=plotting.SERIES[i], label=str(y))
    ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax[1].set_ylabel("Share of pixel-dekads valid")
    ax[1].set_title("Monthly valid share (dilated, about 3 px)")
    ax[1].legend(ncol=2)
    m = t.groupby(["train_year", "test_year"], as_index=False).agg(oa_loss=("oa_loss", "mean"), diff_valid=("diff_valid", "mean"))
    for i, y in enumerate(YEARS):
        g = m[m["test_year"] == y]
        ax[2].scatter(g["diff_valid"], g["oa_loss"], color=plotting.SERIES[i], label=f"test {y}", s=22)
    rho = vl.spearman(m["diff_valid"], m["oa_loss"])
    ax[2].axvline(0, color=plotting.GRID, lw=0.8)
    ax[2].set_xlabel("Valid dekads, train year minus test year")
    ax[2].set_ylabel("OA loss, mean of RF, SVM, MLP (points)")
    ax[2].set_title(f"Spearman rho = {rho['spearman_rho']:.2f} (n = {rho['n']}, p = {rho['spearman_p']:.2f})")
    ax[2].legend(ncol=2)
    plotting.save(fig, path)
    plt.close(fig)


def main(force: bool = False) -> None:
    out = RES / "oa_loss_scenarios.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("05_correlate_oa", {"force": force, "primary": catalogue.PRIMARY}, paper=PAPER) as log:
        oa = pd.read_csv(RES / "fig7_oa_digitised.csv")
        bars = pd.read_csv(RES / "fig7_bar_losses_digitised.csv")
        summ = pd.read_csv(RES / "valid_dekads_summary.csv")
        dek = pd.read_csv(RES / "dekad_valid_share.csv")
        tabs, cors, yl = [], [], []
        for v in catalogue.VARIANTS:
            t = scenario_table(oa, summ, v)
            tabs.append(t)
            cors.append(correlations(t, v))
            yl.append(year_level(bars, summ, v))
        t_all = pd.concat(tabs)
        write_csv(t_all, out, "05_correlate_oa", [str(RES / "fig7_oa_digitised.csv"), str(RES / "valid_dekads_summary.csv")])
        c = pd.concat(cors + yl)
        write_csv(c, RES / "oa_loss_correlations.csv", "05_correlate_oa", [str(out)])
        # digitising check: mean cross-year loss per validation year against the paper's printed values and bars
        chk = t_all[t_all["variant"] == catalogue.PRIMARY].groupby(["algo", "test_year"], as_index=False)["oa_loss"].mean()
        chk = chk.merge(bars, left_on=["algo", "test_year"], right_on=["algo", "validation_year"]).drop(columns="validation_year")
        chk["text_value"] = [TEXT_LOSSES.get((a, y), np.nan) for a, y in zip(chk["algo"], chk["test_year"])]
        chk["minus_bars"] = chk["oa_loss"] - chk["mean_loss_bars"]
        write_csv(chk, RES / "fig7_digitising_check.csv", "05_correlate_oa", [str(out)])
        log.add("digitising_abs_error_vs_bars_max", float(chk["minus_bars"].abs().max()))
        log.add("digitising_abs_error_vs_bars_mean", float(chk["minus_bars"].abs().mean()))
        figure(summ, dek, t_all[t_all["variant"] == catalogue.PRIMARY], paths.figures_dir(PAPER) / "verhulst_valid_dekads_vs_oa_loss")
        with pd.option_context("display.width", 250, "display.max_rows", 200):
            print(chk.round(2).to_string(index=False))
            print(c[(c["variant"] == catalogue.PRIMARY)].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
