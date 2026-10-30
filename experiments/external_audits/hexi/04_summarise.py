#!/usr/bin/env python
"""Gap structure of the weekly composites per domain, state and year under four validity definitions
(no mask, QA60 = the paper's recipe, Cloud Score+ clear, QA60 and Cloud Score+), residual cloud in
QA60-passed observations, monthly QA60 flag rates, and how many samples a cloud-affected rule would drop."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.hexi import catalogue as C, hexilib as hl  # noqa: E402

RES = paths.results_dir(C.PAPER)
ROOT = paths.paper_root(C.PAPER)
SCENARIOS = {"no_mask": "n", "qa60": "q", "cs50": "cs50", "qa60_cs50": "q_cs50"}
SEASON = C.SEASON_WEEKS


def load_year(year: int, pts: pd.DataFrame) -> dict[str, np.ndarray]:
    """band -> (n_points, 52) count matrices in the row order of points.csv."""
    files = sorted((ROOT / "chunks").glob(f"counts_{year}_c*_w*.csv.gz"))
    if not files:
        raise RuntimeError(f"no chunks for {year}; run 02_weekly_counts.py")
    blocks = {}
    for f in files:
        blocks.setdefault(f.name.split("_w")[-1], []).append(pd.read_csv(f).set_index("pid"))
    df = pd.concat([pd.concat(v) for v in blocks.values()], axis=1).reindex(pts["pid"])
    return {b: np.column_stack([df[f"w{k:02d}_{b}"].to_numpy() for k in range(C.N_WEEKS)]) for b in ["n", "q", *C.CS_THRESHOLDS, *[f"q_{k}" for k in C.CS_THRESHOLDS]]}


def groups(pts: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {g: (pts["group"] == g).to_numpy() for g in C.STATES + [C.TARGET]}
    out["Source (pooled)"] = (pts["domain"] == "source").to_numpy()
    return out


def main(force: bool = False) -> None:
    out = RES / "weekly_gap_summary.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    pts = pd.read_csv(RES / "points.csv")
    gm = groups(pts)
    summ, resid, prof, month, rules = [], [], [], [], []
    a, b = SEASON
    with RunLog("04_summarise", {"force": force}, paper=C.PAPER) as log:
        for year in C.YEARS:
            m = load_year(year, pts)
            nan_rows = int(np.isnan(m["n"]).any(1).sum())
            log.count(f"points_without_data_{year}", nan_rows)
            ok = ~np.isnan(m["n"]).any(1)  # a point with no sampled row (outside every footprint) is dropped
            for g, sel in gm.items():
                s = sel & ok
                dom = "target" if g == C.TARGET else "source"
                for sc, band in SCENARIOS.items():
                    v = m[band][s] > 0
                    summ.append({"year": year, "group": g, "scenario": sc, **hl.summary(v)})
                    if g in ("Source (pooled)", C.TARGET):
                        for k in range(C.N_WEEKS):
                            prof.append({"year": year, "domain": dom, "scenario": sc, "week": k + 1, "valid_share": float(v[:, k].mean())})
                        emp = ~v[:, a : b + 1]
                        ns = emp.shape[1]
                        rules.append({"year": year, "domain": dom, "scenario": sc, "n_points": int(s.sum()), "drop_any_empty_season_week": float(emp.any(1).mean()), "drop_ge_25pct_season_weeks_empty": float((emp.mean(1) >= 0.25).mean()), "drop_ge_50pct_season_weeks_empty": float((emp.mean(1) >= 0.5).mean()), "drop_season_gap_ge_4": float((hl.gap_metrics(v, SEASON)["season_longest_gap"] >= 4).mean()), "season_weeks": ns})
                n = m["n"][s]
                row = {"year": year, "group": g, "images_per_week_mean": float(n.mean()), "images_per_year_mean": float(n.sum(1).mean()), "qa60_flagged_obs_share": float(1 - m["q"][s].sum() / n.sum())}
                for k in C.CS_THRESHOLDS:
                    r = hl.residual_cloud(m["q"][s], m[f"q_{k}"][s])
                    row.update({f"resid_{k}_{kk}": vv for kk, vv in r.items()})
                    row[f"cs_flagged_obs_share_{k}"] = float(1 - m[k][s].sum() / n.sum())
                resid.append(row)
                if g in ("Source (pooled)", C.TARGET):
                    for mo in range(1, 13):
                        ks = [k for k in range(C.N_WEEKS) if hl.week_bounds(year, k)[0].month == mo]
                        nn = m["n"][s][:, ks].sum()
                        month.append({"year": year, "domain": dom, "month": mo, "images_sampled": int(nn), "qa60_flagged_share": float(1 - m["q"][s][:, ks].sum() / nn) if nn else np.nan, "cs50_flagged_share": float(1 - m["cs50"][s][:, ks].sum() / nn) if nn else np.nan, "qa60_pass_cs50_flagged_share": float(1 - m["q_cs50"][s][:, ks].sum() / m["q"][s][:, ks].sum()) if m["q"][s][:, ks].sum() else np.nan})
        ins = [str(ROOT / "chunks")]
        write_csv(pd.DataFrame(summ), out, "04_summarise", ins)
        write_csv(pd.DataFrame(resid), RES / "residual_cloud.csv", "04_summarise", ins)
        write_csv(pd.DataFrame(prof), RES / "weekly_valid_share.csv", "04_summarise", ins)
        write_csv(pd.DataFrame(month), RES / "monthly_qa60.csv", "04_summarise", ins)
        write_csv(pd.DataFrame(rules), RES / "discard_rules.csv", "04_summarise", ins)
        s = pd.DataFrame(summ)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(s[s["group"].isin(["Source (pooled)", C.TARGET])][["year", "group", "scenario", "valid_weeks_mean", "interp_share", "edge_share", "longest_interp_mean", "season_valid_weeks_mean", "season_longest_gap_mean"]].round(2).to_string())
            print(pd.DataFrame(resid).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
