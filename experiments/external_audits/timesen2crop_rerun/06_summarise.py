#!/usr/bin/env python
"""Apply the pass marks fixed in the design
(docs/superpowers/specs/2026-11-05-timesen2crop-rerun-design.md) and write the tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timesen2crop_rerun import tslib as tl  # noqa: E402

PAPER = "timesen2crop_rerun"
RES = paths.results_dir(PAPER)
PAPER_OA = {"lstm": 83.44, "tempcnn": 81.71}
NONZERO = [("Z", "M"), ("Z", "L"), ("M", "M"), ("L", "L"), ("Zm", "Zm")]


def t_ci(d: np.ndarray) -> tuple[float, float, float]:
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return m, m - h, m + h


def main() -> None:
    with RunLog("06_summarise", {}, paper=PAPER) as log:
        df = pd.read_csv(RES / "train_eval.csv")
        for c in ("oa", "f1m", "oa_reweighted"):
            df[c] *= 100
        keys = ["model", "train_fill", "test_fill", "tile"]
        s = df.groupby(keys)[["oa", "f1m", "oa_reweighted"]].agg(["mean", "std"]).round(2)
        s.columns = ["_".join(c) for c in s.columns]
        write_csv(s.reset_index(), RES / "condition_summary.csv", "06_summarise", ["results/train_eval.csv"])

        def vals(model, tr, te, tile, metric="f1m"):
            q = df[(df.model == model) & (df.train_fill == tr) & (df.test_fill == te) & (df.tile == tile)].sort_values("seed")
            return q.set_index("seed")[metric]

        gate, gaps, lifts, assay, verdict = [], [], [], [], []
        for model in PAPER_OA:
            oa = vals(model, "Z", "Z", tl.TEST_TILE, "oa")
            ok = abs(oa.mean() - PAPER_OA[model]) <= 2
            gate.append({"model": model, "oa_33UVP_2018": oa.mean(), "sd": oa.std(), "paper_oa": PAPER_OA[model], "diff": oa.mean() - PAPER_OA[model], "within_2_points": ok})
            for metric in ("f1m", "oa", "oa_reweighted"):
                d = vals(model, "Z", "Z", tl.TEST_TILE, metric) - vals(model, "Z", "Z", tl.YEAR_TILE, metric)
                m, lo, hi = t_ci(d.values)
                gaps.append({"model": model, "metric": metric, "gap_2018_minus_2019": m, "ci_low": lo, "ci_high": hi})
            base = {t: vals(model, "Z", "Z", t) for t in (tl.VAL_TILE, tl.TEST_TILE, tl.YEAR_TILE)}
            for tr, te in NONZERO:
                row = {"model": model, "train_fill": tr, "test_fill": te}
                for t in (tl.VAL_TILE, tl.TEST_TILE, tl.YEAR_TILE):
                    m, lo, hi = t_ci((vals(model, tr, te, t) - base[t]).values)
                    row |= {f"lift_{t}": m, f"lo_{t}": lo, f"hi_{t}": hi, f"f1m_{t}": vals(model, tr, te, t).mean()}
                lifts.append(row)
            d = vals(model, "M", "M", tl.YEAR_TILE) - vals(model, "M", "Z", tl.YEAR_TILE)
            m, lo, hi = t_ci(d.values)
            assay.append({"model": model, "tile": tl.YEAR_TILE, "fall_MtoZ_vs_MtoM": m, "ci_low": lo, "ci_high": hi, "passes_10pt": m >= 10})
            d = vals(model, "M", "M", tl.TEST_TILE) - vals(model, "M", "Z", tl.TEST_TILE)
            m, lo, hi = t_ci(d.values)
            assay.append({"model": model, "tile": tl.TEST_TILE, "fall_MtoZ_vs_MtoM": m, "ci_low": lo, "ci_high": hi, "passes_10pt": m >= 10})
        lifts = pd.DataFrame(lifts)
        for model, g in lifts.groupby("model"):
            b = g.loc[g[f"lift_{tl.VAL_TILE}"].idxmax()]
            verdict.append({"model": model, "best_on_33TWN": f"{b['train_fill']} to {b['test_fill']}", "lift_2019": b[f"lift_{tl.YEAR_TILE}"], "lo_2019": b[f"lo_{tl.YEAR_TILE}"], "hi_2019": b[f"hi_{tl.YEAR_TILE}"],
                            "lift_2018": b[f"lift_{tl.TEST_TILE}"], "lo_2018": b[f"lo_{tl.TEST_TILE}"], "hi_2018": b[f"hi_{tl.TEST_TILE}"],
                            "supported_model": bool(b[f"lift_{tl.YEAR_TILE}"] >= 5 and b[f"lo_{tl.YEAR_TILE}"] > 0 and b[f"lift_{tl.YEAR_TILE}"] > b[f"lift_{tl.TEST_TILE}"]),
                            "weak_model": bool(b[f"lift_{tl.YEAR_TILE}"] >= 1 and b[f"lo_{tl.YEAR_TILE}"] > 0)})
        verdict = pd.DataFrame(verdict)
        # the design: a model that misses the reproduction gate is reported but not interpreted
        passed = {g["model"] for g in gate if g["within_2_points"]}
        verdict["passes_gate"] = verdict["model"].isin(passed)
        v = verdict[verdict["passes_gate"]]
        both = verdict["passes_gate"].all()
        overall = "Supported" if both and v["supported_model"].all() else ("Weak" if v["weak_model"].any() else "Null")
        verdict["overall"] = overall
        pass_assay = all(a["passes_10pt"] for a in assay if a["tile"] == tl.YEAR_TILE)
        verdict["assay_passes_both_models"] = pass_assay
        for n, t in (("reproduction", gate), ("year_gap", gaps), ("assay_check", assay)):
            write_csv(pd.DataFrame(t), RES / f"{n}.csv", "06_summarise", ["results/train_eval.csv"])
        write_csv(lifts, RES / "fill_lifts.csv", "06_summarise", ["results/train_eval.csv"])
        write_csv(verdict, RES / "verdict.csv", "06_summarise", ["results/fill_lifts.csv", "results/assay_check.csv"])
        pd.set_option("display.width", 250)
        for n, t in (("gate", pd.DataFrame(gate)), ("gaps", pd.DataFrame(gaps)), ("assay", pd.DataFrame(assay)), ("verdict", verdict)):
            print(n); print(t.round(2).to_string())
        print(lifts[["model", "train_fill", "test_fill", f"lift_{tl.VAL_TILE}", f"lift_{tl.TEST_TILE}", f"lo_{tl.TEST_TILE}", f"hi_{tl.TEST_TILE}", f"lift_{tl.YEAR_TILE}", f"lo_{tl.YEAR_TILE}", f"hi_{tl.YEAR_TILE}"]].round(2).to_string())
        log.count("rows", len(df))


if __name__ == "__main__":
    main()
