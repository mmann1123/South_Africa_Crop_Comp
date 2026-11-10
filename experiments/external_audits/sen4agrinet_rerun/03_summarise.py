#!/usr/bin/env python
"""Summarise the Sen4AgriNet PAD re-run from the per-patch confusion matrices
written by 02_run_conditions.py. Every F1 is computed from confusion matrices
summed over the patches of a subset, so it is the same statistic the authors
report; 95 percent intervals resample patches (500 bootstrap draws, seed 42)
and recompute the contrast on each draw.

Tables (results/):
  reproduction.csv     condition a, all test patches, per scenario and model:
                       authors' weighted F1 against the README value, corrected
                       weighted F1 and macro F1
  bound_gap.csv        Scenario 1 (random split) minus Scenario 2 or 3, same model
  month_deletion.csv   per month m (1 to 12, all of which the released models consume): F1 on patches that have m, as published and
                       with m deleted and refilled; cost = published minus deleted
  pattern_imposition.csv  all patches, and patches with all 12 months: published, source-pattern,
                       target-pattern and matched-random deletion (mean over 5
                       draws), with source-minus-target and pattern-minus-random
  predicted_share.csv  sum over months of (test minus train empty-month rate) x
                       month cost, against the bound gap
  cloud.csv            Hollstein cloud mask before the median, minus published
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet_rerun import evalcore as ec  # noqa: E402

PAPER = "sen4agrinet_rerun"
RES = paths.results_dir(PAPER)
S4A = paths.AUDIT_DIR / "sen4agrinet" / "results"
K = 12
README_F1W = {(1, "unet"): 82.61, (1, "convlstm"): 85.18, (1, "convstar"): 80.38,
              (2, "unet"): 57.85, (2, "convlstm"): 56.56, (2, "convstar"): 52.15,
              (3, "unet"): 43.54, (3, "convlstm"): 40.47, (3, "convstar"): 34.45}
METRICS = {"f1w_authors": ("cm_authors", ec.authors_weighted_f1), "f1w_corrected": ("cm_corrected", ec.corrected_weighted_f1),
           "f1m_corrected": ("cm_corrected", ec.corrected_macro_f1)}
N_BOOT = 500


def load() -> pd.DataFrame:
    files = sorted((RES / "raw").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in ("cm_authors", "cm_corrected"):
        df[c] = df[c].map(lambda v: np.asarray(v, dtype=np.int64))
    return df.drop_duplicates(["scenario", "arch", "stem", "cond", "draw"])


def stack(df: pd.DataFrame, col: str, stems: list[str]) -> np.ndarray:
    """(n_stems, K, K) confusion matrices in stem order."""
    m = df.set_index("stem")[col]
    return np.stack([m[s].reshape(K, K) for s in stems])


def f1(cms: np.ndarray, fn) -> float:
    return 100 * fn(cms.sum(axis=0))


def contrast(a: np.ndarray, b: np.ndarray, fn, rng) -> tuple[float, float, float, float, float]:
    """F1(a), F1(b), F1(b) - F1(a), and a patch-bootstrap CI of the difference."""
    fa, fb = f1(a, fn), f1(b, fn)
    n = len(a)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        d.append(f1(b[i], fn) - f1(a[i], fn))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return fa, fb, fb - fa, lo, hi


def draw_mean(df: pd.DataFrame, prefix: str, col: str, stems: list[str]) -> np.ndarray:
    """Sum of the 5 draws' matrices divided by 5 keeps counts comparable to one pass."""
    mats = [stack(df[df["cond"] == f"{prefix}_{k}"], col, stems) for k in range(5)]
    return np.mean(mats, axis=0)


def empty_rates(scenario: int, split: str) -> pd.Series:
    mm = pd.read_csv(S4A / "month_missing_by_split.csv")
    comp = pd.read_csv(S4A / "split_composition.csv").groupby(["scenario", "split", "country", "year"])["n_patches"].sum().reset_index()
    m = mm[(mm["scenario"] == scenario) & (mm["split"] == split)].merge(comp, on=["scenario", "split", "country", "year"])
    return m.groupby("month").apply(lambda g: np.average(g["share_missing"], weights=g["n_patches"]))


def main() -> None:
    with RunLog("03_summarise", {}, paper=PAPER) as log:
        df = load()
        rng = np.random.default_rng(42)
        rep, gap, mon, pat, pred, cld = [], [], [], [], [], []
        a = df[df["cond"] == "a"]
        for (s, arch), g in a.groupby(["scenario", "arch"]):
            stems = sorted(g["stem"])
            row = {"scenario": s, "arch": arch, "patches": len(stems), "readme_f1w": README_F1W[(s, arch)]}
            for name, (col, fn) in METRICS.items():
                row[name] = f1(stack(g, col, stems), fn)
            row["diff_authors_minus_readme"] = row["f1w_authors"] - row["readme_f1w"]
            rep.append(row)
        rep = pd.DataFrame(rep)
        write_csv(rep, RES / "reproduction.csv", "03_summarise", ["results/raw/*.parquet"])
        r1 = rep[rep["scenario"] == 1].set_index("arch")
        for _, r in rep[rep["scenario"] != 1].iterrows():
            gap.append({"scenario": r["scenario"], "arch": r["arch"], **{f"gap_{k}": r1.loc[r["arch"], k] - r[k] for k in METRICS}})
        gap = pd.DataFrame(gap)
        write_csv(gap, RES / "bound_gap.csv", "03_summarise", ["results/reproduction.csv"])
        full = df[df["scenario"] != 1]
        for (s, arch), g in full.groupby(["scenario", "arch"]):
            done = set(g.loc[g["cond"] == "cloud", "stem"])  # patches whose full condition set has been written
            g = g[g["stem"].isin(done)]
            base = g[g["cond"] == "a"]
            for mth in range(12):
                dl = g[g["cond"] == f"del_m{mth}"]
                stems = sorted(dl["stem"])
                if not stems:
                    continue
                for name, (col, fn) in METRICS.items():
                    fa, fb, d, lo, hi = contrast(stack(base, col, stems), stack(dl, col, stems), fn, rng)
                    mon.append({"scenario": s, "arch": arch, "month": mth + 1, "patches_with_month": len(stems), "metric": name,
                                "f1_published": fa, "f1_deleted": fb, "cost": -d, "cost_ci_low": -hi, "cost_ci_high": -lo})
            subsets = {"all_patches": sorted(base["stem"]), "complete_year": sorted(base.loc[base["window_complete"], "stem"])}
            for (subset, wc), (name, (col, fn)) in ((x, y) for x in subsets.items() for y in METRICS.items()):
                if len(wc) < 20:
                    continue
                A = stack(base, col, wc)
                S, T, R = (draw_mean(g, p, col, wc) for p in ("src", "tgt", "rnd"))
                _, fs, dsa, _, _ = contrast(A, S, fn, rng)
                _, ft, _, _, _ = contrast(A, T, fn, rng)
                _, fr, _, _, _ = contrast(A, R, fn, rng)
                _, _, d_st, lo_st, hi_st = contrast(T, S, fn, rng)
                _, _, d_sr, lo_sr, hi_sr = contrast(R, S, fn, rng)
                _, _, d_tr, lo_tr, hi_tr = contrast(R, T, fn, rng)
                nd = g[g["stem"].isin(wc)].groupby("cond")["n_deleted"].mean()
                pat.append({"scenario": s, "arch": arch, "metric": name, "subset": subset, "patches": len(wc),
                            "f1_published": f1(A, fn), "f1_source_pattern": fs, "f1_target_pattern": ft, "f1_random": fr,
                            "source_minus_target": d_st, "st_ci_low": lo_st, "st_ci_high": hi_st,
                            "source_minus_random": d_sr, "sr_ci_low": lo_sr, "sr_ci_high": hi_sr,
                            "target_minus_random": d_tr, "tr_ci_low": lo_tr, "tr_ci_high": hi_tr,
                            "months_deleted_source": nd.filter(like="src").mean(), "months_deleted_target": nd.filter(like="tgt").mean(),
                            "months_deleted_random": nd.filter(like="rnd").mean()})
            cl = g[g["cond"] == "cloud"]
            stems = sorted(cl["stem"])
            for name, (col, fn) in METRICS.items():
                fa, fb, d, lo, hi = contrast(stack(base, col, stems), stack(cl, col, stems), fn, rng)
                cld.append({"scenario": s, "arch": arch, "metric": name, "patches": len(stems), "f1_published": fa, "f1_cloud_masked": fb,
                            "change": d, "ci_low": lo, "ci_high": hi})
        mon = pd.DataFrame(mon)
        pat = pd.DataFrame(pat)
        cld = pd.DataFrame(cld)
        for s in sorted(full["scenario"].unique()):
            et, es = empty_rates(s, "test"), empty_rates(s, "train")
            for (arch, name), g in mon[(mon["scenario"] == s) & (mon["metric"] != "f1w_authors")].groupby(["arch", "metric"]):
                # the authors' metric is NaN on any subset where a class gets no prediction, so it is left out here
                cost = g.set_index("month")["cost"]
                diff = (et - es).reindex(cost.index)
                p = float((diff * cost).sum())
                gg = gap[(gap["scenario"] == s) & (gap["arch"] == arch)][f"gap_{name}"].iloc[0]
                pred.append({"scenario": s, "arch": arch, "metric": name, "predicted_loss_from_fill_difference": p, "bound_gap": gg,
                             "share_of_gap": p / gg if gg else np.nan,
                             **{f"empty_diff_m{m}": diff.get(m, np.nan) for m in range(1, 13)}})
        write_csv(mon, RES / "month_deletion.csv", "03_summarise", ["results/raw/*.parquet"])
        write_csv(pat, RES / "pattern_imposition.csv", "03_summarise", ["results/raw/*.parquet", "../sen4agrinet/results/patch_time_stacks.csv.gz"])
        write_csv(pd.DataFrame(pred), RES / "predicted_share.csv", "03_summarise", ["results/month_deletion.csv", "../sen4agrinet/results/month_missing_by_split.csv"])
        write_csv(cld, RES / "cloud.csv", "03_summarise", ["results/raw/*.parquet"])
        log.count("rows", len(df))
        print(rep.round(2).to_string())
        print(gap.round(2).to_string())
        print(pat[pat["metric"] == "f1w_corrected"].round(2).to_string())
        print(pd.DataFrame(pred).round(3).to_string())
        print(cld.round(2).to_string())


if __name__ == "__main__":
    main()
