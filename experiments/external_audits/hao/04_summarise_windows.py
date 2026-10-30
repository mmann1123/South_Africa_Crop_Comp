#!/usr/bin/env python
"""Window-level gap structure per domain under the paper's recipe (empty 15-day windows, share
filled by the two-sided neighbour average, runs of empty windows), plus the residual cloud in
windows the Hengshui L1C mask passes. Reads the per-chunk counts from 03_ee_windows.py.

Rules. HLS stream (CONUS 2017, NE and AB 2018): fm = Fmask cirrus/cloud/adjacent/shadow clear
(primary), obs = any observation (no quality screen), fmsnow = fm and no snow/ice. L8 + S2 L1C
stream (HS 2018; NE, AB, CONUS as controls): paper = L8 QA_PIXEL clear or S2 L1C mask clear,
nomask = any observation, cs65 / cs80 = L8 clear or S2 Cloud Score+ cs_cdf >= 0.65 / 0.80."""
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
from experiments.external_audits.hao import catalogue as cat, haolib  # noqa: E402

PAPER = cat.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
T = cat.N_WINDOWS
STREAM_RULES = {"hls": ["fm", "obs", "fmsnow"], "l8s2": ["paper", "nomask", "cs65", "cs80"]}
PRIMARY = {"hls": "fm", "l8s2": "paper"}


def load(stream: str, domain: str) -> pd.DataFrame | None:
    files = sorted((ROOT / "chunks").glob(f"{stream}_{domain.replace('@', '-')}_*.csv.gz"))
    if not files:
        return None
    pts = pd.read_csv(RES / "points.csv")
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("pid")
    return df.merge(pts[["pid", "crop", "lon", "lat"]], on="pid", how="left").sort_values("pid").reset_index(drop=True)


def valid_matrix(df: pd.DataFrame, stream: str, rule: str) -> np.ndarray:
    """(n, 18) bool: a composite exists in the window under the rule."""
    if stream == "hls":
        return haolib.combine_valid(df, rule)
    c = lambda a: np.column_stack([df[f"w{k:02d}_{a}"].to_numpy() > 0 for k in range(T)])  # noqa: E731
    if rule == "paper":
        return c("l8_clear") | c("s2_qa60")
    if rule == "nomask":
        return c("l8_obs") | c("s2_obs")
    return c("l8_clear") | c("s2_" + rule)


def main(force: bool = False) -> None:
    out = RES / "window_shares.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    shares, summ, pref, bycrop, resid = [], [], [], [], []
    with RunLog("04_summarise_windows", {"force": force}, paper=PAPER) as log:
        for stream, domains in (("hls", ["CONUS", "NE", "AB", "CONUS@2018", "NE@2017", "AB@2017"]), ("l8s2", ["HS", "NE", "AB", "CONUS"])):
            for dom in domains:
                df = load(stream, dom)
                if df is None:
                    continue
                n_drop = 0
                if stream == "hls":  # points no HLS granule covers in Earth Engine all season: the paper only used tiles with data
                    has = valid_matrix(df, "hls", "obs").any(1)
                    n_drop, df = int((~has).sum()), df[has].reset_index(drop=True)
                log.count(f"points_{stream}_{dom}", len(df))
                log.count(f"dropped_no_hls_{dom}", n_drop)
                for rule in STREAM_RULES[stream]:
                    if dom == "CONUS" and stream == "l8s2" and rule in ("cs65", "cs80"):
                        continue  # Cloud Score+ starts in 2017-03-28
                    v = valid_matrix(df, stream, rule)
                    key = {"stream": stream, "domain": dom, "rule": rule}
                    s = haolib.window_shares(v)
                    s["doy_start"] = cat.DOY_START + cat.WINDOW_DAYS * (s["window"] - 1)
                    s["doy_end"] = s["doy_start"] + cat.WINDOW_DAYS
                    for k, val in key.items():
                        s[k] = val
                    shares.append(s)
                    n_empty = (~v).sum(1)
                    lr = np.array([haolib.longest_run(~r) for r in v])
                    cls = haolib.fill_classes(v)
                    m, lo, hi = haolib.bootstrap_ci(n_empty / T)
                    summ.append({**key, "n_points": len(df), "n_dropped_no_hls": n_drop, "empty_share": m, "empty_ci_lo": lo, "empty_ci_hi": hi,
                                 "empty_windows_mean": float(n_empty.mean()),
                                 "interior_filled_share": float(cls["interior"].mean()), "edge_share": float(cls["edge"].mean()),
                                 "strict_fillable_share": float(cls["strict"].mean()),
                                 "share_points_any_empty": float((n_empty > 0).mean()),
                                 "share_points_run_ge2": float((lr >= 2).mean()), "share_points_run_ge3": float((lr >= 3).mean()),
                                 "longest_run_mean": float(lr.mean()),
                                 "empty_share_early_w1_5": float((~v[:, :5]).mean()), "empty_share_mid_w6_11": float((~v[:, 5:11]).mean()),
                                 "empty_share_late_w12_18": float((~v[:, 11:]).mean())})
                    for k in range(1, T + 1):
                        pref.append({**key, **haolib.prefix_metrics(v, k)})
                    if stream == "hls" and dom == "CONUS" and rule == PRIMARY[stream]:
                        for reg, info in cat.REGIONS.items():  # CONUS points of the crops the target region grows
                            sub = v[df["crop"].isin(info["crops"]).to_numpy()]
                            for k in range(1, T + 1):
                                pref.append({"stream": stream, "domain": f"CONUS[{reg} crops]", "rule": rule, **haolib.prefix_metrics(sub, k)})
                            n_sub = (~sub).sum(1)
                            summ.append({"stream": stream, "domain": f"CONUS[{reg} crops]", "rule": rule, "n_points": len(sub), "empty_share": float(n_sub.mean() / T), "empty_windows_mean": float(n_sub.mean()),
                                         "empty_share_early_w1_5": float((~sub[:, :5]).mean()), "empty_share_mid_w6_11": float((~sub[:, 5:11]).mean()), "empty_share_late_w12_18": float((~sub[:, 11:]).mean())})
                    if rule == PRIMARY[stream]:
                        for crop, g in df.groupby("crop"):
                            sv = valid_matrix(g, stream, rule)
                            sc = haolib.window_shares(sv)
                            sc["stream"], sc["domain"], sc["rule"], sc["crop"] = stream, dom, rule, crop
                            bycrop.append(sc)
                if stream == "l8s2":
                    paper = valid_matrix(df, stream, "paper")
                    strict = valid_matrix(df, stream, "cs65")
                    nomask = valid_matrix(df, stream, "nomask") if dom != "CONUS" else paper
                    l8_only = np.column_stack([df[f"w{k:02d}_l8_clear"].to_numpy() > 0 for k in range(T)])
                    s2_obs = np.column_stack([df[f"w{k:02d}_s2_obs"].to_numpy() > 0 for k in range(T)])
                    resid.append(pd.DataFrame({
                        "domain": dom, "window": np.arange(1, T + 1), "n_points": len(df),
                        "paper_valid_share": paper.mean(0),
                        "strict_valid_share": strict.mean(0) if dom != "CONUS" else np.nan,
                        "only_cloud_contaminated_share": (paper & ~strict).mean(0) if dom != "CONUS" else np.nan,
                        "l8_clear_share": l8_only.mean(0), "s2_any_obs_share": s2_obs.mean(0),
                        "no_obs_at_all_share": (~nomask).mean(0),
                    }))
        sh = pd.concat(shares, ignore_index=True)
        write_csv(sh, out, "04_summarise_windows", [str(ROOT / "chunks")])
        write_csv(pd.DataFrame(summ), RES / "domain_summary.csv", "04_summarise_windows", [str(ROOT / "chunks")])
        write_csv(pd.DataFrame(pref), RES / "prefix_summary.csv", "04_summarise_windows", [str(ROOT / "chunks")])
        write_csv(pd.concat(bycrop, ignore_index=True), RES / "window_shares_by_crop.csv", "04_summarise_windows", [str(ROOT / "chunks")])
        write_csv(pd.concat(resid, ignore_index=True), RES / "hs_residual_cloud.csv", "04_summarise_windows", [str(ROOT / "chunks")])
        s = pd.DataFrame(summ)
        log.count("summary_rows", len(s))
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
