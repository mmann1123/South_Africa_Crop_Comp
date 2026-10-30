#!/usr/bin/env python
"""Relate the per-length transfer deficit of Hao et al. 2020 (local-training OA minus transfer OA,
digitised from Figs. 4, 6, 8) to how much emptier the target series are than the CONUS 2017
training pool, for the first k windows (k = 1..18, series DOY 60 to 60 + 15 k). Descriptive:
Spearman correlations over the 18 lengths per region and pooled (n = 54); lengths of one region
share windows, so the n are not independent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.hao import catalogue as cat, haolib  # noqa: E402

PAPER = cat.PAPER
RES = paths.results_dir(PAPER)
TARGET = {"HS": ("l8s2", "paper"), "AB": ("hls", "fm"), "NE": ("hls", "fm")}
METRICS = ["empty_share", "share_points_any_empty", "share_points_run_ge2", "edge_share"]


def main(force: bool = False) -> None:
    out = RES / "gap_vs_oa.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    oa = pd.read_csv(RES / "fig_oa_digitised.csv")
    pref = pd.read_csv(RES / "prefix_summary.csv")
    rows = []
    with RunLog("05_relate_oa", {"force": force}, paper=PAPER) as log:
        for reg, (stream, rule) in TARGET.items():
            t = pref[(pref["domain"] == reg) & (pref["stream"] == stream) & (pref["rule"] == rule)].set_index("k")
            for ref_name in ("CONUS", f"CONUS[{reg} crops]"):
                ref = pref[(pref["domain"] == ref_name) & (pref["stream"] == "hls") & (pref["rule"] == "fm")].set_index("k")
                for k in range(1, cat.N_WINDOWS + 1):
                    end = cat.DOY_START + cat.WINDOW_DAYS * k
                    tl = oa[(oa["region"] == reg) & (oa["series"] == "TL") & (oa["doy_end"] == end)]["oa"].iloc[0]
                    lo = oa[(oa["region"] == reg) & (oa["series"] == "LO") & (oa["doy_end"] == end)]["oa"].iloc[0]
                    row = {"region": reg, "reference": ref_name, "k": k, "doy_end": end, "oa_tl": tl, "oa_lo": lo, "oa_gap_lo_minus_tl": round(lo - tl, 2)}
                    for m in METRICS:
                        row[f"target_{m}"], row[f"ref_{m}"] = t.loc[k, m], ref.loc[k, m]
                        row[f"diff_{m}"] = t.loc[k, m] - ref.loc[k, m]
                    rows.append(row)
        df = pd.DataFrame(rows)
        write_csv(df, out, "05_relate_oa", [str(RES / "fig_oa_digitised.csv"), str(RES / "prefix_summary.csv")])
        cor = []
        for ref_kind, g_ref in (("all CONUS", df[df["reference"] == "CONUS"]), ("crop-matched CONUS", df[df["reference"] != "CONUS"])):
            for scope, g in [(r, g_ref[g_ref["region"] == r]) for r in TARGET] + [("pooled", g_ref)]:
                for m in METRICS:
                    for ycol in ("oa_gap_lo_minus_tl", "oa_tl"):
                        cor.append({"reference": ref_kind, "scope": scope, "metric": f"diff_{m}", "outcome": ycol, **haolib.spearman(g[f"diff_{m}"], g[ycol])})
        c = pd.DataFrame(cor)
        write_csv(c, RES / "gap_vs_oa_correlations.csv", "05_relate_oa", [str(out)])
        log.count("rows", len(df))
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(c[c["outcome"] == "oa_gap_lo_minus_tl"].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
