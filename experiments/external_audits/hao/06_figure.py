#!/usr/bin/env python
"""Figure: empty 15-day windows by window and domain, Hengshui under its own and stricter masks,
the per-length OA deficit of transfer learning, and the target-minus-CONUS difference in empty
share of the first k windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.hao import catalogue as cat  # noqa: E402

PAPER = cat.PAPER
RES = paths.results_dir(PAPER)
COL = {"CONUS": plotting.SERIES[0], "NE": plotting.SERIES[1], "AB": plotting.SERIES[2], "HS": plotting.SERIES[3]}
LABEL = {"CONUS": "CONUS 2017 (HLS)", "NE": "Nebraska 2018 (HLS)", "AB": "Alberta 2018 (HLS)", "HS": "Hengshui 2018 (L8 + S2 L1C)"}


def main(force: bool = False) -> None:
    out = paths.figures_dir(PAPER) / "hao_empty_windows_vs_oa_gap.png"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    plotting.theme()
    sh = pd.read_csv(RES / "window_shares.csv")
    gap = pd.read_csv(RES / "gap_vs_oa.csv")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    x = lambda d: d["doy_end"]  # noqa: E731
    a = ax[0, 0]
    for dom, stream, rule in (("CONUS", "hls", "fm"), ("NE", "hls", "fm"), ("AB", "hls", "fm"), ("HS", "l8s2", "paper")):
        d = sh[(sh["domain"] == dom) & (sh["stream"] == stream) & (sh["rule"] == rule)]
        a.plot(x(d), 100 * d["empty_share"], color=COL[dom], marker="o", label=LABEL[dom])
    a.set_title("Windows with no usable observation")
    a.set_ylabel("share of points (%)")
    a.set_xlabel("window end (DOY)")
    a.legend()
    b = ax[0, 1]
    for dom, ls in (("HS", "-"), ("NE", "--"), ("AB", ":")):
        for rule, c in (("paper", plotting.SERIES[3]), ("cs65", plotting.SERIES[6])):
            d = sh[(sh["domain"] == dom) & (sh["stream"] == "l8s2") & (sh["rule"] == rule)]
            if len(d):
                b.plot(x(d), 100 * d["empty_share"], color=c, ls=ls, label=f"{dom}, {'L1C mask' if rule == 'paper' else 'Cloud Score+ 0.65'}")
    b.set_title("L8 + S2 L1C stream: paper mask vs Cloud Score+")
    b.set_xlabel("window end (DOY)")
    b.legend(fontsize=7, ncol=2)
    c = ax[1, 0]
    g = gap[gap["reference"] == "CONUS"]
    for reg in ("HS", "AB", "NE"):
        d = g[g["region"] == reg]
        c.plot(x(d), d["oa_gap_lo_minus_tl"], color=COL[reg], marker="o", label=reg)
    c.set_title("OA deficit of transfer (LO minus TL, digitised)")
    c.set_ylabel("points")
    c.set_xlabel("series end (DOY)")
    c.legend()
    d_ax = ax[1, 1]
    for reg in ("HS", "AB", "NE"):
        d = g[g["region"] == reg]
        d_ax.plot(x(d), 100 * d["diff_empty_share"], color=COL[reg], marker="o", label=reg)
    d_ax.axhline(0, color=plotting.INK_SECONDARY, lw=0.8)
    d_ax.set_title("Empty share of windows 1..k, target minus CONUS")
    d_ax.set_ylabel("percentage points")
    d_ax.set_xlabel("series end (DOY)")
    d_ax.legend()
    fig.tight_layout()
    plotting.save(fig, out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
