#!/usr/bin/env python
"""Figure: QA60 flag rate by month and year, weekly share of valid composites in 2022 under the paper's QA60 mask
and under Cloud Score+, and per-group empty and interpolated shares of the weekly series."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.hexi import catalogue as C  # noqa: E402

RES = paths.results_dir(C.PAPER)
FIG = paths.figures_dir(C.PAPER)


def main(force: bool = False) -> None:
    out = FIG / "hexi_gap_structure.png"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    plotting.theme()
    mon = pd.read_csv(RES / "monthly_qa60.csv")
    prof = pd.read_csv(RES / "weekly_valid_share.csv")
    summ = pd.read_csv(RES / "weekly_gap_summary.csv")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    col = {"source": plotting.SERIES[1], "target": plotting.SERIES[0]}
    label = {"source": "Midwest source", "target": "Hexi target"}

    a = ax[0, 0]
    for dom in ("source", "target"):
        for year, ls in zip(C.YEARS, [":", "-", "--"]):
            d = mon[(mon["domain"] == dom) & (mon["year"] == year)]
            a.plot(d["month"], d["qa60_flagged_share"], ls, color=col[dom], label=f"{label[dom]} {year}")
    a.set(xlabel="month", ylabel="share of observations flagged by QA60", title="QA60 cloud flags, L1C", xticks=range(1, 13))
    a.legend(ncol=2)

    a = ax[0, 1]
    for dom in ("source", "target"):
        for sc, ls in (("qa60", "-"), ("qa60_cs50", "--")):
            d = prof[(prof["domain"] == dom) & (prof["year"] == C.PAPER_YEAR) & (prof["scenario"] == sc)]
            a.plot(d["week"], d["valid_share"], ls, color=col[dom], label=f"{label[dom]}, {'QA60 (paper)' if sc == 'qa60' else 'QA60 and Cloud Score+'}")
    a.set(xlabel="week of 2022", ylabel="share of points with a valid weekly composite", title="Non-empty weekly composites, 2022", ylim=(0, 1.02))
    a.legend()

    for a, sc, ttl in ((ax[1, 0], "qa60", "2022 weekly series under the paper's QA60 mask"), (ax[1, 1], "qa60_cs50", "2022 weekly series after also removing Cloud Score+ cloud")):
        d = summ[(summ["year"] == C.PAPER_YEAR) & (summ["scenario"] == sc)].set_index("group")
        order = C.STATES + ["Source (pooled)", C.TARGET]
        d = d.loc[order]
        x = np.arange(len(order))
        a.bar(x, d["interp_share"], color=plotting.SERIES[2], label="interpolated (interior empty weeks)")
        a.bar(x, d["edge_share"], bottom=d["interp_share"], color=plotting.SERIES[3], label="edge (before first / after last valid)")
        a.set(xticks=x, ylabel="share of 52 weeks", title=ttl, ylim=(0, 1))
        a.set_xticklabels([o.replace(" (pooled)", "\n(pooled)").replace("North ", "N. ").replace("South ", "S. ") for o in order], fontsize=7)
        a.legend(loc="upper left")
    fig.tight_layout()
    plotting.save(fig, out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
