#!/usr/bin/env python
"""Figure for the Sen4AgriNet re-run (corrected macro F1 throughout):
(a) cost of deleting and refilling each month, per model and scenario;
(b) per model and scenario, the in-domain-minus-transfer bound gap against the
loss predicted from the test-train fill difference, the target-minus-source
pattern cost and the gain from cloud masking."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402

PAPER = "sen4agrinet_rerun"
RES = paths.results_dir(PAPER)
M = "f1m_corrected"
ARCHS = ["unet", "convlstm", "convstar"]
COL = {"unet": "#4a7ab5", "convlstm": "#d08c3c", "convstar": "#5a9e6f"}


def main() -> None:
    mon = pd.read_csv(RES / "month_deletion.csv").query("metric == @M")
    gap = pd.read_csv(RES / "bound_gap.csv")
    pred = pd.read_csv(RES / "predicted_share.csv").query("metric == @M")
    pat = pd.read_csv(RES / "pattern_imposition.csv").query("metric == @M and subset == 'all_patches'")
    cld = pd.read_csv(RES / "cloud.csv").query("metric == @M")
    scen = sorted(mon["scenario"].unique())
    fig, ax = plt.subplots(1, len(scen) + 1, figsize=(5 * (len(scen) + 1), 4))
    for i, s in enumerate(scen):
        for a in ARCHS:
            g = mon[(mon["scenario"] == s) & (mon["arch"] == a)].sort_values("month")
            ax[i].errorbar(g["month"] + (ARCHS.index(a) - 1) * 0.2, g["cost"],
                           yerr=[g["cost"] - g["cost_ci_low"], g["cost_ci_high"] - g["cost"]], fmt="o", ms=4, lw=1, color=COL[a], label=a)
        ax[i].axhline(0, color="grey", lw=0.8)
        ax[i].set_xticks(range(1, 13))
        ax[i].set_xlabel("Month deleted and refilled")
        ax[i].set_ylabel("Macro-F1 cost (points)")
        ax[i].set_title(f"({'ab'[i]}) Scenario {s}: month deletion")
    ax[0].legend(frameon=False)
    rows = []
    for s in scen:
        for a in ARCHS:
            rows.append({"label": f"S{s} {a}",
                         "gap": gap.query("scenario == @s and arch == @a")[f"gap_{M}"].iloc[0],
                         "pred": pred.query("scenario == @s and arch == @a")["predicted_loss_from_fill_difference"].iloc[0],
                         "pattern": pat.query("scenario == @s and arch == @a")["source_minus_target"].iloc[0],
                         "cloud": cld.query("scenario == @s and arch == @a")["change"].iloc[0]})
    r = pd.DataFrame(rows)
    y = np.arange(len(r))
    h = 0.2
    axb = ax[-1]
    axb.barh(y - 1.5 * h, r["gap"], h, color="#bbbbbb", label="bound gap (S1 minus transfer)")
    axb.barh(y - 0.5 * h, r["pred"], h, color="#c0504d", label="predicted from fill difference")
    axb.barh(y + 0.5 * h, r["pattern"], h, color="#8064a2", label="extra cost of target-domain gaps")
    axb.barh(y + 1.5 * h, r["cloud"], h, color="#4bacc6", label="gain from cloud masking")
    axb.set_yticks(y, r["label"])
    axb.invert_yaxis()
    axb.axvline(0, color="grey", lw=0.8)
    axb.set_xlabel("Macro-F1 points")
    axb.set_title(f"({'abc'[len(scen)]}) Gap against missingness effects")
    axb.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out = paths.figures_dir(PAPER) / "sen4agrinet_rerun.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(out)


if __name__ == "__main__":
    main()
