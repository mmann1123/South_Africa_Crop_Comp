#!/usr/bin/env python
"""Monthly composites per tile by the paper's rule, and the empty-month and flag
statistics the paper never reports (results/empty_month_share.csv: share of
pixels with no clear acquisition per tile and month; results/acquisitions_per_month.csv).
Composites cached as <root>/composites/<tile>.npz: C (N, 12, 9) float32 NaN when empty, y."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timesen2crop_rerun import tslib as tl  # noqa: E402

PAPER = "timesen2crop_rerun"


def main() -> None:
    root = paths.paper_root(PAPER)
    (root / "composites").mkdir(exist_ok=True)
    with RunLog("02_composites", {}, paper=PAPER) as log:
        share, acq, n = [], [], {}
        for tile in tl.TRAIN_TILES + [tl.VAL_TILE, tl.TEST_TILE, tl.YEAR_TILE]:
            z = np.load(root / "arrays" / f"{tile}.npz")
            C = tl.composite(z["X"], z["flag"], z["dates"])
            np.savez(root / "composites" / f"{tile}.npz", C=C, y=z["y"])
            p = tl.present_mask(C)
            n[tile] = len(C)
            share.append({"tile": tile, **{tl.MONTHS[k]: 1 - p[:, k].mean() for k in range(12)}, "mean_empty_months": (~p).sum(1).mean(),
                          "zero_valued_clear_rows": int(((z["X"] == 0).all(2) & (z["flag"] == 0)).sum()), "non_clear_share": float((z["flag"] != 0).mean())})
            m = tl.month_index(z["dates"])
            acq.append({"tile": tile, **{tl.MONTHS[k]: int((m == k).sum()) for k in range(12)}})
        s = pd.DataFrame(share)
        tr = s[s["tile"].isin(tl.TRAIN_TILES)]
        w = np.array([n[t] for t in tr["tile"]], float)
        pool = {"tile": "POOL_13_TRAIN", **{c: float(np.average(tr[c], weights=w)) for c in tl.MONTHS + ["mean_empty_months", "non_clear_share"]}}
        s = pd.concat([s, pd.DataFrame([pool])], ignore_index=True)
        write_csv(s, paths.results_dir(PAPER) / "empty_month_share.csv", "02_composites", ["arrays/*.npz"])
        write_csv(pd.DataFrame(acq), paths.results_dir(PAPER) / "acquisitions_per_month.csv", "02_composites", ["arrays/*.npz"])
        pd.set_option("display.width", 250)
        print(s.round(3).to_string())
        print(pd.DataFrame(acq).to_string())
        log.count("tiles", len(share))


if __name__ == "__main__":
    main()
