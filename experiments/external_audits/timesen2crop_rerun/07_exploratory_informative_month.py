#!/usr/bin/env python
"""EXPLORATORY, added after the design's verdict was seen; not covered by its pass marks.

Closest analogue of the AI4FoodSecurity case on this data. A median-trained (M)
model has never seen a zero. One month is blanked at random in 40.6 percent of
pixels (our September share), all other gaps keep the median fill, and the blanked
pixel-months are given either zeros or the training medians. The zero-minus-median
contrast is the cost of hard zero fill for that month, per month, tile, model and
seed (3 draws). results/exploratory_month_zero_vs_median.csv"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timesen2crop_rerun import trainlib as lib, tslib as tl, tsmodels as tm  # noqa: E402

SHARE, DRAWS = 0.406, 3


def main() -> None:
    res, root = paths.results_dir(lib.PAPER), paths.paper_root(lib.PAPER)
    hp = json.loads((res / "hp_choice.json").read_text())
    med = np.load(root / "pool_medians.npy")
    with RunLog("07_exploratory", {"share": SHARE, "draws": DRAWS}, paper=lib.PAPER) as log:
        rows = []
        for tile in (tl.TEST_TILE, tl.YEAR_TILE):
            C, y = lib.comp(tile)
            pres = tl.present_mask(C)
            x0 = tl.fill_median(C, med)
            for name in ("lstm", "tempcnn"):
                for seed in range(5):
                    m = tm.build(name, hp[name], 9).cuda()
                    m.load_state_dict(torch.load(root / "runs" / f"{name}_M_seed{seed}.pt", map_location="cuda"))
                    base = lib.score(y, tm.predict(m, x0))
                    for k in range(12):
                        cand = np.flatnonzero(pres[:, k])
                        if len(cand) < 0.5 * len(C):
                            continue
                        for d in range(DRAWS):
                            rng = np.random.default_rng([42, seed, k, d])
                            sel = rng.choice(cand, int(SHARE * len(C)), replace=False)
                            xm, xz = x0.copy(), x0.copy()
                            xm[sel, k] = med[k]
                            xz[sel, k] = 0.0
                            a, b = lib.score(y, tm.predict(m, xm)), lib.score(y, tm.predict(m, xz))
                            rows.append({"tile": tile, "model": name, "seed": seed, "month": tl.MONTHS[k], "draw": d, "f1m_base": base["f1m"], "oa_base": base["oa"],
                                         "f1m_median": a["f1m"], "f1m_zero": b["f1m"], "oa_median": a["oa"], "oa_zero": b["oa"]})
                    del m
                print(tile, name, flush=True)
        df = pd.DataFrame(rows)
        write_csv(df, res / "exploratory_month_zero_vs_median.csv", "07_exploratory", ["runs/*_M_seed*.pt", "composites/*.npz"])
        log.count("rows", len(df))


if __name__ == "__main__":
    main()
