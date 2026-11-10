#!/usr/bin/env python
"""Test-time interventions on the frozen zero-trained (Z) and median-trained (M)
models, 5 seeds each, each input built with the model's own training fill.

month_deletion.csv: per tile and month, macro F1 and OA on pixels that have the
  month, as they are and with the month blanked; cost = as is minus blanked.
pattern_imposition.csv: on 33UVP 2018, blank months absent from a random 2019
  pixel (cross_year), or the same number of this pixel's months at random
  (matched_random), 5 draws; the same in reverse on 33UVP 2019 with 2018 patterns.
by_empty_count.csv: OA of the Z model by a pixel's number of empty months and
  tile, with class-prior reweighting to 33UVP 2018 where a bin is large enough."""
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

SEEDS, DRAWS = range(5), 5


def random_same_count(present: np.ndarray, n_del: np.ndarray, rng) -> np.ndarray:
    """keep mask: for each pixel, delete n_del months chosen at random among its present months."""
    r = rng.random(present.shape)
    r[~present] = np.inf
    rank = r.argsort(1).argsort(1)
    return ~((rank < n_del[:, None]) & present)


def main() -> None:
    res, root = paths.results_dir(lib.PAPER), paths.paper_root(lib.PAPER)
    hp = json.loads((res / "hp_choice.json").read_text())
    med = np.load(root / "pool_medians.npy")
    with RunLog("05_testtime", {"seeds": list(SEEDS), "draws": DRAWS}, paper=lib.PAPER) as log:
        T = {t: lib.comp(t) for t in (tl.TEST_TILE, tl.YEAR_TILE)}
        prior = np.bincount(T[tl.TEST_TILE][1], minlength=16) / len(T[tl.TEST_TILE][1])
        P = {t: tl.present_mask(c) for t, (c, _) in T.items()}
        mon, pat, cnt = [], [], []
        for name in ("lstm", "tempcnn"):
            for fill in ("Z", "M"):
                for seed in SEEDS:
                    m = tm.build(name, hp[name], 9).cuda()
                    m.load_state_dict(torch.load(root / "runs" / f"{name}_{fill}_seed{seed}.pt", map_location="cuda"))
                    base = {"model": name, "train_fill": fill, "seed": seed}
                    for tile, (C, y) in T.items():
                        p0 = tm.predict(m, tm.make_input(C, fill, med))
                        for k in range(12):
                            sel = np.flatnonzero(P[tile][:, k])
                            if len(sel) < 1000:
                                continue
                            keep = np.ones((len(sel), 12), bool)
                            keep[:, k] = False
                            p1 = tm.predict(m, tm.make_input(tl.apply_pattern(C[sel], keep), fill, med))
                            a, b = lib.score(y[sel], p0[sel]), lib.score(y[sel], p1)
                            mon.append({**base, "tile": tile, "month": tl.MONTHS[k], "pixels": len(sel), "f1m_asis": a["f1m"], "f1m_blanked": b["f1m"],
                                        "oa_asis": a["oa"], "oa_blanked": b["oa"]})
                        if fill == "Z":
                            ne = (~P[tile]).sum(1)
                            for n in np.unique(ne):
                                s = ne == n
                                if s.sum() >= 500:
                                    cnt.append({**base, "tile": tile, "empty_months": int(n), "pixels": int(s.sum()), **lib.score(y[s], p0[s], prior)})
                    for tile, other in ((tl.TEST_TILE, tl.YEAR_TILE), (tl.YEAR_TILE, tl.TEST_TILE)):
                        C, y = T[tile]
                        pres = P[tile]
                        a0 = lib.score(y, tm.predict(m, tm.make_input(C, fill, med)))
                        for d in range(DRAWS):
                            rng = np.random.default_rng([42, seed, d, 0 if tile == tl.TEST_TILE else 1])
                            keep = P[other][rng.integers(0, len(P[other]), len(C))]
                            n_del = (pres & ~keep).sum(1)
                            kr = random_same_count(pres, n_del, rng)
                            sx = lib.score(y, tm.predict(m, tm.make_input(tl.apply_pattern(C, keep), fill, med)))
                            sr = lib.score(y, tm.predict(m, tm.make_input(tl.apply_pattern(C, kr), fill, med)))
                            pat.append({**base, "tile": tile, "draw": d, "months_deleted": float(n_del.mean()), "f1m_asis": a0["f1m"], "oa_asis": a0["oa"],
                                        "f1m_cross_year": sx["f1m"], "oa_cross_year": sx["oa"], "f1m_matched_random": sr["f1m"], "oa_matched_random": sr["oa"]})
                    print(name, fill, seed, flush=True)
                    del m
        for fn, rows in (("month_deletion.csv", mon), ("pattern_imposition.csv", pat), ("by_empty_count.csv", cnt)):
            write_csv(pd.DataFrame(rows), res / fn, "05_testtime", ["runs/*.pt", "composites/*.npz"])
        log.count("month_rows", len(mon))


if __name__ == "__main__":
    main()
