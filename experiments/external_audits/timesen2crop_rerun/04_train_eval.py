#!/usr/bin/env python
"""Train LSTM and TempCNN under each training fill, 5 seeds, hyperparameters fixed
from hp_choice.json (30 epochs, best checkpoint on 33TWN), and score every test
fill of the design on 33TWN, 33UVP 2018 and 33UVP 2019. Resumable per run.
Output: <root>/runs/<model>_<fill>_seed<k>.pt, results/runs/<same>.csv, results/train_eval.csv."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timesen2crop_rerun import trainlib as lib, tslib as tl, tsmodels as tm  # noqa: E402

TEST_FILLS = {"Z": ["Z", "M", "L"], "M": ["M", "Z"], "L": ["L"], "Zm": ["Zm"]}
EPOCHS = 30


def main(models: list[str], fills: list[str], seeds: list[int]) -> None:
    res, root = paths.results_dir(lib.PAPER), paths.paper_root(lib.PAPER)
    (root / "runs").mkdir(exist_ok=True)
    (res / "runs").mkdir(exist_ok=True)
    hp = json.loads((res / "hp_choice.json").read_text())
    with RunLog("04_train_eval", {"models": models, "fills": fills, "seeds": seeds, "epochs": EPOCHS, "hp": hp}, paper=lib.PAPER) as log:
        C, y = lib.load_train()
        med = lib.pool_medians(C)
        np.save(root / "pool_medians.npy", med)
        yt = torch.from_numpy(y).cuda()
        tests = {t: lib.comp(t) for t in (tl.VAL_TILE, tl.TEST_TILE, tl.YEAR_TILE)}
        prior = np.bincount(tests[tl.TEST_TILE][1], minlength=16) / len(tests[tl.TEST_TILE][1])
        Cv, yv = tests[tl.VAL_TILE]
        for fill in fills:
            x = torch.from_numpy(tm.make_input(C, fill, med)).cuda()
            xv = tm.make_input(Cv, fill, med)
            d = x.shape[2]
            for name in models:
                for seed in seeds:
                    tag = f"{name}_{fill}_seed{seed}"
                    out = res / "runs" / f"{tag}.csv"
                    if out.exists():
                        continue
                    t0 = time.time()
                    torch.manual_seed(seed)
                    m = tm.build(name, hp[name], d).cuda()
                    best, state = tm.train(m, x, yt, xv, yv, hp[name], EPOCHS, seed)
                    torch.save(state, root / "runs" / f"{tag}.pt")
                    m.load_state_dict(state)
                    rows = []
                    for tf in TEST_FILLS[fill]:
                        for tile, (Ct, yy) in tests.items():
                            p = tm.predict(m, tm.make_input(Ct, tf, med))
                            rows.append({"model": name, "train_fill": fill, "test_fill": tf, "seed": seed, "tile": tile, **lib.score(yy, p, prior)})
                    pd.DataFrame(rows).to_csv(out, index=False)
                    r = pd.DataFrame(rows).query("test_fill == @fill")
                    print(tag, f"val {best:.4f}", r.set_index("tile")[["oa", "f1m"]].round(4).to_dict("index"), f"{(time.time() - t0) / 60:.1f} min", flush=True)
                    del m
            del x
        df = pd.concat([pd.read_csv(f) for f in sorted((res / "runs").glob("*.csv"))], ignore_index=True)
        write_csv(df, res / "train_eval.csv", "04_train_eval", ["results/runs/*.csv"])
        log.count("rows", len(df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "tempcnn"])
    ap.add_argument("--fills", nargs="+", default=["Z", "M", "L", "Zm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    main(**vars(ap.parse_args()))
