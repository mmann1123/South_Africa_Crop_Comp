#!/usr/bin/env python
"""Random search over the paper's ranges (10 configurations per model, zero fill,
10 epochs), chosen on 33TWN accuracy as the paper does. results/hp_search.csv, hp_choice.json."""
from __future__ import annotations

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

N_CONF, EPOCHS = 10, 10


def main() -> None:
    res = paths.results_dir(lib.PAPER)
    with RunLog("03_tune", {"n_conf": N_CONF, "epochs": EPOCHS}, paper=lib.PAPER) as log:
        C, y = lib.load_train()
        med = lib.pool_medians(C)
        x = torch.from_numpy(tm.make_input(C, "Z", med)).cuda()
        yt = torch.from_numpy(y).cuda()
        Cv, yv = lib.comp(tl.VAL_TILE)
        xv = tm.make_input(Cv, "Z", med)
        rows = []
        for name in ("lstm", "tempcnn"):
            rng = np.random.default_rng(42 if name == "lstm" else 43)
            for k in range(N_CONF):
                hp = tm.sample_hp(name, rng)
                torch.manual_seed(k)
                m = tm.build(name, hp, 9).cuda()
                t0 = time.time()
                acc, _ = tm.train(m, x, yt, xv, yv, hp, EPOCHS, k)
                rows.append({"model": name, "conf": k, **hp, "val_oa": acc, "minutes": (time.time() - t0) / 60})
                print(rows[-1], flush=True)
                del m
        df = pd.DataFrame(rows)
        write_csv(df, res / "hp_search.csv", "03_tune", ["composites/*.npz"])
        choice = {n: {k: (v.item() if hasattr(v, "item") else v) for k, v in g.loc[g["val_oa"].idxmax()].dropna().items() if k in ("lr", "wd", "hidden", "layers", "kernel")}
                  for n, g in df.groupby("model")}
        for c in choice.values():
            for k in ("hidden", "layers", "kernel"):
                if k in c:
                    c[k] = int(c[k])
        (res / "hp_choice.json").write_text(json.dumps(choice, indent=1))
        print(choice)
        log.count("configs", len(df))


if __name__ == "__main__":
    main()
