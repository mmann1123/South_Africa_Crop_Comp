"""Verify the reference MiniRocket outputs (../models/minirocket_*) with checked features.

For every seed, recompute test and holdout probabilities from the saved model
(MiniRocket + feature scaler + L2 head) using features.checked_transform (each batch
computed twice and required to agree) and require them to equal the saved
proba_*_seed_{s}.npy. Read-only for ../; writes results/verify_reference.csv.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/verify_reference.py
"""
import os

import features as fx
import common
import numpy as np
import pandas as pd
import torch
import train_minirocket as tm

torch.set_num_threads(tm.TORCH_THREADS)
rows = []
for level in ["field", "pixel"]:
    data = common.load_training_data(level)
    X_hold, _ = common.load_holdout(level, data["feature_cols"], data["scaler"])
    ref_dir = os.path.join(common.MODELS_DIR, f"minirocket_{level}")
    for seed in common.SEEDS_ENSEMBLE:
        m, _ = fx.load_seed_model(level, seed)
        head = tm.LogRegHead(m["C"])
        head.W, head.b = torch.from_numpy(m["W"]), torch.from_numpy(m["b"])
        for kind, X in [("test", data["X_test"]), ("holdout", X_hold)]:
            p = np.concatenate([head.predict_proba(F) for F in fx.iter_features(m, X)])
            saved = np.load(os.path.join(ref_dir, f"proba_{kind}_seed_{seed}.npy"))
            d = float(np.abs(p - saved).max())
            rows.append({"level": level, "seed": seed, "split": kind, "rows": len(p),
                         "max_abs_diff": d, "identical": bool(d == 0.0)})
            print(rows[-1], flush=True)
out = os.path.join(fx.RESULTS_DIR, "verify_reference.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("all identical:", all(r["identical"] for r in rows), "->", out)
