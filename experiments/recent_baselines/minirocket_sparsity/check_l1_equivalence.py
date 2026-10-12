"""Check the FISTA L1 head against scikit-learn's LogisticRegression(penalty='l1', solver='saga').

Field-level MiniRocket features of seed 42 (regenerated exactly, see features.py).
Writes results/l1_equivalence.csv.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/check_l1_equivalence.py
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import features as fx
from sparse_head import L1Head

import common  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

torch.set_num_threads(16)
data = common.load_training_data("field")
m, _ = fx.load_seed_model("field", 42)
F_tr = fx.features(m, data["X_train"])
F_va = fx.features(m, data["X_val"])
y, K = data["y_train"], len(data["le"].classes_)

def objective(C, W, b):
    """sklearn's L1 objective / n, for comparing solvers (lower = closer to the optimum)."""
    X, yt = torch.from_numpy(F_tr), torch.from_numpy(y.astype(np.int64))
    n = len(y)
    cw = torch.tensor(n / (K * np.bincount(y, minlength=K)), dtype=torch.float32)
    ce = torch.nn.functional.cross_entropy(X @ W + b, yt, weight=cw, reduction="none")
    return float((C * ce.sum() + W.abs().sum()) / n)


rows = []
for C in [0.01, 0.1]:  # saga is single-threaded and too slow at larger C
    t = time.time(); ours = L1Head(C).fit(F_tr, y, K); t_o = time.time() - t
    t = time.time()
    sk = LogisticRegression(penalty="l1", solver="saga", C=C, class_weight="balanced",
                            max_iter=20000, tol=1e-8, random_state=0).fit(F_tr, y)
    t_s = time.time() - t
    po, ps = ours.predict_proba(F_va), sk.predict_proba(F_va)
    rows.append({"C": C, "ours_iter": ours.n_iter_, "ours_converged": ours.converged_,
                 "ours_nonzero_features": ours.n_selected_,
                 "sk_nonzero_features": int((np.abs(sk.coef_).sum(0) > 0).sum()),
                 "ours_objective": objective(C, ours.W, ours.b),
                 "sk_objective": objective(C, torch.from_numpy(sk.coef_.T.astype(np.float32)).contiguous(),
                                           torch.from_numpy(sk.intercept_.astype(np.float32))),
                 "sk_iter": int(sk.n_iter_[0]),
                 "val_pred_agreement": float((po.argmax(1) == ps.argmax(1)).mean()),
                 "max_abs_proba_diff": float(np.abs(po - ps).max()),
                 "ours_s": round(t_o, 1), "sk_s": round(t_s, 1)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(fx.RESULTS_DIR, "l1_equivalence.csv"), index=False)
