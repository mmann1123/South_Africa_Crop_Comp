"""Check that the PyTorch L-BFGS logistic head equals scikit-learn's LogisticRegression.

Fits both on field-level MiniRocket features (seed 42) for each C and compares the
validation predictions and probabilities. Writes results/head_equivalence.csv.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/check_head_equivalence.py
"""
import os
import time

import common
import numpy as np
import pandas as pd
import torch
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_minirocket import C_GRID, N_JOBS, TORCH_THREADS, LogRegHead

torch.set_num_threads(TORCH_THREADS)
data = common.load_training_data("field")
seq = lambda X: common.to_sequences(X).transpose(0, 2, 1)
mr = MiniRocket(n_kernels=10000, random_state=42, n_jobs=N_JOBS).fit(seq(data["X_train"]))
fs = StandardScaler().fit(mr.transform(seq(data["X_train"])))
F_tr = fs.transform(mr.transform(seq(data["X_train"]))).astype(np.float32)
F_va = fs.transform(mr.transform(seq(data["X_val"]))).astype(np.float32)
y_tr, K = data["y_train"], len(data["le"].classes_)

rows = []
for C in C_GRID:
    for it in [500, 5000]:
        t = time.time(); ours = LogRegHead(C, max_iter=it).fit(F_tr, y_tr, K); t_ours = time.time() - t
        t = time.time()
        sk = LogisticRegression(C=C, class_weight="balanced", max_iter=it, tol=1e-8).fit(F_tr, y_tr)
        t_sk = time.time() - t
        po, ps = ours.predict_proba(F_va), sk.predict_proba(F_va)
        rows.append({"C": C, "max_iter": it, "ours_iter": ours.n_iter_, "sk_iter": int(sk.n_iter_[0]),
                     "val_pred_agreement": float((po.argmax(1) == ps.argmax(1)).mean()),
                     "max_abs_proba_diff": float(np.abs(po - ps).max()),
                     "ours_s": round(t_ours, 1), "sk_s": round(t_sk, 1)})
        print(rows[-1])
out = os.path.join(common.RESULTS_DIR, "head_equivalence.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
