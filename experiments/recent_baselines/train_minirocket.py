"""MiniRocket baseline (reviewer R2-Q3a) under the published L-TAE-S protocol.

MiniRocket (Dempster et al., KDD 2021; aeon implementation) turns each 10-month x 6-band
series into ~10,000 random-convolution features (no feature selection) and a linear
head classifies them. The head is multinomial logistic regression, not ridge, so the
model yields class probabilities (the bake-off redux notes ridge heads give 0/1 outputs).

Same inputs, split, scaler, seeds and field aggregation as L-TAE-S
(experiments/field_reduction/train_ltae_sparse_{field,pixel}.py). Model selection uses
the validation split only (C chosen on validation macro-F1, the analogue of early
stopping). Class balance: 'balanced' class weights, the analogue of focal alpha = 1/count.

Logistic head: full-batch L-BFGS in PyTorch (CPU) minimising scikit-learn's
LogisticRegression objective exactly,  C * sum_i w_{y_i} * CE_i + 0.5 * ||W||^2  (bias
unpenalised, w = n / (K * count_k)). scikit-learn's lbfgs took ~9 min per fit at this
width, too slow for 5 seeds x 4 C values x 2 levels.

Pixel level: training uses at most PIXEL_CAP pixels per training field (drawn with the
seed), because 10k features for all 4.8M training pixels would need ~190 GB. Validation,
in-region test and holdout use every pixel (batched), with per-field majority vote.

Run (repo root):
  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/train_minirocket.py --level field
  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/train_minirocket.py --level pixel
Smoke test: add  --seeds 42 --c-grid 1 --out-tag smoke
"""
import argparse
import json
import os
import sys
import time

import common

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from aeon.transformations.collection.convolution_based import MiniRocket  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

N_KERNELS = 10000           # MiniRocket default
C_GRID = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
# Rule fixed after the first field run peaked at the grid edge (C=0.01, all 5 seeds;
# archived in superseded/): while the best validation C is the smallest tried, add a
# decade below it, down to C_FLOOR. Judged on the validation split only.
C_FLOOR = 1e-8
PIXEL_CAP = 50              # training pixels per field (pixel level only)
LBFGS_MAX_ITER = 1000      # converges in 343-588 iterations on field data (results/head_equivalence.csv)
N_JOBS = 16                 # aeon transform threads (transform is deterministic)
TORCH_THREADS = 16          # fixed for reproducible CPU matmuls
BATCH = 200_000             # rows per transform batch for val/test/holdout


# ------------------------------------------------------------------ head

class LogRegHead:
    """Multinomial logistic regression with scikit-learn's objective (see module doc)."""

    def __init__(self, C, max_iter=LBFGS_MAX_ITER):
        self.C, self.max_iter = C, max_iter

    def fit(self, F, y, num_classes):
        torch.manual_seed(0)
        X = torch.from_numpy(F)
        yt = torch.from_numpy(y.astype(np.int64))
        counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        cw = torch.tensor(len(y) / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)
        n, d = X.shape
        W = torch.zeros(d, num_classes, requires_grad=True)
        b = torch.zeros(num_classes, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=self.max_iter, history_size=20,
                                tolerance_grad=1e-6, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(X @ W + b, yt, weight=cw, reduction="none")
            # sklearn objective divided by n (same minimiser, better conditioned)
            loss = (self.C * ce.sum() + 0.5 * (W * W).sum()) / n
            loss.backward()
            return loss

        opt.step(closure)
        self.n_iter_ = opt.state[opt._params[0]]["n_iter"]
        self.W, self.b = W.detach(), b.detach()
        return self

    def predict_proba(self, F):
        with torch.no_grad():
            return torch.softmax(torch.from_numpy(F) @ self.W + self.b, dim=1).numpy()


# ----------------------------------------------------------- feature pipe

def features(mr, fscaler, X):
    """(N, 60) scaled inputs -> standardised MiniRocket features, float32."""
    F = mr.transform(common.to_sequences(X).transpose(0, 2, 1))  # aeon: (N, C, T)
    return fscaler.transform(F).astype(np.float32) if fscaler is not None else F


def batched_proba(mr, fscaler, heads, X):
    """Class probabilities of each head on X, transforming in batches."""
    out = [[] for _ in heads]
    for i in range(0, len(X), BATCH):
        F = features(mr, fscaler, X[i:i + BATCH])
        for j, h in enumerate(heads):
            out[j].append(h.predict_proba(F))
    return [np.concatenate(o) for o in out]


def cap_pixels(fids, cap, seed):
    """Indices of at most `cap` rows per field, drawn with RandomState(seed)."""
    rng = np.random.RandomState(seed)
    idx = pd.Series(np.arange(len(fids))).groupby(fids)
    keep = [g.values if len(g) <= cap else rng.choice(g.values, cap, replace=False)
            for _, g in idx]
    return np.sort(np.concatenate(keep))


# ------------------------------------------------------------------ main

def run_seed(seed, data, X_hold, level, num_classes, out_dir, c_grid):
    common.seed_everything(seed)
    t0 = time.time()
    if level == "pixel":
        sel = cap_pixels(data["fid_train"], PIXEL_CAP, seed)
        X_tr, y_tr = data["X_train"][sel], data["y_train"][sel]
    else:
        X_tr, y_tr = data["X_train"], data["y_train"]

    mr = MiniRocket(n_kernels=N_KERNELS, random_state=seed, n_jobs=N_JOBS)
    mr.fit(common.to_sequences(X_tr).transpose(0, 2, 1))
    F_tr = mr.transform(common.to_sequences(X_tr).transpose(0, 2, 1))
    fscaler = StandardScaler().fit(F_tr)
    F_tr = fscaler.transform(F_tr).astype(np.float32)

    c_grid = sorted(c_grid)
    heads, val_f1 = [], []

    def add(Cs):
        new_heads = [LogRegHead(C).fit(F_tr, y_tr, num_classes) for C in Cs]
        for C, h, p in zip(Cs, new_heads, batched_proba(mr, fscaler, new_heads, data["X_val"])):
            # validation macro-F1 at the training unit (pixels / fields), as in L-TAE-S early stopping
            heads.append(h)
            val_f1.append(f1_score(data["y_val"], p.argmax(1), average="macro"))
            print(f"  C={C:g}: val_f1={val_f1[-1]:.4f}  lbfgs_iter={h.n_iter_}")

    add(c_grid)
    while int(np.argmax(val_f1)) == 0 and c_grid[0] / 10 >= C_FLOOR:
        # best at the lower edge: prepend one decade down (lists stay sorted by C)
        c_new = c_grid[0] / 10
        prev_heads, prev_f1 = heads[:], val_f1[:]
        heads.clear(), val_f1.clear()
        add([c_new])
        heads.extend(prev_heads), val_f1.extend(prev_f1)
        c_grid.insert(0, c_new)
    del F_tr
    best = int(np.argmax(val_f1))  # ties -> smallest C (strongest regularisation)
    head = heads[best]
    train_seconds = time.time() - t0

    t1 = time.time()
    p_test = batched_proba(mr, fscaler, [head], data["X_test"])[0]
    p_hold = batched_proba(mr, fscaler, [head], X_hold)[0]
    predict_seconds = time.time() - t1

    np.save(os.path.join(out_dir, f"proba_test_seed_{seed}.npy"), p_test)
    np.save(os.path.join(out_dir, f"proba_holdout_seed_{seed}.npy"), p_hold)
    joblib.dump({"minirocket": mr, "feature_scaler": fscaler, "W": head.W.numpy(),
                 "b": head.b.numpy(), "C": c_grid[best]},
                os.path.join(out_dir, f"minirocket_{level}_seed_{seed}.joblib"))
    info = {"seed": seed, "train_seconds": round(train_seconds, 1),
            "predict_seconds": round(predict_seconds, 1), "n_train_rows": int(len(y_tr)),
            "C_grid": c_grid, "val_f1_by_C": [round(v, 6) for v in val_f1],
            "chosen_C": c_grid[best], "lbfgs_iter": [int(h.n_iter_) for h in heads],
            "lbfgs_converged": [int(h.n_iter_) < LBFGS_MAX_ITER for h in heads]}
    json.dump(info, open(os.path.join(out_dir, f"seed_{seed}.json"), "w"), indent=2)
    print(f"  seed {seed}: C={c_grid[best]:g}, train {train_seconds:.0f}s, predict {predict_seconds:.0f}s")
    return info


def write_holdout(probs, fids, le, level, stem):
    preds = probs.argmax(1)
    if level == "pixel":
        out_fids, field_pred = common.field_majority_vote(fids, preds)
        pdf = pd.DataFrame(probs, columns=le.classes_)
        pdf.insert(0, "fid", fids)
        proba = pdf.groupby("fid").mean().loc[out_fids]
    else:
        out_fids, field_pred = fids, preds
        proba = pd.DataFrame(probs, columns=le.classes_, index=pd.Index(fids, name="fid"))
    path = os.path.join(common.PRED_DIR, f"{stem}.csv")
    common.write_predictions(out_fids, le.inverse_transform(field_pred), path)
    proba.reset_index().to_parquet(os.path.join(common.PRED_DIR, f"{stem}_proba.parquet"), index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["field", "pixel"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=common.SEEDS_ENSEMBLE)
    ap.add_argument("--c-grid", type=float, nargs="+", default=C_GRID)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(TORCH_THREADS)

    name = f"minirocket_{args.level}" + (f"_{args.out_tag}" if args.out_tag else "")
    out_dir = os.path.join(common.MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== MiniRocket ({args.level}) -> {out_dir}")

    data = common.load_training_data(args.level)
    checks = common.check_against_published(data)
    le = data["le"]
    num_classes = len(le.classes_)
    X_hold, hold_fids = common.load_holdout(args.level, data["feature_cols"], data["scaler"])
    joblib.dump(data["scaler"], os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))
    joblib.dump(data["feature_cols"], os.path.join(out_dir, "feature_columns.joblib"))

    seed_info = []
    for seed in args.seeds:
        js = os.path.join(out_dir, f"seed_{seed}.json")
        if os.path.exists(js) and not args.force:
            print(f"--- seed {seed}: exists, skipping")
            seed_info.append(json.load(open(js)))
            continue
        print(f"--- seed {seed}")
        seed_info.append(run_seed(seed, data, X_hold, args.level, num_classes, out_dir, args.c_grid))

    # 5-seed ensemble: mean class probabilities
    p_test = np.mean([np.load(os.path.join(out_dir, f"proba_test_seed_{s}.npy")) for s in args.seeds], 0)
    p_hold_seeds = {s: np.load(os.path.join(out_dir, f"proba_holdout_seed_{s}.npy")) for s in args.seeds}
    for s, p in p_hold_seeds.items():
        write_holdout(p, hold_fids, le, args.level, f"{name}_seed{s}")
    pred_csv = write_holdout(np.mean(list(p_hold_seeds.values()), 0), hold_fids, le, args.level, name)
    in_region = common.in_region_field_metrics(args.level, data["fid_test"], data["y_test"], p_test.argmax(1))
    print(f"In-region (field level): {in_region}")
    print(f"Holdout predictions: {pred_csv}")

    common.write_metadata(out_dir, {
        "model": "minirocket", "level": args.level, "name": name,
        "protocol": "aeon MiniRocket (10,000 kernels, random_state=seed) -> StandardScaler -> "
                    "multinomial logistic regression (balanced class weights, L2, C chosen on "
                    "validation macro-F1); 5-seed mean probabilities",
        "n_kernels": N_KERNELS, "c_grid": args.c_grid,
        "pixel_cap_per_field": PIXEL_CAP if args.level == "pixel" else None,
        "lbfgs_max_iter": LBFGS_MAX_ITER, "torch_threads": TORCH_THREADS, "aeon_n_jobs": N_JOBS,
        "seeds": args.seeds, "n_fields": data["n_fields"], "n_rows": data["n_rows"],
        "input_checks": checks, "per_seed": seed_info,
        "per_seed_seconds_mean": round(float(np.mean([s["train_seconds"] for s in seed_info])), 1),
        "training_time_sec_total": round(float(np.sum([s["train_seconds"] for s in seed_info])), 1),
        "metrics": in_region, "holdout_predictions": os.path.relpath(pred_csv, common.REPO),
    })


if __name__ == "__main__":
    main()
