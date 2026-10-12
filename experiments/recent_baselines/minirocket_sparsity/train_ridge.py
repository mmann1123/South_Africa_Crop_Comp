"""MiniRocket with its canonical head: RidgeClassifierCV (robustness check for the L2 logistic head).

As in aeon / sktime and Dempster et al.: standardised MiniRocket features ->
RidgeClassifierCV(alphas=logspace(-3, 3, 10)), alpha chosen by its built-in efficient
leave-one-out CV on the training rows; class_weight='balanced' to match how every other
model in the study handles imbalance. Same features and rows as the reference MiniRocket
(features.py: checked transform + per-seed identity check). Ridge gives decision scores,
not probabilities; Table 1's Xent is computed from hard labels, so all metrics apply.
5-seed ensemble = mean decision score.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/train_ridge.py --level field
"""
import argparse
import json
import os
import sys
import time

import features as fx  # sets sys.path
import common

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import train_minirocket as tm  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
GRIDS = {
    "canonical": np.logspace(-3, 3, 10),  # aeon / sktime MiniRocket default
    # canonical grid chose its upper edge (1000) on every field seed; extended upward,
    # same edge rule as the logistic head (selection still LOO on training rows)
    "extended": np.logspace(-3, 8, 12),
}
BLAS_THREADS = 16                # fixed for reproducible linear algebra


def scores_on(clf, m, X):
    return np.concatenate([clf.decision_function(F) for F in fx.iter_features(m, X)])


def write_holdout(scores, fids, le, level, stem):
    preds = scores.argmax(1)
    if level == "pixel":
        out_fids, field_pred = common.field_majority_vote(fids, preds)
    else:
        out_fids, field_pred = fids, preds
    path = os.path.join(fx.PRED_DIR, f"{stem}.csv")
    common.write_predictions(out_fids, le.inverse_transform(field_pred), path)
    return path


def run_seed(seed, data, X_hold, level, out_dir, alphas):
    m, src = fx.load_seed_model(level, seed)
    max_dp = fx.identity_check(m, data, level, seed)
    rows = fx.train_rows(data, level, seed)
    F_tr = fx.features(m, data["X_train"][rows])
    y_tr = data["y_train"][rows]
    t0 = time.time()
    clf = RidgeClassifierCV(alphas=alphas, class_weight="balanced").fit(F_tr, y_tr)
    train_s = time.time() - t0
    del F_tr
    edge = clf.alpha_ in (alphas[0], alphas[-1])
    t1 = time.time()
    np.save(os.path.join(out_dir, f"scores_test_seed_{seed}.npy"), scores_on(clf, m, data["X_test"]))
    np.save(os.path.join(out_dir, f"scores_holdout_seed_{seed}.npy"), scores_on(clf, m, X_hold))
    np.savez(os.path.join(out_dir, f"ridge_seed_{seed}.npz"), coef=clf.coef_, intercept=clf.intercept_,
             alpha=clf.alpha_)
    info = {"seed": seed, "train_seconds": round(train_s, 1), "predict_seconds": round(time.time() - t1, 1),
            "n_train_rows": int(len(y_tr)), "alpha": float(clf.alpha_), "alpha_at_grid_edge": bool(edge),
            "feature_source": src, "feature_identity_max_dp": max_dp}
    json.dump(info, open(os.path.join(out_dir, f"seed_{seed}.json"), "w"), indent=2)
    print(f"  seed {seed}: alpha={clf.alpha_:.4g}{' (GRID EDGE)' if edge else ''}, train {train_s:.0f}s")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["field", "pixel"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=common.SEEDS_ENSEMBLE)
    ap.add_argument("--grid", choices=list(GRIDS), default="canonical")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    alphas = GRIDS[args.grid]
    base = "minirocket_ridge" if args.grid == "canonical" else "minirocket_ridge_ext"
    name = f"{base}_{args.level}" + (f"_{args.out_tag}" if args.out_tag else "")
    out_dir = os.path.join(fx.MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== MiniRocket + RidgeClassifierCV ({args.level}) -> {out_dir}")

    with threadpool_limits(limits=BLAS_THREADS):
        data = common.load_training_data(args.level)
        checks = common.check_against_published(data)
        le = data["le"]
        X_hold, hold_fids = common.load_holdout(args.level, data["feature_cols"], data["scaler"])
        seed_info = []
        for seed in args.seeds:
            js = os.path.join(out_dir, f"seed_{seed}.json")
            if os.path.exists(js) and not args.force:
                print(f"--- seed {seed}: exists, skipping")
                seed_info.append(json.load(open(js)))
                continue
            print(f"--- seed {seed}")
            seed_info.append(run_seed(seed, data, X_hold, args.level, out_dir, alphas))

    load = lambda kind, s: np.load(os.path.join(out_dir, f"scores_{kind}_seed_{s}.npy"))
    for s in args.seeds:
        write_holdout(load("holdout", s), hold_fids, le, args.level, f"{name}_seed{s}")
    pred_csv = write_holdout(np.mean([load("holdout", s) for s in args.seeds], 0), hold_fids, le, args.level, name)
    ens_test = np.mean([load("test", s) for s in args.seeds], 0)
    in_region = common.in_region_field_metrics(args.level, data["fid_test"], data["y_test"], ens_test.argmax(1))
    print(f"In-region (field level): {in_region}\nHoldout predictions: {pred_csv}")

    common.write_metadata(out_dir, {
        "model": "minirocket_ridge", "level": args.level, "name": name,
        "protocol": "frozen MiniRocket features (reference seeds) -> RidgeClassifierCV(alphas=logspace(-3,3,10), "
                    "class_weight='balanced'), alpha by built-in LOO CV on training rows; 5-seed mean decision score",
        "grid": args.grid, "alphas": alphas.tolist(), "blas_threads": BLAS_THREADS,
        "seeds": args.seeds, "n_fields": data["n_fields"], "n_rows": data["n_rows"], "input_checks": checks,
        "per_seed": seed_info,
        "per_seed_seconds_mean": round(float(np.mean([s["train_seconds"] for s in seed_info])), 1),
        "training_time_sec_total": round(float(np.sum([s["train_seconds"] for s in seed_info])), 1),
        "metrics": in_region, "holdout_predictions": os.path.relpath(pred_csv, common.REPO),
    })


if __name__ == "__main__":
    main()
