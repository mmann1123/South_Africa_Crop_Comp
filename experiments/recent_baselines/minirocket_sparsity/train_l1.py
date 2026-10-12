"""MiniRocket-L1: L1-regularised logistic head on the frozen reference MiniRocket features.

Same features and training rows as the reference MiniRocket (features.py, identity check
per seed); same C grid and lower-edge extension rule; C chosen on validation macro-F1
over all validation rows. The C path is fitted from strongest to weakest regularisation,
each fit warm-started from the previous one. 5-seed mean probabilities.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/train_l1.py --level field
Smoke test: --seeds 42 --c-grid 0.01 --out-tag smoke
"""
import argparse
import json
import os
import sys
import time

import features as fx  # sets sys.path
import common

common.set_determinism()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from sparse_head import L1Head  # noqa: E402
import train_minirocket as tm  # noqa: E402
from train_gated import write_holdout  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # FISTA on GPU (deterministic CUDA settings)


def val_f1s(m, heads, data):
    probs = [[] for _ in heads]
    for F in fx.iter_features(m, data["X_val"]):
        for j, h in enumerate(heads):
            probs[j].append(h.predict_proba(F))
    return [f1_score(data["y_val"], np.concatenate(p).argmax(1), average="macro") for p in probs]


def run_seed(seed, data, X_hold, level, out_dir, c_grid):
    m, src = fx.load_seed_model(level, seed)
    max_dp = fx.identity_check(m, data, level, seed)
    rows = fx.train_rows(data, level, seed)
    F_tr = fx.features(m, data["X_train"][rows])
    y_tr = data["y_train"][rows]
    K = len(data["le"].classes_)
    t0 = time.time()

    fits = {}

    def fit_path(Cs):
        """Fit Cs in increasing order, warm-starting each from the closest fitted smaller C."""
        for C in sorted(Cs):
            prev = max([c for c in fits if c < C], default=None)
            W0, b0 = (fits[prev].W.numpy(), fits[prev].b.numpy()) if prev is not None else (None, None)
            fits[C] = L1Head(C, device=DEVICE).fit(F_tr, y_tr, K, W0, b0)
            h = fits[C]
            print(f"  C={C:g}: iter={h.n_iter_} converged={h.converged_} nonzero={h.n_selected_}")

    c_grid = sorted(c_grid)
    # Walk the C path from strong to weak regularisation, scoring each fit on validation,
    # and stop once validation macro-F1 has fallen at two successive C values. Weakly
    # regularised L1 fits are very slow at pixel level (hours per C) and past the
    # validation peak they cannot be selected. Selection still uses validation only.
    f1 = {}
    for C in sorted(c_grid):
        fit_path([C])
        f1[C] = val_f1s(m, [fits[C]], data)[0]
        path = [f1[c] for c in sorted(f1)]
        if len(path) >= 3 and path[-1] < path[-2] < path[-3]:
            print(f"  validation F1 fell twice in a row; path stopped at C={C:g}")
            break
    c_grid = sorted(f1)
    while min(f1, key=lambda c: (-f1[c], c)) == c_grid[0] and c_grid[0] / 10 >= tm.C_FLOOR:
        c_new = c_grid[0] / 10  # best at the lower edge: extend one decade down
        fits_prev = dict(fits)
        fits.clear()
        fit_path([c_new])  # cold start: nothing smaller is fitted
        fits.update(fits_prev)
        f1[c_new] = val_f1s(m, [fits[c_new]], data)[0]
        c_grid.insert(0, c_new)
    for c in c_grid:
        print(f"  C={c:g}: val_f1={f1[c]:.4f}")
    best = min(f1, key=lambda c: (-f1[c], c))  # ties -> smallest C
    head = fits[best]
    train_s = time.time() - t0
    del F_tr

    t1 = time.time()
    p_test = np.concatenate([head.predict_proba(F) for F in fx.iter_features(m, data["X_test"])])
    p_hold = np.concatenate([head.predict_proba(F) for F in fx.iter_features(m, X_hold)])
    np.save(os.path.join(out_dir, f"proba_test_seed_{seed}.npy"), p_test)
    np.save(os.path.join(out_dir, f"proba_holdout_seed_{seed}.npy"), p_hold)
    np.savez(os.path.join(out_dir, f"l1_head_seed_{seed}.npz"), W=head.W.numpy(), b=head.b.numpy(), C=best)
    info = {"seed": seed, "train_seconds": round(train_s, 1), "predict_seconds": round(time.time() - t1, 1),
            "n_train_rows": int(len(y_tr)), "C_grid": c_grid,
            "val_f1_by_C": [round(f1[c], 6) for c in c_grid], "chosen_C": best,
            "nonzero_features_by_C": [fits[c].n_selected_ for c in c_grid],
            "fista_iter": [fits[c].n_iter_ for c in c_grid],
            "fista_converged": [bool(fits[c].converged_) for c in c_grid],
            "nonzero_features_chosen": head.n_selected_,
            "feature_source": src, "feature_identity_max_dp": max_dp}
    json.dump(info, open(os.path.join(out_dir, f"seed_{seed}.json"), "w"), indent=2)
    print(f"  seed {seed}: C={best:g}, {head.n_selected_} features, train {train_s:.0f}s")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["field", "pixel"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=common.SEEDS_ENSEMBLE)
    ap.add_argument("--c-grid", type=float, nargs="+", default=tm.C_GRID)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(tm.TORCH_THREADS)

    name = f"minirocket_l1_{args.level}" + (f"_{args.out_tag}" if args.out_tag else "")
    out_dir = os.path.join(fx.MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== MiniRocket-L1 ({args.level}) -> {out_dir}")

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
        seed_info.append(run_seed(seed, data, X_hold, args.level, out_dir, args.c_grid))

    load = lambda kind, s: np.load(os.path.join(out_dir, f"proba_{kind}_seed_{s}.npy"))
    for s in args.seeds:
        write_holdout(np.log(np.clip(load("holdout", s), 1e-30, None)), hold_fids, le, args.level, f"{name}_seed{s}")
    p_hold = np.mean([load("holdout", s) for s in args.seeds], 0)
    p_test = np.mean([load("test", s) for s in args.seeds], 0)
    pred_csv = write_holdout(np.log(np.clip(p_hold, 1e-30, None)), hold_fids, le, args.level, name)
    in_region = common.in_region_field_metrics(args.level, data["fid_test"], data["y_test"], p_test.argmax(1))
    print(f"In-region (field level): {in_region}\nHoldout predictions: {pred_csv}")

    common.write_metadata(out_dir, {
        "model": "minirocket_l1", "level": args.level, "name": name, "c_grid": args.c_grid,
        "fista_device": DEVICE,
        "protocol": "frozen MiniRocket features (reference seeds) -> L1 multinomial logistic "
                    "(FISTA, balanced class weights), C on validation macro-F1 with lower-edge "
                    "extension; 5-seed mean probabilities",
        "seeds": args.seeds, "n_fields": data["n_fields"], "n_rows": data["n_rows"], "input_checks": checks,
        "per_seed": seed_info,
        "per_seed_seconds_mean": round(float(np.mean([s["train_seconds"] for s in seed_info])), 1),
        "training_time_sec_total": round(float(np.sum([s["train_seconds"] for s in seed_info])), 1),
        "metrics": in_region, "holdout_predictions": os.path.relpath(pred_csv, common.REPO),
    })


if __name__ == "__main__":
    main()
