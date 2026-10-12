"""MiniRocket-S (sparsemax gate) and its dense control (softmax gate) on frozen MiniRocket features.

The features are exactly those of the reference MiniRocket runs (features.py, identity
check per seed); only the head differs. Training harness = published L-TAE-S
(focal loss gamma=2 alpha=1/count, AdamW lr 1e-3 wd 1e-4, cosine T_max=100, patience 15
on validation macro-F1, AMP, grad clip 1.0, 5-seed mean logits). Pixel level:
early stopping on <=50 validation pixels per field (see README).

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/train_gated.py --level field --gate sparsemax
  ... --gate softmax ; --level pixel
Smoke test: --seeds 42 --max-epochs 2 --out-tag smoke
"""
import argparse
import json
import os
import sys
import time

import features as fx  # sets sys.path (parent experiment, field_reduction)
import common

common.set_determinism()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from models_arch import (  # noqa: E402
    WeightedFocalLoss, compute_focal_loss_weights, evaluate_sparse, get_device, train_epoch_sparse,
)
from sparse_head import GatedHead  # noqa: E402
import train_minirocket as tm  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS, LR, WEIGHT_DECAY, PATIENCE = 100, 1e-3, 1e-4, 15
BATCH_SIZE = {"field": 128, "pixel": 2048}
VAL_CAP = 50  # pixel level: validation pixels per field for early stopping
HEAD_CFG = dict(n_head=16, d_k=8, bottleneck=64, gamma=1.5, dropout=0.3)


def loader(F, y, bs, shuffle=False, seed=None):
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(torch.from_numpy(F), torch.from_numpy(y.astype(np.int64))),
                      batch_size=bs, shuffle=shuffle, generator=gen, num_workers=0, pin_memory=True)


@torch.no_grad()
def logits_on(model, m, X, bs, device):
    """Model logits on all rows of X, transforming MiniRocket features in batches."""
    out = []
    for F in fx.iter_features(m, X):
        lg, _ = evaluate_sparse(model, loader(F.astype(np.float16), np.zeros(len(F)), bs), device)
        out.append(lg)
    return torch.cat(out).numpy()


def write_holdout(logits, fids, le, level, stem):
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()
    preds = probs.argmax(1)
    if level == "pixel":
        out_fids, field_pred = common.field_majority_vote(fids, preds)
        pdf = pd.DataFrame(probs, columns=le.classes_)
        pdf.insert(0, "fid", fids)
        proba = pdf.groupby("fid").mean().loc[out_fids]
    else:
        out_fids, field_pred = fids, preds
        proba = pd.DataFrame(probs, columns=le.classes_, index=pd.Index(fids, name="fid"))
    path = os.path.join(fx.PRED_DIR, f"{stem}.csv")
    common.write_predictions(out_fids, le.inverse_transform(field_pred), path)
    proba.reset_index().to_parquet(os.path.join(fx.PRED_DIR, f"{stem}_proba.parquet"), index=False)
    return path


def run_seed(seed, data, X_hold, level, gate, out_dir, device, max_epochs):
    m, src = fx.load_seed_model(level, seed)
    max_dp = fx.identity_check(m, data, level, seed)
    rows = fx.train_rows(data, level, seed)
    F_tr = fx.features(m, data["X_train"][rows], dtype=np.float16)
    y_tr = data["y_train"][rows]
    if level == "pixel":
        vrows = tm.cap_pixels(data["fid_val"], VAL_CAP, seed)
    else:
        vrows = np.arange(len(data["y_val"]))
    F_va = fx.features(m, data["X_val"][vrows], dtype=np.float16)
    y_va = data["y_val"][vrows]
    K = len(data["le"].classes_)
    bs = BATCH_SIZE[level]

    common.seed_everything(seed)
    criterion = WeightedFocalLoss(compute_focal_loss_weights(y_tr, K), gamma=2.0).to(device)
    model = GatedHead(F_tr.shape[1], K, gate=gate, **HEAD_CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)
    amp = torch.amp.GradScaler("cuda")
    tr_loader, va_loader = loader(F_tr, y_tr, bs, True, seed), loader(F_va, y_va, bs)
    ckpt = os.path.join(out_dir, f"gated_{gate}_{level}_seed_{seed}.pt")
    torch.save(model.state_dict(), ckpt)

    best, best_ep, wait, ep_run, t0 = 0.0, 0, 0, 0, time.time()
    for ep in range(max_epochs):
        loss, _ = train_epoch_sparse(model, opt, criterion, tr_loader, amp, device)
        vl, vy = evaluate_sparse(model, va_loader, device)
        vf1 = f1_score(vy, vl.argmax(1).tolist(), average="macro")
        sched.step()
        ep_run = ep + 1
        if ep == 0 or ep_run % 10 == 0:
            print(f"  epoch {ep_run}: loss={loss:.4f} val_f1={vf1:.4f} ({(time.time()-t0)/ep_run:.1f}s/epoch)")
        if vf1 > best:
            best, best_ep, wait = vf1, ep_run, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop at epoch {ep_run}")
                break
    train_s = time.time() - t0

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    stats = model.mask_stats(loader(F_va[:20000], y_va[:20000], bs), device)
    del F_tr, F_va
    t1 = time.time()
    np.save(os.path.join(out_dir, f"logits_test_seed_{seed}.npy"), logits_on(model, m, data["X_test"], bs, device))
    np.save(os.path.join(out_dir, f"logits_holdout_seed_{seed}.npy"), logits_on(model, m, X_hold, bs, device))
    info = {"seed": seed, "train_seconds": round(train_s, 1), "predict_seconds": round(time.time() - t1, 1),
            "epochs_run": ep_run, "best_epoch": best_ep, "best_val_f1_macro": round(best, 6),
            "n_train_rows": int(len(y_tr)), "n_val_rows": int(len(y_va)),
            "mask_stats_val": stats, "feature_source": src, "feature_identity_max_dp": max_dp}
    json.dump(info, open(os.path.join(out_dir, f"seed_{seed}.json"), "w"), indent=2)
    print(f"  seed {seed}: best epoch {best_ep}, val_f1 {best:.4f}, {train_s:.0f}s; masks {stats}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["field", "pixel"], required=True)
    ap.add_argument("--gate", choices=["sparsemax", "softmax"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=common.SEEDS_ENSEMBLE)
    ap.add_argument("--max-epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(tm.TORCH_THREADS)

    model_name = {"sparsemax": "minirocket_s", "softmax": "minirocket_softmax"}[args.gate]
    name = f"{model_name}_{args.level}" + (f"_{args.out_tag}" if args.out_tag else "")
    out_dir = os.path.join(fx.MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    print(f"=== {model_name} ({args.level}) -> {out_dir}")

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
        seed_info.append(run_seed(seed, data, X_hold, args.level, args.gate, out_dir, device, args.max_epochs))

    load = lambda kind, s: np.load(os.path.join(out_dir, f"logits_{kind}_seed_{s}.npy"))
    for s in args.seeds:
        write_holdout(load("holdout", s), hold_fids, le, args.level, f"{name}_seed{s}")
    ens_test = np.mean([load("test", s) for s in args.seeds], 0)
    pred_csv = write_holdout(np.mean([load("holdout", s) for s in args.seeds], 0), hold_fids, le, args.level, name)
    in_region = common.in_region_field_metrics(args.level, data["fid_test"], data["y_test"], ens_test.argmax(1))
    print(f"In-region (field level): {in_region}\nHoldout predictions: {pred_csv}")

    common.write_metadata(out_dir, {
        "model": model_name, "gate": args.gate, "level": args.level, "name": name,
        "head_config": HEAD_CFG, "batch_size": BATCH_SIZE[args.level], "max_epochs": args.max_epochs,
        "pixel_val_cap_per_field": VAL_CAP if args.level == "pixel" else None,
        "protocol": "frozen MiniRocket features (reference seeds) -> GatedHead; L-TAE-S harness; "
                    "5-seed mean logits",
        "seeds": args.seeds, "n_fields": data["n_fields"], "n_rows": data["n_rows"], "input_checks": checks,
        "per_seed": seed_info,
        "per_seed_seconds_mean": round(float(np.mean([s["train_seconds"] for s in seed_info])), 1),
        "training_time_sec_total": round(float(np.sum([s["train_seconds"] for s in seed_info])), 1),
        "metrics": in_region, "holdout_predictions": os.path.relpath(pred_csv, common.REPO),
    })


if __name__ == "__main__":
    main()
