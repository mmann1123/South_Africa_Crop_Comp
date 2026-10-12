"""TimesNet baseline (reviewer R2-Q3a) under the published L-TAE-S protocol.

Same inputs, split, scaler, loss, optimiser, schedule, early stopping, seeds and
ensembling as experiments/field_reduction/train_ltae_sparse_{field,pixel}.py; only the
backbone differs (TimesNet, vendored from THUML Time-Series-Library, see tslib_timesnet/).
Writes only under experiments/recent_baselines/.

Run (repo root):
  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/train_timesnet.py --level field
  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/train_timesnet.py --level pixel
Smoke test: add  --seeds 42 --max-epochs 2 --out-tag smoke
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import common  # sets sys.path for the field_reduction helpers; must precede torch use

common.set_determinism()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from models_arch import (  # noqa: E402
    TemporalDataset, WeightedFocalLoss, compute_focal_loss_weights, evaluate, get_device,
    train_epoch, T_SEQ, N_BANDS,
)
from tslib_timesnet import TSLIB_COMMIT, load_timesnet  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

# Training protocol: identical to the published L-TAE-S scripts
N_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
BATCH_SIZE = {"field": 128, "pixel": 2048}
NUM_WORKERS = {"field": 2, "pixel": 4}

# TimesNet hyperparameters: TSLib classification defaults scaled to a 10-step series.
# Fixed before any holdout result was seen (see README); not tuned.
TIMESNET_CFG = dict(task_name="classification", seq_len=T_SEQ, label_len=0, pred_len=0,
                    enc_in=N_BANDS, d_model=32, d_ff=64, e_layers=2, top_k=2,
                    num_kernels=6, embed="fixed", freq="h", dropout=0.1)


class TimesNetClassifier(nn.Module):
    """Wraps TSLib's TimesNet so it takes (B, T, C) and returns logits.

    No padding: every series has all 10 months, so the padding mask is all ones.
    """

    def __init__(self, num_classes):
        super().__init__()
        Model = load_timesnet()
        self.net = Model(SimpleNamespace(**TIMESNET_CFG, num_class=num_classes))

    def forward(self, x):
        mark = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        return self.net(x, mark, None, None)


def make_loader(X, y, batch_size, shuffle, workers, seed=None):
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TemporalDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                      generator=gen, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0)


def train_one_seed(seed, data, num_classes, criterion, out_dir, device, max_epochs, level):
    """Train one seed exactly as the published L-TAE-S loop; returns timing info."""
    common.seed_everything(seed)
    bs, nw = BATCH_SIZE[level], NUM_WORKERS[level]
    train_loader = make_loader(data["X_train"], data["y_train"], bs, True, nw, seed)
    val_loader = make_loader(data["X_val"], data["y_val"], bs, False, nw)

    model = TimesNetClassifier(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    scaler_amp = torch.amp.GradScaler("cuda")
    ckpt = os.path.join(out_dir, f"timesnet_{level}_seed_{seed}.pt")
    torch.save(model.state_dict(), ckpt)

    best_f1, best_epoch, wait, epochs_run = 0.0, 0, 0, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        loss = train_epoch(model, optimizer, criterion, train_loader, scaler_amp, device)
        val_logits, val_labels = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_logits.argmax(1).tolist(), average="macro")
        scheduler.step()
        epochs_run = epoch + 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: loss={loss:.4f} val_f1={val_f1:.4f} "
                  f"({(time.time()-t0)/epochs_run:.1f}s/epoch)")
        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop at epoch {epoch+1}")
                break
    seconds = time.time() - t0
    info = {"seed": seed, "train_seconds": round(seconds, 1), "epochs_run": epochs_run,
            "best_epoch": best_epoch, "best_val_f1_macro": round(best_f1, 6),
            "mean_epoch_seconds": round(seconds / max(epochs_run, 1), 2)}
    json.dump(info, open(os.path.join(out_dir, f"seed_{seed}.json"), "w"), indent=2)
    return info


def load_seed(seed, num_classes, out_dir, device, level):
    model = TimesNetClassifier(num_classes).to(device)
    ckpt = os.path.join(out_dir, f"timesnet_{level}_seed_{seed}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["field", "pixel"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=common.SEEDS_ENSEMBLE)
    ap.add_argument("--max-epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--out-tag", default="", help="suffix for smoke/repro runs, e.g. 'smoke'")
    ap.add_argument("--force", action="store_true", help="retrain seeds that already exist")
    args = ap.parse_args()

    name = f"timesnet_{args.level}" + (f"_{args.out_tag}" if args.out_tag else "")
    out_dir = os.path.join(common.MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    print(f"=== TimesNet ({args.level}) -> {out_dir} | device {device} | TSLib {TSLIB_COMMIT[:7]}")

    data = common.load_training_data(args.level)
    checks = common.check_against_published(data)
    num_classes = len(data["le"].classes_)
    alpha = compute_focal_loss_weights(data["y_train"], num_classes)
    criterion = WeightedFocalLoss(alpha, gamma=2.0).to(device)
    joblib.dump(data["scaler"], os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(data["le"], os.path.join(out_dir, "label_encoder.joblib"))
    joblib.dump(data["feature_cols"], os.path.join(out_dir, "feature_columns.joblib"))

    seed_info = []
    for seed in args.seeds:
        done = os.path.exists(os.path.join(out_dir, f"seed_{seed}.json"))
        if done and not args.force:
            print(f"--- seed {seed}: checkpoint exists, skipping training")
            seed_info.append(json.load(open(os.path.join(out_dir, f"seed_{seed}.json"))))
            continue
        print(f"--- seed {seed}")
        seed_info.append(train_one_seed(seed, data, num_classes, criterion, out_dir,
                                        device, args.max_epochs, args.level))

    # In-region (internal 20% test split) and holdout predictions, 5-seed mean logits.
    # Evaluation batch size and order are fixed: TimesNet picks its FFT periods from the
    # batch-mean spectrum, so a sample's output depends on its batch.
    bs = BATCH_SIZE[args.level]
    test_loader = make_loader(data["X_test"], data["y_test"], bs, False, 0)
    X_hold, hold_fids = common.load_holdout(args.level, data["feature_cols"], data["scaler"])
    hold_loader = make_loader(X_hold, np.zeros(len(X_hold), dtype=np.int64), bs, False, 0)

    test_logits, hold_logits = [], []
    for seed in args.seeds:
        model = load_seed(seed, num_classes, out_dir, device, args.level)
        tl, test_labels = evaluate(model, test_loader, device)
        hl, _ = evaluate(model, hold_loader, device)
        test_logits.append(tl)
        hold_logits.append(hl)
        _write_holdout(hl, hold_fids, data["le"], args.level, f"{name}_seed{seed}")
    ens_test = torch.stack(test_logits).mean(0)
    ens_hold = torch.stack(hold_logits).mean(0)
    in_region = common.in_region_field_metrics(
        args.level, data["fid_test"], np.asarray(test_labels), ens_test.argmax(1).numpy())
    pred_csv = _write_holdout(ens_hold, hold_fids, data["le"], args.level, name)
    print(f"In-region (field level): {in_region}")
    print(f"Holdout predictions: {pred_csv}")

    common.write_metadata(out_dir, {
        "model": "timesnet", "level": args.level, "name": name,
        "protocol": "L-TAE-S harness: focal loss (gamma=2, alpha=1/count), AdamW lr=1e-3 wd=1e-4, "
                    "cosine T_max=100, early stop patience 15 on val macro-F1, AMP, grad clip 1.0, "
                    "5-seed mean logits",
        "tslib_commit": TSLIB_COMMIT, "timesnet_config": TIMESNET_CFG,
        "batch_size": bs, "max_epochs": args.max_epochs, "seeds": args.seeds,
        "n_fields": data["n_fields"], "n_rows": data["n_rows"], "input_checks": checks,
        "per_seed": seed_info,
        "per_seed_seconds_mean": round(float(np.mean([s["train_seconds"] for s in seed_info])), 1),
        "training_time_sec_total": round(float(np.sum([s["train_seconds"] for s in seed_info])), 1),
        "metrics": in_region, "holdout_predictions": os.path.relpath(pred_csv, common.REPO),
    })


def _write_holdout(logits, fids, le, level, stem):
    """Field-level holdout CSV (fid, crop_name) plus per-field class probabilities."""
    probs = torch.softmax(logits.float(), dim=1).numpy()
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


if __name__ == "__main__":
    main()
