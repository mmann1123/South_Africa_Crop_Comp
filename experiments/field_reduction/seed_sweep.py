"""Subsample-seed sweep for the data-efficiency experiment (optimized, GPU-resident).

Re-runs the field-reduction training across MULTIPLE subsample draws per fraction so we
can plot mean +/- band and test whether the per-fraction wiggle (e.g. L-TAE-S's 0.50 dip)
is a single-draw sampling artifact.

Optimization vs the original train_*.py: the full pixel set (~1 GB) lives on the GPU and we
iterate shuffled index slices directly -- no DataLoader, no workers, no per-batch H2D copy.
Everything else (architecture, focal loss, AdamW, cosine schedule, 100 epochs, patience 15,
5-seed ensemble, field-majority-vote OOS) matches models_arch / train_ltae_sparse_pixel.

OOS is scored inline against the holdout labels GeoJSON, so nothing is written to disk except
the results CSV (appended incrementally, so partial progress survives interruption).

Usage:
    python seed_sweep.py --models ltae_sparse_pixel --fractions 0.50 --subsample-seeds 42   # validate
    python seed_sweep.py --models ltae_sparse_pixel ltae_pixel \
        --fractions 0.25 0.50 0.75 1.00 --subsample-seeds 42 101 202 303 404 --out results/seed_sweep.csv
"""
import argparse, copy, json, os, random, sys, time
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiment_config import MERGED_DL_PATH, MERGED_DL_TEST_PATH, SEEDS_ENSEMBLE, TEST_LABELS_GEOJSON
from models_arch import (
    LTAESparse, LTAE, WeightedFocalLoss, compute_focal_loss_weights,
    get_chrono_feature_cols, train_epoch_sparse, evaluate_sparse,
    train_epoch, evaluate, get_device, T_SEQ, N_BANDS,
)
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS, BATCH_SIZE, LR, PATIENCE = 100, 2048, 1e-3, 15
LAMBDA_SPARSE, GAMMA, TIME_VARYING_GATE = 1e-3, 1.5, True


class GPULoader:
    """Drop-in iterable yielding GPU (X, y) batches by slicing GPU-resident tensors."""
    def __init__(self, X, y, batch_size, shuffle, device):
        self.X = torch.as_tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32, device=device)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        self.bs, self.shuffle, self.device = batch_size, shuffle, device
    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs
    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, self.bs):
            j = idx[i:i + self.bs]
            yield self.X[j], self.y[j]


def build_model(model_key, num_classes, device):
    if model_key == "ltae_sparse_pixel":
        return LTAESparse(in_channels=N_BANDS, d_model=128, n_head=16, d_k=8, dropout=0.3,
                          num_classes=num_classes, gamma=GAMMA, time_varying_gate=TIME_VARYING_GATE).to(device)
    elif model_key == "ltae_pixel":
        return LTAE(in_channels=N_BANDS, num_classes=num_classes).to(device)
    raise ValueError(model_key)


def train_one_seed(model_key, seed, train_loader, val_loader, criterion, num_classes, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = build_model(model_key, num_classes, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)
    amp = torch.amp.GradScaler("cuda")
    is_sparse = model_key == "ltae_sparse_pixel"
    best_f1, best_state, patience = -1.0, copy.deepcopy(model.state_dict()), 0
    ep_times = []
    for epoch in range(N_EPOCHS):
        t = time.time()
        if is_sparse:
            train_epoch_sparse(model, opt, criterion, train_loader, amp, device, lambda_sparse=LAMBDA_SPARSE)
            vl, vy = evaluate_sparse(model, val_loader, device)
        else:
            train_epoch(model, opt, criterion, train_loader, amp, device)
            vl, vy = evaluate(model, val_loader, device)
        vf1 = f1_score(vy, vl.argmax(dim=1).tolist(), average="macro")
        sched.step(); ep_times.append(time.time() - t)
        if vf1 > best_f1:
            best_f1, best_state, patience = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, np.mean(ep_times), epoch + 1


@torch.no_grad()
def oos_logits(model_key, model, Xt_gpu, device):
    model.eval()
    is_sparse = model_key == "ltae_sparse_pixel"
    out = []
    with torch.amp.autocast("cuda"):
        for i in range(0, Xt_gpu.shape[0], 8192):
            xb = Xt_gpu[i:i + 8192]
            logits = model(xb)[0] if is_sparse else model(xb)
            out.append(logits.float())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ltae_sparse_pixel"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.50])
    ap.add_argument("--subsample-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", default="results/seed_sweep.csv")
    args = ap.parse_args()

    device = get_device()
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # ---- load once ----
    print("Loading train pixels...")
    df = pd.read_parquet(MERGED_DL_PATH)
    feature_cols = [c for c in get_chrono_feature_cols(df) if not df[c].isna().all()]
    df[feature_cols] = df[feature_cols].fillna(0)
    le = LabelEncoder(); df["label"] = le.fit_transform(df["crop_name"]); num_classes = len(le.classes_)
    train_fids, val_fids, test_fids = get_fid_split_dl(df)
    val_mask = df["fid"].isin(val_fids)
    Xv_raw = df.loc[val_mask, feature_cols].values.astype(np.float32)
    yv = df.loc[val_mask, "label"].values
    print(f"  {df.shape[0]} pixels, {len(feature_cols)} feats, {num_classes} classes")

    print("Loading holdout pixels...")
    dft = pd.read_parquet(MERGED_DL_TEST_PATH)
    for c in feature_cols:
        if c not in dft.columns:
            dft[c] = 0
    dft[feature_cols] = dft[feature_cols].fillna(0)
    Xtest_raw = dft[feature_cols].values.astype(np.float32)
    test_fids_arr = dft["fid"].values

    import json as _json
    gt = {f["properties"]["fid"]: f["properties"]["crop_name"]
          for f in _json.load(open(TEST_LABELS_GEOJSON))["features"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write("model,fraction,subsample_seed,oos_f1_macro,oos_kappa,n_seeds,mean_epoch_s,epochs_run\n")

    for model_key in args.models:
        for frac in args.fractions:
            for sseed in args.subsample_seeds:
                t0 = time.time()
                sub = subsample_train_fids(df, train_fids, frac, seed=sseed)
                tmask = df["fid"].isin(sub)
                scaler = StandardScaler()
                Xtr = np.nan_to_num(scaler.fit_transform(df.loc[tmask, feature_cols].values).astype(np.float32))
                ytr = df.loc[tmask, "label"].values
                Xvv = np.nan_to_num(scaler.transform(Xv_raw).astype(np.float32))
                train_loader = GPULoader(Xtr, ytr, BATCH_SIZE, True, device)
                val_loader = GPULoader(Xvv, yv, BATCH_SIZE, False, device)
                criterion = WeightedFocalLoss(compute_focal_loss_weights(ytr, num_classes), gamma=2.0).to(device)
                # holdout scaled with THIS cell's scaler, GPU-resident
                Xt_gpu = torch.as_tensor(
                    np.nan_to_num(scaler.transform(Xtest_raw).astype(np.float32)).reshape(-1, T_SEQ, N_BANDS),
                    dtype=torch.float32, device=device)

                print(f"\n=== {model_key} frac={frac} sseed={sseed} | train fids {tmask.sum()} px ===")
                ens = None
                for k, seed in enumerate(SEEDS_ENSEMBLE):
                    model, mean_ep, n_ep = train_one_seed(model_key, seed, train_loader, val_loader,
                                                          criterion, num_classes, device)
                    lg = oos_logits(model_key, model, Xt_gpu, device)
                    ens = lg if ens is None else ens + lg
                    if k == 0:
                        print(f"    seed {seed}: {mean_ep:.2f}s/epoch x {n_ep} epochs")
                ens /= len(SEEDS_ENSEMBLE)
                px_pred = ens.argmax(dim=1).cpu().numpy()
                pred_df = pd.DataFrame({"fid": test_fids_arr, "pred": px_pred})
                field_pred = pred_df.groupby("fid")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])
                fl = pd.DataFrame({"pred": le.inverse_transform(field_pred.values)}, index=field_pred.index)
                fl["true"] = [gt.get(i) for i in fl.index]
                fl = fl.dropna()
                f1m = f1_score(fl["true"], fl["pred"], average="macro")
                kap = cohen_kappa_score(fl["true"], fl["pred"])
                dt = time.time() - t0
                print(f"    -> OOS macro-F1={f1m:.4f}, kappa={kap:.4f}  ({dt/60:.1f} min for the 5-seed cell)")
                with open(args.out, "a") as f:
                    f.write(f"{model_key},{frac},{sseed},{f1m:.4f},{kap:.4f},{len(SEEDS_ENSEMBLE)},{mean_ep:.2f},{n_ep}\n")
                del Xt_gpu; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
