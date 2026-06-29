"""TabNet arm of the subsample-seed variance sweep (the robust control).

TabNet is the pytorch_tabnet library (its own fit loop), so the GPU-resident/compile tricks
used for the L-TAE family don't apply. We replicate train_tabnet_pixel.py's preprocessing and
predict_oos.py's holdout inference exactly, but: (a) single training seed (vary only the
subsample seed), and (b) a larger batch/virtual-batch to keep it tractable. TabNet's draw
robustness comes from its sparsemax feature masks, not the batch size, so the band shape (the
point of the control) is preserved; the absolute level shifts slightly from the published
batch-1024 curve, which we note. Rows append to the same results CSV the L-TAE family uses.

Usage:
    python seed_sweep_tabnet.py --fractions 0.50 0.75 0.25 --subsample-seeds 42 101 202 303 404 \
        --batch-size 4096 --virtual-batch-size 256 --out results/seed_sweep_var.csv
    python seed_sweep_tabnet.py --fractions 0.50 --subsample-seeds 42 --n-test 1   # timing probe
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score
from pytorch_tabnet import TabNetClassifier  # 4.6.0 path (pytorch_tabnet.tab_model was removed)

from experiment_config import MERGED_DL_PATH, MERGED_DL_TEST_PATH, TEST_LABELS_GEOJSON
from models_arch import F1MacroMetric
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)
TRAIN_SEED = 42
CROP_ID_TO_NAME = {1: "Wheat", 2: "Barley", 3: "Canola", 4: "Lucerne/Medics", 5: "Small grain grazing"}


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, input, target, weight=None):  # 4.6.0 passes weight= into loss_fn; ignore it
        alpha = self.alpha.to(input.device)
        ce = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce)
        return (alpha[target] * ((1 - pt) ** self.gamma) * ce).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.50, 0.75, 0.25])
    ap.add_argument("--subsample-seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--virtual-batch-size", type=int, default=256)
    ap.add_argument("--out", default="results/seed_sweep_var.csv")
    args = ap.parse_args()
    print(f"TabNet sweep: batch={args.batch_size}, vbs={args.virtual_batch_size}, train_seed={TRAIN_SEED}")

    # ---- load + preprocess train pixels once (mirrors train_tabnet_pixel.py) ----
    print("Loading train pixels...")
    df = pd.read_parquet(MERGED_DL_PATH).drop(columns=["May"], errors="ignore")
    exclude = {"id", "point", "fid", "crop_id", "crop_name"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median()).fillna(0)
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"])
    one_hot = [c for c in df.columns if c.startswith("Type_")]
    feature_columns = numeric_cols + one_hot
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    features = df[feature_columns].astype(np.float32)
    le = LabelEncoder(); df["crop_label"] = le.fit_transform(df["crop_id"])
    targets = df["crop_label"].values
    train_fids, val_fids, test_fids = get_fid_split_dl(df)
    val_mask = df["fid"].isin(val_fids)
    X_val, y_val = features[val_mask].values, targets[val_mask]

    # ---- holdout, preprocessed once (mirrors predict_oos.predict_tabnet) ----
    print("Loading holdout pixels...")
    dt = pd.read_parquet(MERGED_DL_TEST_PATH)
    if "Type" in dt.columns:
        dt = pd.get_dummies(dt, columns=["Type"])
    for c in feature_columns:
        if c not in dt.columns:
            dt[c] = 0
    present_numeric = [c for c in numeric_cols if c in dt.columns]
    dt[present_numeric] = dt[present_numeric].fillna(dt[present_numeric].median()).fillna(0)
    dt[numeric_cols] = scaler.transform(dt[numeric_cols])
    dt[numeric_cols] = dt[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    Xtest = dt[feature_columns].astype(np.float32).values
    test_fids_arr = dt["fid"].values
    gt = {f["properties"]["fid"]: f["properties"]["crop_name"]
          for f in json.load(open(TEST_LABELS_GEOJSON))["features"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write("model,fraction,subsample_seed,oos_f1_macro,oos_kappa,n_seeds,mean_epoch_s,epochs_run\n")

    for frac in args.fractions:
        for sseed in args.subsample_seeds:
            t0 = time.time()
            sub = subsample_train_fids(df, train_fids, frac, seed=sseed)
            tmask = df["fid"].isin(sub)
            X_train, y_train = features[tmask].values, targets[tmask]
            cc = np.maximum(np.bincount(y_train, minlength=len(le.classes_)).astype(np.float64), 1.0)
            alpha = torch.tensor((1.0 / cc) / (1.0 / cc).sum() * len(le.classes_), dtype=torch.float32)
            print(f"\n=== tabnet_pixel frac={frac} sseed={sseed} | train {tmask.sum()} px ===")
            model = TabNetClassifier(
                n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
                optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-3),
                scheduler_params={"step_size": 10, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR, seed=TRAIN_SEED, verbose=0)
            model.fit(X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)],
                      eval_metric=[F1MacroMetric], loss_fn=WeightedFocalLoss(alpha, gamma=2.0),
                      max_epochs=100, patience=10, batch_size=args.batch_size,
                      virtual_batch_size=args.virtual_batch_size, num_workers=0, drop_last=False)
            n_ep = len(model.history["loss"])
            # OOS: pixel proba -> argmax -> crop_id -> crop_name -> field majority vote
            proba = model.predict_proba(Xtest)
            px_pred = np.argmax(proba, axis=1)
            crop_ids = le.inverse_transform(px_pred)
            names = np.array([CROP_ID_TO_NAME[c] for c in crop_ids])
            pdf = pd.DataFrame({"fid": test_fids_arr, "pred": names})
            field = pdf.groupby("fid")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])
            fl = pd.DataFrame({"pred": field.values}, index=field.index)
            fl["true"] = [gt.get(i) for i in fl.index]; fl = fl.dropna()
            f1m = f1_score(fl["true"], fl["pred"], average="macro")
            kap = cohen_kappa_score(fl["true"], fl["pred"])
            dt_min = (time.time() - t0) / 60
            mean_ep = (time.time() - t0) / max(n_ep, 1)
            print(f"    -> OOS macro-F1={f1m:.4f}, kappa={kap:.4f}  ({n_ep} epochs, {dt_min:.1f} min)")
            with open(args.out, "a") as f:
                f.write(f"tabnet_pixel,{frac},{sseed},{f1m:.4f},{kap:.4f},1,{mean_ep:.2f},{n_ep}\n")


if __name__ == "__main__":
    main()
