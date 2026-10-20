"""Sparse (L-TAE-S-style) gate on the top deep backbones: pixel TempCNN and pixel CNN-BiLSTM.

Each backbone is trained with exactly its published recipe (data layout, scaling, split,
seeds, optimizer, batch size, loss, early stopping), with one change: a gate block in front.

Gate block (mirrors L-TAE-S, experiments/field_reduction/models_arch.SparseFeatureGate):
  per time step, Linear(6 -> 128) embedding; 16 heads, each with Linear(128 -> 128) + BatchNorm
  logits times a relaxation prior (gamma = 1.5), turned into a mask over the 128 channels by
  sparsemax (--gate sparsemax) or softmax (--gate softmax, the dense control); each head's
  masked embedding is projected to 8 channels (L-TAE-S's d_k) and the 16 heads are concatenated
  into a 128-channel sequence, which feeds the unchanged backbone (first layer now takes 128
  input channels instead of 6). The gate runs in fp32 (sparsemax's sort and cumulative sum).
  L-TAE-S's constant L1 mask term is omitted: it is constant under sparsemax (reviewer R1-4).

Holdout predictions are written for two holdout files, with the published inference
preprocessing of each backbone:
  orig      data/merged_dl_test.parquet (missing September filled with 0, as published)
  trainmed  experiments/holdout_fill_check/data/merged_dl_test_trainmedian.parquet

Run in the deep_field env:
  python experiments/sparse_gate_backbones/train.py --backbone tempcnn --gate sparsemax
Smoke test: add --seeds 42 --max-epochs 2 --tag smoke
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "field_reduction"))
from config import MERGED_DL_PATH, MERGED_DL_TEST_PATH  # noqa: E402
from models_arch import sparsemax  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
device = torch.device("cuda")
HOLDOUT = {"orig": MERGED_DL_TEST_PATH,
           "trainmed": os.path.join(REPO, "experiments", "holdout_fill_check", "data",
                                    "merged_dl_test_trainmedian.parquet")}
BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
MONTHS = ["January", "February", "March", "April", "July", "August", "September",
          "October", "November", "December"]
T, C = 10, 6
D_MODEL, N_HEAD, D_K, GAMMA = 128, 16, 8, 1.5

# Published recipes (deep_learn/src/Deep Learning/Pixel_Field_Level/{tempcnn_model,cnn_bilstm}.py)
RECIPE = {
    "tempcnn": dict(seeds=[42, 101, 202, 303, 404], val_size=0.2, batch=2048, layout="time_major",
                    scale=True, opt="adamw_cosine", epochs=100, patience=15),
    "cnn_bilstm": dict(seeds=[0, 1, 2, 3, 4], val_size=0.1, batch=1024, layout="band_major",
                       scale=False, opt="adam", epochs=100, patience=15),
}


# ---------------------------------------------------------------- gate
class GateBlock(nn.Module):
    """(B, T, 6) -> (B, T, 128) via 16 masked, projected views of a 128-d embedding."""

    def __init__(self, gate):
        super().__init__()
        self.gate = gate
        self.embed = nn.Linear(C, D_MODEL)
        self.fc = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(N_HEAD)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(N_HEAD)])
        self.proj = nn.ModuleList([nn.Linear(D_MODEL, D_K) for _ in range(N_HEAD)])

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            z = self.embed(x.float())
            B, Tn, E = z.shape
            prior = torch.ones_like(z)
            outs, masks = [], []
            for h in range(N_HEAD):
                logits = self.bn[h](self.fc[h](z.reshape(B * Tn, E))).view(B, Tn, E) * prior
                m = sparsemax(logits, dim=-1) if self.gate == "sparsemax" else F.softmax(logits, dim=-1)
                prior = prior * torch.clamp(GAMMA - m, min=0)
                outs.append(self.proj[h](z * m))
                masks.append(m)
        self.last_masks = torch.stack(masks, 1).detach()  # (B, H, T, E) for sparsity stats
        return torch.cat(outs, dim=-1)


# ---------------------------------------------------------------- backbones (published, in_ch changed)
class TempCNN(nn.Module):
    def __init__(self, in_ch, num_classes, dropout=0.3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 3, padding=1), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(dropout))
        self.conv1, self.conv2, self.conv3 = block(in_ch, 64), block(64, 128), block(128, 256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, num_classes))

    def forward(self, x):  # (B, T, ch)
        x = self.conv3(self.conv2(self.conv1(x.permute(0, 2, 1))))
        return self.head(self.pool(x).squeeze(-1))


class CNNBiLSTM(nn.Module):
    def __init__(self, in_ch, num_classes, conv_filters=64, lstm_hidden=64, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, conv_filters, kernel_size, padding=kernel_size // 2)
        self.bilstm = nn.LSTM(conv_filters, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, x):  # (B, T, ch)
        x = F.relu(self.conv1(x.permute(0, 2, 1))).permute(0, 2, 1)
        x, _ = self.bilstm(x)
        return self.fc(self.dropout(x[:, -1, :]))


class Gated(nn.Module):
    def __init__(self, backbone, gate, num_classes):
        super().__init__()
        self.gate = GateBlock(gate)
        self.net = (TempCNN if backbone == "tempcnn" else CNNBiLSTM)(N_HEAD * D_K, num_classes)

    def forward(self, x):
        return self.net(self.gate(x))


# ---------------------------------------------------------------- data (published preprocessing)
def feature_cols(df, layout):
    """time_major: [B2_Jan..hue_Jan, B2_Feb, ...] (TempCNN); band_major: parquet column order
    filtered to the band columns, i.e. [B2_Jan..B2_Dec, B6_Jan, ...] (CNN-BiLSTM)."""
    if layout == "time_major":
        cols = [f"{b}_{m}" for m in MONTHS for b in BANDS]
    else:
        cols = [c for c in df.columns if any(b in c for b in ["B2_", "B6_", "B11_", "B12_", "hue_", "EVI_"])]
    return [c for c in cols if c in df.columns and not df[c].isna().all()]


def to_seq(X, layout):
    """Flat (N, 60) -> (N, T, C) for the gate."""
    if layout == "time_major":
        return X.reshape(-1, T, C)
    return X.reshape(-1, C, T).transpose(0, 2, 1)


def load_train(r):
    df = pd.read_parquet(MERGED_DL_PATH)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["crop_name"])
    cols = feature_cols(df, r["layout"])
    df[cols] = df[cols].fillna(0)
    fids = df["fid"].unique()
    tr, te = train_test_split(fids, test_size=0.2, random_state=42)
    tr, va = train_test_split(tr, test_size=r["val_size"], random_state=42)
    parts = {k: df[df["fid"].isin(f)] for k, f in [("train", tr), ("val", va), ("test", te)]}
    scaler = StandardScaler().fit(parts["train"][cols].values) if r["scale"] else None
    out = {"le": le, "cols": cols, "scaler": scaler, "test_fid": parts["test"]["fid"].values}
    for k, p in parts.items():
        X = p[cols].values
        if scaler is not None:
            X = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        out[f"X_{k}"] = to_seq(X.astype(np.float32), r["layout"])
        out[f"y_{k}"] = p["label"].values
    return out


def load_holdout(path, d, r):
    df = pd.read_parquet(path)
    for c in d["cols"]:
        if c not in df.columns:
            df[c] = 0
    df[d["cols"]] = df[d["cols"]].fillna(0)  # published inference rule for both backbones
    X = df[d["cols"]].values
    if d["scaler"] is not None:
        X = np.nan_to_num(d["scaler"].transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    return to_seq(X.astype(np.float32), r["layout"]), df["fid"].values


# ---------------------------------------------------------------- train / eval
def loader(X, y, bs, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(np.asarray(y, dtype=np.int64))),
                      batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction="none")
        return (self.alpha[y] * (1 - torch.exp(-ce)) ** self.gamma * ce).mean()


@torch.no_grad()
def logits_of(model, X, bs, stats=False):
    model.eval()
    out, nz = [], []
    for (xb, _) in loader(X, np.zeros(len(X)), bs, False):
        with torch.amp.autocast("cuda"):
            out.append(model(xb.to(device, non_blocking=True)).float().cpu())
        if stats:
            nz.append((model.gate.last_masks > 0).float().sum(-1).mean().item())
    return torch.cat(out), (float(np.mean(nz)) if stats else None)


def field_vote(fids, preds):
    s = pd.DataFrame({"fid": fids, "p": preds}).groupby("fid")["p"].agg(lambda x: Counter(x).most_common(1)[0][0])
    return s.index.values, s.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(RECIPE))
    ap.add_argument("--gate", required=True, choices=["sparsemax", "softmax"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = dict(RECIPE[a.backbone])
    seeds = a.seeds or r["seeds"]
    epochs = a.max_epochs or r["epochs"]
    name = f"{a.backbone}_{a.gate}" + (f"_{a.tag}" if a.tag else "")
    mdir, pdir = os.path.join(HERE, "models", name), os.path.join(HERE, "predictions")
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    d = load_train(r)
    nc = len(d["le"].classes_)
    counts = np.maximum(np.bincount(d["y_train"], minlength=nc).astype(float), 1.0)
    alpha = torch.tensor((1 / counts) / (1 / counts).sum() * nc, dtype=torch.float32)
    holdouts = {k: load_holdout(p, d, r) for k, p in HOLDOUT.items()}
    print(f"=== {name}: train {len(d['y_train'])}, val {len(d['y_val'])}, test {len(d['y_test'])} pixels", flush=True)

    test_logits, hold_logits, info = [], {k: [] for k in holdouts}, []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = Gated(a.backbone, a.gate, nc).to(device)
        if r["opt"] == "adamw_cosine":
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=r["epochs"], eta_min=1e-6)
        else:
            opt, sched = torch.optim.Adam(model.parameters(), lr=1e-3), None
        crit = FocalLoss(alpha).to(device)
        amp = torch.amp.GradScaler("cuda")
        tl = loader(d["X_train"], d["y_train"], r["batch"], True)
        path = os.path.join(mdir, f"seed_{seed}.pt")
        best, wait, t0, ep = -1.0, 0, time.time(), 0
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    loss = crit(model(xb), yb)
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt)
                amp.update()
            vl, _ = logits_of(model, d["X_val"], r["batch"])
            vf1 = f1_score(d["y_val"], vl.argmax(1).numpy(), average="macro")
            if sched:
                sched.step()
            print(f"  seed {seed} epoch {ep}: val F1m {vf1:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if vf1 > best:
                best, wait = vf1, 0
                torch.save(model.state_dict(), path)
            else:
                wait += 1
                if wait >= r["patience"]:
                    break
        model.load_state_dict(torch.load(path, map_location=device))
        tlog, _ = logits_of(model, d["X_test"], r["batch"])
        test_logits.append(tlog)
        nz = None
        for k, (Xh, _) in holdouts.items():
            hlog, s = logits_of(model, Xh, r["batch"], stats=(k == "trainmed"))
            hold_logits[k].append(hlog)
            nz = s if s is not None else nz
        info.append({"seed": seed, "best_val_f1m": round(best, 4), "epochs": ep,
                     "seconds": round(time.time() - t0, 1), "holdout_mean_nonzero_channels_per_head": nz})
        print(f"  seed {seed}: best val F1m {best:.4f}, {ep} epochs, nonzero channels/head {nz}", flush=True)

    ens = torch.stack(test_logits).mean(0).argmax(1).numpy()
    f_ids, f_pred = field_vote(d["test_fid"], ens)
    _, f_true = field_vote(d["test_fid"], d["y_test"])
    in_region = {"f1m": f1_score(f_true, f_pred, average="macro"), "acc": accuracy_score(f_true, f_pred),
                 "kappa": cohen_kappa_score(f_true, f_pred)}
    for k, (_, fids) in holdouts.items():
        p = torch.stack(hold_logits[k]).mean(0).argmax(1).numpy()
        fi, fp = field_vote(fids, p)
        pd.DataFrame({"fid": fi, "crop_name": d["le"].inverse_transform(fp)}).to_csv(
            os.path.join(pdir, f"{name}_{k}.csv"), index=False)
    joblib.dump({"scaler": d["scaler"], "le": d["le"], "cols": d["cols"]}, os.path.join(mdir, "artifacts.joblib"))
    json.dump({"backbone": a.backbone, "gate": a.gate, "recipe": r, "seeds": seeds,
               "in_region_field": {k: round(v, 4) for k, v in in_region.items()}, "per_seed": info},
              open(os.path.join(mdir, "metadata.json"), "w"), indent=2)
    print(f"=== {name} in-region (field): {in_region}", flush=True)


if __name__ == "__main__":
    main()
