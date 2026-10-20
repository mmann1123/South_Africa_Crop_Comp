"""Data-efficiency sweep for pixel TempCNN and CNN-BiLSTM, dense and sparse-gated.

Same design as experiments/field_reduction/seed_sweep.py (and the holdout-fill sweep):
training fields subsampled per (fraction, draw) with field_reduction.subsample.subsample_train_fids
(stratified by crop, from the backbone's own training split), one training seed, holdout scored
on the September-filled file (experiments/holdout_fill_check/data/merged_dl_test_trainmedian.parquet).
Each backbone keeps its published recipe (train.py: RECIPE); 'dense' is the published architecture
(6 input channels, no gate), 'sparsemax' adds the L-TAE-S-style gate block. For a given backbone,
dense and gated runs use identical subsamples. When TempCNN is scaled, the scaler is fit on the
subsample, as in seed_sweep.py.

Data are loaded once per process; finished (model, fraction, draw) rows in --out are skipped.
Run in the deep_field env:
  python experiments/sparse_gate_backbones/sweep_backbones.py --backbone tempcnn --variants dense sparsemax \
      --fractions 0.25 0.5 0.75 --draws 42 101 202 303 404 --out results_sweep.csv
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train as tr  # noqa: E402  (recipes, gate, backbones, helpers)
from subsample import subsample_train_fids  # noqa: E402  (experiments/field_reduction, via train.py's sys.path)

sys.path.insert(0, os.path.join(tr.REPO, "out_of_sample"))
from compare_predictions import load_ground_truth  # noqa: E402
sys.stdout.reconfigure(line_buffering=True)


def build(backbone, variant, nc):
    if variant == "dense":
        return (tr.TempCNN if backbone == "tempcnn" else tr.CNNBiLSTM)(tr.C, nc)
    return tr.Gated(backbone, variant, nc)


def run_one(backbone, variant, r, seed, Xtr, ytr, Xva, yva, nc):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = build(backbone, variant, nc).to(tr.device)
    if r["opt"] == "adamw_cosine":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=r["epochs"], eta_min=1e-6)
    else:
        opt, sched = torch.optim.Adam(model.parameters(), lr=1e-3), None
    counts = np.maximum(np.bincount(ytr, minlength=nc).astype(float), 1.0)
    alpha = torch.tensor((1 / counts) / (1 / counts).sum() * nc, dtype=torch.float32)
    crit = tr.FocalLoss(alpha).to(tr.device)
    amp = torch.amp.GradScaler("cuda")
    tl = tr.loader(Xtr, ytr, r["batch"], True)
    best, wait, state, ep, t0 = -1.0, 0, None, 0, time.time()
    for ep in range(1, r["epochs"] + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(tr.device, non_blocking=True), yb.to(tr.device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(xb), yb)
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(opt)
            amp.update()
        vl, _ = tr.logits_of(model, Xva, r["batch"])
        vf1 = f1_score(yva, vl.argmax(1).numpy(), average="macro")
        if sched:
            sched.step()
        if vf1 > best:
            best, wait = vf1, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= r["patience"]:
                break
    model.load_state_dict(state)
    return model, ep, (time.time() - t0) / ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(tr.RECIPE))
    ap.add_argument("--variants", nargs="+", default=["dense", "sparsemax"], choices=["dense", "sparsemax", "softmax"])
    ap.add_argument("--fractions", nargs="+", type=float, required=True)
    ap.add_argument("--draws", nargs="+", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results_sweep.csv"))
    a = ap.parse_args()
    r = tr.RECIPE[a.backbone]
    seed = r["seeds"][0]
    torch.backends.cudnn.benchmark = True

    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out)
        done = set(zip(d["model"], d["fraction"].round(4), d["subsample_seed"]))
    else:
        with open(a.out, "w") as f:
            f.write("model,fraction,subsample_seed,oos_f1_macro,oos_kappa,train_seed,epochs_run,mean_epoch_s,n_train_px\n")

    df = pd.read_parquet(tr.MERGED_DL_PATH)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["crop_name"])
    nc = len(le.classes_)
    cols = tr.feature_cols(df, r["layout"])
    df[cols] = df[cols].fillna(0)
    fids = df["fid"].unique()
    trf, _ = train_test_split(fids, test_size=0.2, random_state=42)
    trf, vaf = train_test_split(trf, test_size=r["val_size"], random_state=42)
    va = df[df["fid"].isin(vaf)]
    hold = pd.read_parquet(tr.HOLDOUT["trainmed"])
    for c in cols:
        if c not in hold.columns:
            hold[c] = 0
    hold[cols] = hold[cols].fillna(0)
    Xh_raw, hfids = hold[cols].values.astype(np.float32), hold["fid"].values
    del hold
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    print(f"=== {a.backbone}: {len(df)} train-tile pixels, holdout {len(Xh_raw)} pixels", flush=True)

    for frac in a.fractions:
        for draw in a.draws:
            todo = [v for v in a.variants if (f"{a.backbone}_{v}", round(frac, 4), draw) not in done]
            if not todo:
                continue
            sub = subsample_train_fids(df, trf, frac, seed=draw)
            t = df[df["fid"].isin(sub)]
            Xtr, Xva, Xh = t[cols].values.astype(np.float32), va[cols].values.astype(np.float32), Xh_raw
            if r["scale"]:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xva, Xh = (np.nan_to_num(sc.transform(x), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                                for x in (Xtr, Xva, Xh))
            Xtr, Xva, Xh = (tr.to_seq(x, r["layout"]) for x in (Xtr, Xva, Xh))
            for v in todo:
                model, ep, sec = run_one(a.backbone, v, r, seed, Xtr, t["label"].values, Xva, va["label"].values, nc)
                hl, _ = tr.logits_of(model, Xh, r["batch"])
                fi, fp = tr.field_vote(hfids, hl.argmax(1).numpy())
                pred = pd.Series(le.inverse_transform(fp), index=fi)
                j = pd.concat([gt.rename("t"), pred.rename("p")], axis=1, join="inner")
                f1m = f1_score(j["t"], j["p"], average="macro")
                kap = cohen_kappa_score(j["t"], j["p"])
                with open(a.out, "a") as f:
                    f.write(f"{a.backbone}_{v},{frac},{draw},{f1m:.4f},{kap:.4f},{seed},{ep},{sec:.1f},{len(t)}\n")
                print(f"  {a.backbone}_{v} frac={frac} draw={draw}: OOS F1m {f1m:.4f} ({ep} epochs, {sec:.0f}s/epoch)", flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
