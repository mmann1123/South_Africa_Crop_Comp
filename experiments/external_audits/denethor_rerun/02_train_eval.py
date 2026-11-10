#!/usr/bin/env python
"""Retrain DENETHOR's PseLTae on Planet Fusion daily data (18E-242N 2018) and
test it on 17E-243N 2019, for conditions A0, A1 and A2 (pflib.py) and seeds.

Recipe from train_psetae.py / test_psetae.py and the paper: Adam lr 1e-3,
weight decay 1e-6, cross-entropy, batch size 64 and 100 epochs (paper text;
the script defaults 4 and 150 contradict it), 25 % validation split with
RandomState(0), checkpoint kept when validation loss beats every earlier epoch
from the third epoch on (the script's rule), test on that checkpoint with one
stochastic pass (the authors' test) and, as a stability check, the mean of
ten passes. cudnn is left enabled (the authors disabled it to avoid a GTX 970
error; it changes speed, not the model). One GPU process at a time.

Output per run: results/runs/<cond>_seed<k>/trainlog.csv, test_predictions.csv
(fid, true, predicted single pass, predicted by ten-pass mean log-probability),
and results/train_eval_summary.csv across runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.denethor_rerun import pflib  # noqa: E402

PAPER = "denethor_rerun"
TRAIN = ("18E-242N", 2018)
TEST = ("17E-243N", 2019)


def npz(tile_year) -> Path:
    return paths.paper_root(PAPER) / f"pf_{tile_year[0]}_{tile_year[1]}.npz"


def run_epoch(model, data, idx, gen, opt=None, bs=64):
    train = opt is not None
    model.train(train)
    crit = torch.nn.CrossEntropyLoss()
    losses, yt, yp = [], [], []
    for i in range(0, len(idx), bs):
        b = torch.as_tensor(idx[i:i + bs], device=data.device)
        x, y = data.batch(b, gen)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = crit(out, y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
        yt.append(y.cpu())
        yp.append(out.argmax(1).detach().cpu())
    return float(np.mean(losses)), torch.cat(yt).numpy(), torch.cat(yp).numpy()


def scores(y, p) -> dict:
    return {"accuracy": skm.accuracy_score(y, p), "f1_macro": skm.f1_score(y, p, average="macro", zero_division=0),
            "kappa": skm.cohen_kappa_score(y, p), "f1_weighted": skm.f1_score(y, p, average="weighted", zero_division=0)}


def one_run(cond: str, seed: int, epochs: int, bs: int, force: bool) -> dict:
    out = paths.results_dir(PAPER) / "runs" / f"{cond}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "summary.csv"
    if done.exists() and not force:
        return pd.read_csv(done).iloc[0].to_dict()
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr = pflib.FieldTensor(npz(TRAIN), cond)
    te = pflib.FieldTensor(npz(TEST), cond)
    tri, vai = pflib.authors_split(len(tr))
    model = pflib.build_model(5 if cond in ("A1", "A2") else 4).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    ckpt = paths.paper_root(PAPER) / "models" / f"{cond}_seed{seed}_best.pt"
    ckpt.parent.mkdir(exist_ok=True)
    log, best, t0 = [], None, time.time()
    for ep in range(1, epochs + 1):
        # the authors' DataLoader does not shuffle (shuffle=False default); kept as is
        trl, _, _ = run_epoch(model, tr, tri, gen, opt, bs)
        val, yv, pv = run_epoch(model, tr, vai, gen, None, bs)
        s = scores(yv, pv)
        log.append({"epoch": ep, "trainloss": trl, "valloss": val, **{f"val_{k}": v for k, v in s.items()}})
        if len(log) > 2 and val < min(r["valloss"] for r in log[:-1]):
            torch.save(model.state_dict(), ckpt)
            best = ep
    if best is None:
        torch.save(model.state_dict(), ckpt)
        best = epochs
    pd.DataFrame(log).to_csv(out / "trainlog.csv", index=False)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    tei = np.arange(len(te))
    _, yt, p1 = run_epoch(model, te, tei, torch.Generator().manual_seed(1000 + seed), None, bs)
    _, yvb, pvb = run_epoch(model, tr, vai, torch.Generator().manual_seed(2000 + seed), None, bs)
    logps = []
    g10 = torch.Generator().manual_seed(3000 + seed)
    with torch.no_grad():
        for _ in range(10):
            lp = []
            for i in range(0, len(tei), bs):
                x, _ = te.batch(torch.as_tensor(tei[i:i + bs], device=te.device), g10)
                lp.append(torch.log_softmax(model(x), 1).cpu())
            logps.append(torch.cat(lp))
    p10 = torch.stack(logps).mean(0).argmax(1).numpy()
    pd.DataFrame({"fid": te.fid, "true": yt, "pred_single": p1, "pred_mean10": p10}).to_csv(out / "test_predictions.csv", index=False)
    row = {"cond": cond, "seed": seed, "best_epoch": best, "minutes": round((time.time() - t0) / 60, 1),
           **{f"val_{k}": v for k, v in scores(yvb, pvb).items()},
           **{f"test_{k}": v for k, v in scores(yt, p1).items()},
           **{f"test10_{k}": v for k, v in scores(yt, p10).items()}}
    pd.DataFrame([row]).to_csv(done, index=False)
    print(f"{cond} seed {seed}: best epoch {best}, val acc {row['val_accuracy']:.4f}, test acc {row['test_accuracy']:.4f} "
          f"(10-pass {row['test10_accuracy']:.4f}), test F1m {row['test_f1_macro']:.4f}, {row['minutes']} min", flush=True)
    return row


def main(conds: list[str], seeds: list[int], epochs: int, bs: int, force: bool) -> None:
    with RunLog("02_train_eval", {"conds": conds, "seeds": seeds, "epochs": epochs, "batch_size": bs}, paper=PAPER, seed=seeds[0]) as log:
        rows = [one_run(c, s, epochs, bs, force) for s in seeds for c in conds]
        allrows = [pd.read_csv(p).iloc[0].to_dict() for p in sorted((paths.results_dir(PAPER) / "runs").glob("*/summary.csv"))]
        write_csv(pd.DataFrame(allrows), paths.results_dir(PAPER) / "train_eval_summary.csv", "02_train_eval", [str(npz(TRAIN)), str(npz(TEST))])
        log.count("runs_this_call", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=["A0", "A1", "A2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
