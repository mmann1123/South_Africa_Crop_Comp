"""Re-score every head of the MiniRocket gate experiment (supplement Table S-I) on the
training-median-filled holdout, from the saved heads and the frozen per-seed MiniRocket models.

No retraining. HOLDOUT_FILL=train_median is set before any import, so common.load_holdout fills
September from training-tile medians instead of 0. For each level and seed the holdout features
are transformed once and every head is applied to the same batch: the L2 logistic reference
(../models/minirocket_<level>), L1 logistic, canonical ridge, the sparsemax gate (MiniRocket-S)
and the softmax gate. Seeds are combined as in the training scripts (mean probability for the
linear heads, mean decision scores for ridge, mean logits for the gates; pixel level then a
per-field majority vote). Predictions: results/predictions/<name>_trainmed.csv; table:
results/filled_rescoring.csv with the published zero-fill macro-F1 from summary.csv alongside.
Run in the ts_baselines env (pixel level takes about 20 min per seed on the CPU transform):
  python experiments/recent_baselines/minirocket_sparsity/rescore_filled.py [--levels field pixel]
"""
import argparse
import os
import sys

os.environ["HOLDOUT_FILL"] = "train_median"

import features as fx  # noqa: E402  (sets sys.path: parent experiment, field_reduction)
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from models_arch import evaluate_sparse, get_device  # noqa: E402
from sparse_head import GatedHead, L1Head  # noqa: E402
import train_minirocket as tm  # noqa: E402
import train_gated as tg  # noqa: E402
import train_ridge as tr  # noqa: E402
from score_oos import score  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
PUB_NAMES = {"minirocket": "MiniRocket", "minirocket_l1": "MiniRocket-L1", "minirocket_ridge": "MiniRocket-ridge",
             "minirocket_s": "MiniRocket-S", "minirocket_softmax": "MiniRocket-softmax"}


def heads_for(level, seed, d, K, device):
    m, _ = fx.load_seed_model(level, seed)
    l2 = tm.LogRegHead(m["C"])
    l2.W, l2.b = torch.from_numpy(m["W"]), torch.from_numpy(m["b"])
    z = np.load(os.path.join(fx.MODELS_DIR, f"minirocket_l1_{level}", f"l1_head_seed_{seed}.npz"))
    l1 = L1Head(float(z["C"]))
    l1.W, l1.b = torch.from_numpy(z["W"]), torch.from_numpy(z["b"])
    r = np.load(os.path.join(fx.MODELS_DIR, f"minirocket_ridge_{level}", f"ridge_seed_{seed}.npz"))
    ridge = (r["coef"].astype(np.float32), r["intercept"].astype(np.float32))
    gates = {}
    for gate, name in (("sparsemax", "minirocket_s"), ("softmax", "minirocket_softmax")):
        net = GatedHead(d, K, gate=gate, **tg.HEAD_CFG).to(device)
        ck = os.path.join(fx.MODELS_DIR, f"{name}_{level}", f"gated_{gate}_{level}_seed_{seed}.pt")
        net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        net.eval()
        gates[name] = net
    return m, l2, l1, ridge, gates


def run_level(level, seeds, device):
    data = common.load_training_data(level)
    le, K = data["le"], len(data["le"].classes_)
    X_hold, fids = common.load_holdout(level, data["feature_cols"], data["scaler"])
    d = len(fx.load_seed_model(level, seeds[0])[0]["feature_scaler"].mean_)
    bs = tg.BATCH_SIZE[level]
    acc = {k: [] for k in PUB_NAMES}
    for seed in seeds:
        m, l2, l1, (coef, icpt), gates = heads_for(level, seed, d, K, device)
        out = {k: [] for k in PUB_NAMES}
        for F in fx.iter_features(m, X_hold):
            out["minirocket"].append(l2.predict_proba(F))
            out["minirocket_l1"].append(l1.predict_proba(F))
            out["minirocket_ridge"].append(F @ coef.T + icpt)
            F16 = F.astype(np.float16)
            for name, net in gates.items():
                lg, _ = evaluate_sparse(net, tg.loader(F16, np.zeros(len(F)), bs), device)
                out[name].append(lg.numpy())
        for k in out:
            acc[k].append(np.concatenate(out[k]))
        print(f"  {level} seed {seed} done", flush=True)
    rows = []
    for k, arrs in acc.items():
        ens = np.mean(arrs, 0)
        stem = f"{k}_{level}_trainmed"
        if k == "minirocket_ridge":
            csv = tr.write_holdout(ens, fids, le, level, stem)
        elif k in ("minirocket", "minirocket_l1"):
            csv = tg.write_holdout(np.log(np.clip(ens, 1e-30, None)), fids, le, level, stem)
        else:
            csv = tg.write_holdout(ens, fids, le, level, stem)
        s = score(csv)
        rows.append({"model": f"{PUB_NAMES[k]} ({level})", "key": k, "level": level, "f1_macro": s["f1_macro"],
                     "kappa": s["kappa"], "xent": s["xent"], "predictions": os.path.relpath(csv, common.REPO)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["field", "pixel"])
    ap.add_argument("--seeds", type=int, nargs="+", default=common.SEEDS_ENSEMBLE)
    a = ap.parse_args()
    torch.set_num_threads(tm.TORCH_THREADS)
    device = get_device()
    pub = pd.read_csv(os.path.join(fx.RESULTS_DIR, "summary.csv")).set_index("model")["oor_f1m"]
    out = os.path.join(fx.RESULTS_DIR, "filled_rescoring.csv")
    rows = []
    for level in a.levels:
        rows += run_level(level, a.seeds, device)
        df = pd.DataFrame(rows)
        df["f1_macro_published"] = df["model"].map(pub)
        df["delta"] = df["f1_macro"] - df["f1_macro_published"]
        df.to_csv(out, index=False)
        print(df.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
