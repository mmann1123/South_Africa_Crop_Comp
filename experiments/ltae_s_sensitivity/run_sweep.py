"""
Reviewer R1-2: sensitivity of L-TAE-S out-of-region transfer to its sparsity
hyperparameters (gamma, lambda) and the number of attention heads.

Trains field-level L-TAE-S (5 seeds, full training set, same harness and splits as the
published run) for each configuration, predicts the holdout tile, and scores it.

  gamma (relaxation prior)  x  n_head (attention heads, d_k = 8 fixed)
  plus lambda (mask-entropy penalty) at the published gamma / n_head

The published configuration (gamma = 1.5, n_head = 16, no penalty) is re-trained as a
reproducibility check and compared with out_of_sample/predictions_ltae_s_field.csv.

Resumable and non-destructive: every configuration has its own folder under models/;
finished configurations (metadata.json present) are skipped; results/sweep_results.csv is
rewritten after every configuration. Nothing outside experiments/ltae_s_sensitivity/ is
written.

Run: ~/miniconda3/envs/deep_field/bin/python experiments/ltae_s_sensitivity/run_sweep.py [--level field|pixel] [--only TAG ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FR = os.path.join(REPO, "experiments", "field_reduction")
OOS = os.path.join(REPO, "out_of_sample")
PY = sys.executable
SEEDS = [42, 101, 202, 303, 404]

GAMMAS = [1.0, 1.2, 1.5, 2.0, 3.0]
HEADS = [4, 8, 16, 32]
LAMBDAS = [1e-3, 1e-2, 1e-1]


def configs():
    out = [dict(gamma=g, n_head=h, entropy_lambda=0.0) for g in GAMMAS for h in HEADS]
    out += [dict(gamma=1.5, n_head=16, entropy_lambda=l) for l in LAMBDAS]
    return out


def tag(c):
    return f"g{c['gamma']}_h{c['n_head']}_lam{c['entropy_lambda']:g}"


def ground_truth():
    sys.path.insert(0, OOS)
    from compare_predictions import load_ground_truth
    return load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]


def score(csv, gt):
    p = pd.read_csv(csv).drop_duplicates("fid").set_index("fid")["crop_name"]
    m = pd.concat([gt.rename("t"), p.rename("p")], axis=1, join="inner")
    return (f1_score(m.t, m.p, average="macro"), cohen_kappa_score(m.t, m.p), accuracy_score(m.t, m.p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["field", "pixel"], default="field")
    ap.add_argument("--only", nargs="*", help="run only these tags")
    args = ap.parse_args()
    if args.level == "pixel":
        sys.exit("Pixel-level checks use run_pixel_check.py (selected configurations only).")

    sys.path.insert(0, FR)
    from predict_oos import predict_ltae_sparse_field
    gt = ground_truth()
    mdir, pdir, rdir = (os.path.join(HERE, d) for d in ("models", "predictions", "results"))
    for d in (mdir, pdir, rdir, os.path.join(HERE, "logs")):
        os.makedirs(d, exist_ok=True)

    rows = []
    for c in configs():
        t = tag(c)
        if args.only and t not in args.only:
            continue
        out = os.path.join(mdir, t)
        meta_f = os.path.join(out, "metadata.json")
        if not os.path.exists(meta_f):
            t0 = time.time()
            cmd = [PY, "train_ltae_sparse_field.py", "--fraction", "1.0", "--output-dir", out,
                   "--gamma", str(c["gamma"]), "--n-head", str(c["n_head"]),
                   "--entropy-lambda", str(c["entropy_lambda"])]
            with open(os.path.join(HERE, "logs", f"{t}.log"), "w") as log:
                subprocess.run(cmd, cwd=FR, check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"[{t}] trained in {time.time() - t0:.0f}s", flush=True)
        meta = json.load(open(meta_f))
        ens_csv = os.path.join(pdir, f"{t}.csv")
        if not os.path.exists(ens_csv):
            predict_ltae_sparse_field(out, ens_csv)
        f1, k, acc = score(ens_csv, gt)
        seed_f1 = []
        for s in SEEDS:
            sc = os.path.join(pdir, f"{t}_seed{s}.csv")
            if not os.path.exists(sc):
                predict_ltae_sparse_field(out, sc, seeds=[s])
            seed_f1.append(score(sc, gt)[0])
        rows.append({**c, "tag": t, "oor_f1m": f1, "oor_kappa": k, "oor_acc": acc,
                     "seed_f1m_mean": np.mean(seed_f1), "seed_f1m_sd": np.std(seed_f1, ddof=1),
                     "inregion_f1m": meta["metrics"]["f1_macro"], "train_sec": meta.get("training_time_sec")})
        pd.DataFrame(rows).round(4).to_csv(os.path.join(rdir, "sweep_results.csv"), index=False)
        print(f"[{t}] OOR F1m {f1:.3f}  kappa {k:.3f}  in-region {meta['metrics']['f1_macro']:.3f}  "
              f"seed sd {np.std(seed_f1, ddof=1):.3f}", flush=True)

    # Reproducibility check against the published field-level L-TAE-S predictions
    base = os.path.join(pdir, "g1.5_h16_lam0.csv")
    if os.path.exists(base):
        a = pd.read_csv(base).set_index("fid")["crop_name"].sort_index()
        b = pd.read_csv(os.path.join(OOS, "predictions_ltae_s_field.csv")).set_index("fid")["crop_name"].sort_index()
        same = (a == b.reindex(a.index)).mean()
        msg = f"Published-config retrain agrees with published predictions on {same:.1%} of fields"
        print(msg)
        open(os.path.join(rdir, "reproducibility_check.txt"), "w").write(msg + "\n")


if __name__ == "__main__":
    main()
