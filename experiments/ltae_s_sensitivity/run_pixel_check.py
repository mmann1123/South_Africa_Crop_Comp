"""
Reviewer R1-2, pixel-level check: retrain pixel L-TAE-S (5 seeds, full training set,
same harness as the published run) at two settings that bracket the field-level sweep:
the best field setting (gamma 2.0, 4 heads) and the weakest that trained (gamma 1.0,
32 heads). Compared with the published pixel L-TAE-S (gamma 1.5, 16 heads, OOR F1m 0.599).

Resumable and non-destructive: outputs in experiments/ltae_s_sensitivity/pixel/;
results/pixel_check.csv rewritten after each setting.
Run: ~/miniconda3/envs/deep_field/bin/python experiments/ltae_s_sensitivity/run_pixel_check.py
"""
import json
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FR = os.path.join(REPO, "experiments", "field_reduction")
sys.path.insert(0, HERE)
from run_sweep import ground_truth, score  # noqa: E402

SETTINGS = [dict(gamma=2.0, n_head=4), dict(gamma=1.0, n_head=32)]


def main():
    sys.path.insert(0, FR)
    from predict_oos import predict_ltae_sparse_pixel
    gt = ground_truth()
    base = os.path.join(HERE, "pixel")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rows = []
    for c in SETTINGS:
        t = f"pixel_g{c['gamma']}_h{c['n_head']}"
        out = os.path.join(base, t)
        meta_f = os.path.join(out, "metadata.json")
        if not os.path.exists(meta_f):
            t0 = time.time()
            with open(os.path.join(HERE, "logs", f"{t}.log"), "w") as log:
                subprocess.run([sys.executable, "train_ltae_sparse_pixel.py", "--fraction", "1.0",
                                "--output-dir", out, "--gamma", str(c["gamma"]), "--n-head", str(c["n_head"])],
                               cwd=FR, check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"[{t}] trained in {(time.time() - t0) / 60:.0f} min", flush=True)
        csv = os.path.join(base, f"{t}.csv")
        if not os.path.exists(csv):
            predict_ltae_sparse_pixel(out, csv)
        f1, k, acc = score(csv, gt)
        meta = json.load(open(meta_f))
        rows.append({**c, "tag": t, "oor_f1m": f1, "oor_kappa": k, "oor_acc": acc,
                     "inregion_f1m": meta["metrics"]["f1_macro"], "train_sec": meta.get("training_time_sec")})
        pd.DataFrame(rows).round(4).to_csv(os.path.join(HERE, "results", "pixel_check.csv"), index=False)
        print(f"[{t}] OOR F1m {f1:.3f} kappa {k:.3f} in-region {meta['metrics']['f1_macro']:.3f}", flush=True)


if __name__ == "__main__":
    main()
