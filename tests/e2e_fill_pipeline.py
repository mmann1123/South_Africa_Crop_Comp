"""End-to-end check of the HOLDOUT_FILL switch through real model inference (GPU, minutes).

Runs three refactored inference paths with HOLDOUT_FILL=published and HOLDOUT_FILL=train_median
and requires the field predictions to equal the saved re-analysis outputs in
experiments/holdout_fill_check/predictions/ ({key}_orig.csv and {key}_trainmed.csv):

  ltae          out_of_sample/inference_ltae.py           (published rule: zero)
  tabnet        out_of_sample/inference_tabnet.py         (published rule: holdout median)
  ltae_s_field  field_reduction/predict_oos.py            (published rule: zero)

Not part of the unittest suite (slow, needs checkpoints). Run in the deep_field env:
  python tests/e2e_fill_pipeline.py
Exit code 0 only if all six comparisons are identical.
"""
import importlib
import os
import subprocess
import sys
import tempfile

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REF = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
CASES = ["ltae", "tabnet", "ltae_s_field"]

CHILD = r'''
import importlib, os, sys
repo, key, out = sys.argv[1:4]
for p in ["deep_learn/src", "out_of_sample", "experiments/field_reduction", "experiments/holdout_fill_check"]:
    sys.path.insert(0, os.path.join(repo, p))
import run_deep_field as r  # tabnet_shim: loader for the February TabNet checkpoints
r.tabnet_shim()
if key in ("ltae", "tabnet"):
    m = importlib.import_module("inference_" + key)
    m.OUTPUT_CSV = out
    m.main()
else:
    import predict_oos as po
    fn, d = r.PO[key]
    getattr(po, fn)(d, out)
'''


def run(key, fill, out):
    env = dict(os.environ, HOLDOUT_FILL=fill)
    subprocess.run([sys.executable, "-c", CHILD, REPO, key, out], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def same(a, b):
    x = pd.read_csv(a).set_index("fid")["crop_name"].sort_index()
    y = pd.read_csv(b).set_index("fid")["crop_name"].sort_index()
    return x.index.equals(y.index) and (x == y).all(), int((x != y.reindex(x.index)).sum())


if __name__ == "__main__":
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for key in CASES:
            for fill, ref in [("published", "orig"), ("train_median", "trainmed")]:
                out = os.path.join(d, f"{key}_{fill}.csv")
                run(key, fill, out)
                eq, ndiff = same(out, os.path.join(REF, f"{key}_{ref}.csv"))
                ok &= eq
                print(f"{key:14s} HOLDOUT_FILL={fill:13s} vs {key}_{ref}.csv: "
                      f"{'IDENTICAL' if eq else f'{ndiff} fields differ'}", flush=True)
    print("ALL IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
