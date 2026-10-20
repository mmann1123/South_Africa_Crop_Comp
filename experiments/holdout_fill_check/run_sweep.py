"""Data-efficiency sweep (training-field fractions x subsample draws) scored on a chosen holdout.

Runs the published sweep scripts in experiments/field_reduction/ unchanged
(seed_sweep.py: L-TAE, L-TAE-S; seed_sweep_tabnet.py: pixel TabNet; seed_sweep_classical.py:
field XGBoost, pixel logistic regression), redirecting only the holdout input:

  deep models   MERGED_DL_TEST_PATH -> data/merged_dl_test_trainmedian.parquet (trainmed)
                                       or the published data/merged_dl_test.parquet (orig)
  classical     COMBINED_TEST_FEATURES_PATH -> data/combined_test_features_{variant}.parquet

Everything after the double dash is passed to the sweep script, e.g.
  python run_sweep.py --script seed_sweep --variant trainmed -- --models ltae_pixel \
      --fractions 0.25 0.5 0.75 --subsample-seeds 42 101 202 303 404 --out results/sweep_trainmed.csv
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FR = os.path.join(REPO, "experiments", "field_reduction")
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, FR)

DL = {"orig": os.path.join(REPO, "data", "merged_dl_test.parquet"),
      "trainmed": os.path.join(HERE, "data", "merged_dl_test_trainmedian.parquet")}

if __name__ == "__main__":
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, choices=["seed_sweep", "seed_sweep_tabnet", "seed_sweep_classical"])
    ap.add_argument("--variant", required=True, choices=["orig", "trainmed"])
    a = ap.parse_args(own)
    os.chdir(FR)  # the sweep scripts use paths relative to their folder
    mod = importlib.import_module(a.script)
    if a.script == "seed_sweep_classical":
        mod.COMBINED_TEST_FEATURES_PATH = os.path.join(HERE, "data", f"combined_test_features_{a.variant}.parquet")
    else:
        mod.MERGED_DL_TEST_PATH = DL[a.variant]
    rest = [os.path.abspath(os.path.join(HERE, x)) if x.startswith("results/") else x for x in rest]
    sys.argv = [a.script] + rest
    print(f"[sweep] {a.script} variant={a.variant} args={rest}", flush=True)
    mod.main()
