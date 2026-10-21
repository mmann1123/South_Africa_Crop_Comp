"""Re-run holdout inference for the xr_fresh classical models under the two holdout
feature sets rebuilt by rebuild_xrfresh.py / combine_xrfresh.py:

  orig      features rebuilt from the raw holdout series (September missing for ~41% of pixels)
  trainmed  features rebuilt after filling September with training-tile medians

The published feature files were built from an older pixel grid for some fields (85 of 2,415
fields differ, mostly B12), so 'orig' here is the like-for-like baseline for 'trainmed';
score.py reports how far 'orig' sits from the published predictions.

The published inference code is used unchanged; only the input and output paths are
redirected. Voting and Stacking retrain inside their scripts (fixed seeds). Writes only to
experiments/holdout_fill_check/predictions/. Run in the ml_field env:
  ~/miniconda3/envs/ml_field/bin/python experiments/holdout_fill_check/run_classical.py
"""
import argparse
import importlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in [os.path.join(REPO, "deep_learn", "src"), os.path.join(REPO, "out_of_sample"),
          os.path.join(REPO, "experiments", "field_reduction")]:
    sys.path.insert(0, p)

PRED = os.path.join(HERE, "predictions")
VARIANTS = ["orig", "trainmed"]


def combined(v):
    return os.path.join(HERE, "data", f"combined_test_features_{v}.parquet")


def run(key, v):
    if key in ("xgboost", "lgbm", "smote_stacked"):
        mod = importlib.import_module(f"inference_{key}")
        mod.TEST_PARQUET = combined(v)
        mod.OUTPUT_CSV = os.path.join(PRED, f"{key}_{v}.csv")
        mod.main()
    elif key == "base_ml":  # writes predictions_{base_lr,base_rf,base_lgbm,base_xgb}.csv into SCRIPT_DIR
        mod = importlib.import_module("inference_base_ml")
        tmp = os.path.join(PRED, f"_base_ml_{v}")
        os.makedirs(tmp, exist_ok=True)
        mod.TEST_PARQUET, mod.SCRIPT_DIR = combined(v), tmp
        mod.main()
        for short in mod.MODEL_FILES:
            shutil.move(os.path.join(tmp, f"predictions_{short}.csv"), os.path.join(PRED, f"{short}_{v}.csv"))
        os.rmdir(tmp)
    elif key == "stacking":
        mod = importlib.import_module("stacking_consistent_oos")
        mod.COMBINED_TEST_FEATURES_PATH = combined(v)
        mod.OUT_CSV = os.path.join(PRED, f"stacking_{v}.csv")
        mod.main()
    elif key == "voting":
        mod = importlib.import_module("voting_hybrid_oos")
        mod.FEAT = os.path.join(HERE, "data", f"xtest_{v}")
        mod.OUT_CSV = os.path.join(PRED, f"voting_{v}.csv")
        mod.main()
    print(f"[done] {key} {v}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgboost", "lgbm", "smote_stacked", "base_ml", "stacking", "voting"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--pred-dir", default=PRED, help="output folder (default: this experiment's predictions/)")
    args = ap.parse_args()
    PRED = args.pred_dir
    os.makedirs(PRED, exist_ok=True)
    for k in args.models:
        for v in args.variants:
            run(k, v)
