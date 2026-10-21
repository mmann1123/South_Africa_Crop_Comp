"""Fill-value sweep: holdout predictions of every raw-input model under each HOLDOUT_FILL method.

No retraining. Each model's published inference code runs with the missing holdout values
(September for ~41% of pixels; May everywhere) filled by deep_learn/src/missing.py:

  zero          0 (the published rule for the raw-sequence models)
  self_median   the holdout's own column medians, then 0 (the published pixel-TabNet rule)
  train_mean    training-tile column means
  train_median  training-tile column medians
  interp        linear interpolation between neighbouring months of the same band

Predictions already produced by experiments/holdout_fill_check are reused where they are the
same computation (verified by tests/e2e_fill_pipeline.py): {key}_orig.csv is the published
rule (zero, or self_median for pixel TabNet) and {key}_trainmed.csv is train_median. Pass
--no-reuse to recompute them.

Writes experiments/missingness/fill_sweep/predictions/{key}_{method}.csv; finished files are
skipped. Two environments:
  deep_field:    python experiments/missingness/fill_sweep.py --group deep
  ts_baselines:  python experiments/missingness/fill_sweep.py --group ts
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FILL_CHECK = os.path.join(REPO, "experiments", "holdout_fill_check")
OUT = os.path.join(HERE, "fill_sweep", "predictions")
METHODS = ["zero", "self_median", "train_mean", "train_median", "interp"]

# key -> published rule
DEEP = {"ltae": "zero", "ltae_field": "zero", "tempcnn": "zero", "tempcnn_field": "zero",
        "transformer": "zero", "cnn_bilstm": "zero", "tabnet_temporal_field": "zero",
        "tabnet": "self_median", "ltae_s": "zero", "ltae_s_field": "zero", "cnn_bilstm_field": "zero"}
TS = {"timesnet": "zero", "timesnet_field": "zero", "minirocket": "zero", "minirocket_field": "zero"}


def reuse(key, method, published, out):
    """Copy an existing re-analysis prediction file if it is the same computation."""
    src = None
    if method == published:
        src = os.path.join(FILL_CHECK, "predictions", f"{key}_orig.csv")
    elif method == "train_median":
        src = os.path.join(FILL_CHECK, "predictions", f"{key}_trainmed.csv")
    if src and os.path.exists(src):
        shutil.copyfile(src, out)
        return True
    return False


def run_deep(key, out):
    import importlib
    import run_deep_field as r  # in experiments/holdout_fill_check
    if key in r.OOS:
        mod = importlib.import_module(r.OOS[key])
        mod.TEST_PARQUET = r.FILES["orig"]
        mod.OUTPUT_CSV = out
        mod.main()
    else:
        import predict_oos as po
        po.MERGED_DL_TEST_PATH = r.FILES["orig"]
        fn, model_dir = r.PO[key]
        getattr(po, fn)(model_dir, out)


def main_deep(todo):
    for p in ["deep_learn/src", "out_of_sample", "experiments/field_reduction", "experiments/holdout_fill_check"]:
        sys.path.insert(0, os.path.join(REPO, p))
    import run_deep_field as r
    r.tabnet_shim()
    for key, method, out in todo:
        os.environ["HOLDOUT_FILL"] = method
        run_deep(key, out)
        print(f"[done] {key} {method}", flush=True)


def main_ts(todo):
    rb = os.path.join(REPO, "experiments", "recent_baselines")
    sys.path.insert(0, rb)
    sys.path.insert(0, os.path.join(rb, "minirocket_sparsity"))
    import numpy as np
    import torch
    import common
    common.set_determinism()
    by_level = {}
    for key, method, out in todo:
        by_level.setdefault("field" if key.endswith("_field") else "pixel", []).append((key, method, out))
    for level, items in by_level.items():
        data = common.load_training_data(level)
        mr_models = None
        for key, method, out in items:
            os.environ["HOLDOUT_FILL"] = method
            X, fids = common.load_holdout(level, data["feature_cols"], data["scaler"])
            common.PRED_DIR = os.path.dirname(out)
            stem = os.path.basename(out)[:-4]
            if key.startswith("timesnet"):
                import train_timesnet as tn
                device = tn.get_device()
                d = os.path.join(common.MODELS_DIR, f"timesnet_{level}")
                loader = tn.make_loader(X, np.zeros(len(X), dtype=np.int64), tn.BATCH_SIZE[level], False, 0)
                logits = [tn.evaluate(tn.load_seed(s, len(data["le"].classes_), d, device, level), loader, device)[0]
                          for s in common.SEEDS_ENSEMBLE]
                tn._write_holdout(torch.stack(logits).mean(0), fids, data["le"], level, stem)
            else:
                import features as fx
                import train_minirocket as tm
                torch.set_num_threads(tm.TORCH_THREADS)
                if mr_models is None:
                    mr_models = [fx.load_seed_model(level, s)[0] for s in common.SEEDS_ENSEMBLE]
                probs = []
                for m in mr_models:
                    head = tm.LogRegHead(m["C"])
                    head.W, head.b = torch.from_numpy(m["W"]), torch.from_numpy(m["b"])
                    probs.append(np.concatenate([head.predict_proba(F) for F in fx.iter_features(m, X)]))
                tm.write_holdout(np.mean(probs, 0), fids, data["le"], level, stem)
            print(f"[done] {key} {method}", flush=True)
        del data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["deep", "ts"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--no-reuse", action="store_true")
    a = ap.parse_args()
    models = DEEP if a.group == "deep" else TS
    keys = a.models or list(models)
    os.makedirs(OUT, exist_ok=True)
    todo = []
    for key in keys:
        for method in a.methods:
            out = os.path.join(OUT, f"{key}_{method}.csv")
            if os.path.exists(out):
                continue
            if not a.no_reuse and reuse(key, method, models[key], out):
                print(f"[reused] {key} {method}", flush=True)
                continue
            todo.append((key, method, out))
    print(f"{len(todo)} runs to do", flush=True)
    (main_deep if a.group == "deep" else main_ts)(todo)
