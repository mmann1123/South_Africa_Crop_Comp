"""Re-run TimesNet and MiniRocket holdout inference under the two holdout files.

Same design as run_deep_field.py: the saved models and the published preprocessing
(common.load_holdout) are used unchanged; only the holdout file and the output folder
are redirected. MiniRocket features use the checked (double-computed) transform of
minirocket_sparsity/features.py. Run in the ts_baselines env:
  ~/miniconda3/envs/ts_baselines/bin/python experiments/holdout_fill_check/run_ts_baselines.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RB = os.path.join(REPO, "experiments", "recent_baselines")
sys.path.insert(0, RB)
sys.path.insert(0, os.path.join(RB, "minirocket_sparsity"))

import common  # noqa: E402
import torch  # noqa: E402

FILES = {"orig": common.MERGED_DL_TEST_PATH,
         "trainmed": os.path.join(HERE, "data", "merged_dl_test_trainmedian.parquet")}
PRED = os.path.join(HERE, "predictions")
common.PRED_DIR = PRED  # _write_holdout / write_holdout write here


def timesnet(level, data):
    import train_timesnet as tn
    device = tn.get_device()
    out_dir = os.path.join(common.MODELS_DIR, f"timesnet_{level}")
    nc = len(data["le"].classes_)
    bs = tn.BATCH_SIZE[level]
    for variant, path in FILES.items():
        common.MERGED_DL_TEST_PATH = path
        X, fids = common.load_holdout(level, data["feature_cols"], data["scaler"])
        loader = tn.make_loader(X, np.zeros(len(X), dtype=np.int64), bs, False, 0)
        logits = [tn.evaluate(tn.load_seed(s, nc, out_dir, device, level), loader, device)[0]
                  for s in common.SEEDS_ENSEMBLE]
        p = tn._write_holdout(torch.stack(logits).mean(0), fids, data["le"], level,
                              f"timesnet{'_field' if level == 'field' else ''}_{variant}")
        print("[done]", p, flush=True)


def minirocket(level, data):
    import features as fx
    import train_minirocket as tm
    torch.set_num_threads(tm.TORCH_THREADS)
    models = [fx.load_seed_model(level, s)[0] for s in common.SEEDS_ENSEMBLE]
    for variant, path in FILES.items():
        common.MERGED_DL_TEST_PATH = path
        X, fids = common.load_holdout(level, data["feature_cols"], data["scaler"])
        probs = []
        for m in models:
            head = tm.LogRegHead(m["C"])
            head.W, head.b = torch.from_numpy(m["W"]), torch.from_numpy(m["b"])
            probs.append(np.concatenate([head.predict_proba(F) for F in fx.iter_features(m, X)]))
        p = tm.write_holdout(np.mean(probs, 0), fids, data["le"], level,
                             f"minirocket{'_field' if level == 'field' else ''}_{variant}")
        print("[done]", p, flush=True)


if __name__ == "__main__":
    os.makedirs(PRED, exist_ok=True)
    common.set_determinism()
    for level in ["field", "pixel"]:
        data = common.load_training_data(level)
        timesnet(level, data)
        minirocket(level, data)
        del data
