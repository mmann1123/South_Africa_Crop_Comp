"""Pixel TabNet holdout inference: median fill vs. zero fill.

out_of_sample/inference_tabnet.py fills missing holdout values with the holdout's
own column medians before applying the training scaler. The other raw-input models
(L-TAE, L-TAE-S, TempCNN, Transformer, CNN-BiLSTM, TimesNet, MiniRocket, patch CNNs)
fill missing values with 0 before scaling. This script re-runs the published 5-seed
TabNet ensemble under both rules so all models can be compared under the same one.

Steps:
  1. median fill (published rule): must reproduce out_of_sample/predictions_tabnet.csv
     exactly, otherwise the comparison is not valid and the script stops.
  2. zero fill (rule used by the other raw-input models).
  3. score both and run a paired field bootstrap (B=10,000, seed 42) of zero fill vs.
     the published predictions, with an exact McNemar test.

The February checkpoints predate pytorch_tabnet 4.x, which added two feature-grouping
buffers (embedder.embedding_group_matrix, tabnet.encoder.group_attention_matrix). With
no feature groups both default to identity matrices, which is the old behavior, so the
checkpoints are loaded non-strictly and only those two keys may be missing. Step 1's
exact reproduction of the published predictions confirms the load is faithful.

Writes only to experiments/tabnet_fill_check/results/. Run in the deep_field env:
  python experiments/tabnet_fill_check/run.py
"""

import os
import sys

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))

from config import MERGED_DL_TEST_PATH, TABNET_DIR  # noqa: E402
import inference_tabnet as inf  # noqa: E402
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402

OUT = os.path.join(HERE, "results")
PUBLISHED = os.path.join(REPO, "out_of_sample", "predictions_tabnet.csv")
CROP_ID_TO_NAME = {1: "Wheat", 2: "Barley", 3: "Canola", 4: "Lucerne/Medics", 5: "Small grain grazing"}
B, SEED = 10_000, 42


ALLOWED_MISSING = {"embedder.embedding_group_matrix", "tabnet.encoder.group_attention_matrix"}


def load_models():
    """inference_tabnet.load_models, but tolerating the two 4.x grouping buffers."""
    import io
    import json
    import zipfile

    import torch
    from pytorch_tabnet import TabNetClassifier

    models = []
    for seed in inf.SEEDS:
        path = os.path.join(TABNET_DIR, f"tabnet_seed_{seed}.zip")
        model = TabNetClassifier()
        with zipfile.ZipFile(path) as z:
            params = json.load(z.open("model_params.json"))
            params["init_params"]["device_name"] = model.device_name
            state = torch.load(io.BytesIO(z.open("network.pt").read()), map_location=model.device)
        model.__init__(**params["init_params"])
        model._set_network()
        res = model.network.load_state_dict(state, strict=False)
        if set(res.missing_keys) - ALLOWED_MISSING or res.unexpected_keys:
            sys.exit(f"Unexpected checkpoint mismatch for seed {seed}: {res}")
        for name, buf in model.network.named_buffers():
            if name in ALLOWED_MISSING and not torch.equal(buf, torch.eye(buf.shape[0], device=buf.device)):
                sys.exit(f"{name} is not the identity for seed {seed}")
        model.network.eval()
        model.load_class_attrs(params["class_attrs"])
        models.append(model)
        print(f"Loaded tabnet_seed_{seed}.zip (missing only: {sorted(res.missing_keys)})")
    return models


def features(df, fill):
    """Same steps as inference_tabnet.load_test_data, except the NaN rule."""
    feature_columns = load(os.path.join(TABNET_DIR, "tabnet_feature_columns.joblib"))
    scaler = load(os.path.join(TABNET_DIR, "tabnet_scaler.joblib"))
    one_hot = [c for c in feature_columns if c.startswith("Type_")]
    numeric = [c for c in feature_columns if c not in one_hot]
    df = df.copy()
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"])
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    present = [c for c in numeric if c in df.columns]
    if fill == "median":
        df[present] = df[present].fillna(df[present].median())
    df[present] = df[present].fillna(0)
    df[numeric] = scaler.transform(df[numeric])
    df[numeric] = df[numeric].replace([np.inf, -np.inf], 0).fillna(0)
    return df[feature_columns].astype(np.float32).values


def predict(X, fids, models, le):
    pm = np.mean([m.predict_proba(X) for m in models], axis=0)
    fp = inf.aggregate_to_field_level(np.argmax(pm, axis=1), fids)
    names = [CROP_ID_TO_NAME[c] for c in le.inverse_transform(fp.values)]
    return pd.DataFrame({"fid": fp.index, "crop_name": names})


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(MERGED_DL_TEST_PATH)
    fids = df["fid"].values
    le = load(os.path.join(TABNET_DIR, "tabnet_label_encoder.joblib"))
    models = load_models()

    preds = {}
    for fill in ["median", "zero"]:
        preds[fill] = predict(features(df, fill), fids, models, le)
        preds[fill].to_csv(os.path.join(OUT, f"predictions_tabnet_{fill}fill.csv"), index=False)
        print(f"{fill} fill: {len(preds[fill])} fields")

    pub = pd.read_csv(PUBLISHED)
    m = pub.merge(preds["median"], on="fid", suffixes=("_pub", "_rep"))
    n_diff = int((m["crop_name_pub"] != m["crop_name_rep"]).sum())
    repro = len(m) == len(pub) == len(preds["median"]) and n_diff == 0
    print(f"Median-fill reproduction of published predictions: "
          f"{'IDENTICAL' if repro else f'{n_diff} fields differ'} ({len(m)} fields)")
    if not repro:
        sys.exit("Published predictions not reproduced; comparison stopped.")

    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    a = preds["zero"].set_index("fid")["crop_name"]
    b = pub.set_index("fid")["crop_name"]
    j = pd.concat([gt.rename("t"), a.rename("a"), b.rename("b")], axis=1, join="inner")
    classes = sorted(set(j["t"]) | set(j["a"]) | set(j["b"]))
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    yt, ya, yb = (j[c].map(cidx).values for c in ("t", "a", "b"))
    pa, pb = metrics(confusion(yt, ya, K)), metrics(confusion(yt, yb, K))
    rng = np.random.default_rng(SEED)
    N = len(j)
    d = np.empty((B, 3))
    for i in range(B):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        d[i] = metrics(confusion(yt, ya, K, w)) - metrics(confusion(yt, yb, K, w))
    ca, cb = ya == yt, yb == yt
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mcn = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    rows = []
    for k, met in enumerate(["macro_f1", "kappa", "accuracy"]):
        lo, hi = np.percentile(d[:, k], [2.5, 97.5])
        rows.append({"metric": met, "zero_fill": round(pa[k], 4), "published_median_fill": round(pb[k], 4),
                     "delta": round(pa[k] - pb[k], 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "fields_changed": int((ya != yb).sum()), "zero_only_correct": n10,
                     "published_only_correct": n01, "mcnemar_p": round(p_mcn, 4), "n_fields": N})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
