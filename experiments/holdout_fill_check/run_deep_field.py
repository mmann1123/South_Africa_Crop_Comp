"""Re-run holdout inference for the raw-input deep models and the patch CNNs under two fills.

  orig      data/merged_dl_test.parquet as published (missing September filled by each
            pipeline: 0 for the sequence models, holdout medians for pixel TabNet)
  trainmed  experiments/holdout_fill_check/data/merged_dl_test_trainmedian.parquet
            (missing September filled with training-tile medians; see make_filled_holdout.py)

The published inference code is used unchanged: each module's test-file path and output
path are redirected, nothing else. The 'orig' run must reproduce the published
predictions for a model's 'trainmed' result to be comparable; score.py checks this.
Writes only to experiments/holdout_fill_check/predictions/. Run in the deep_field env:
  python experiments/holdout_fill_check/run_deep_field.py [--models ltae tempcnn ...]
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "experiments", "field_reduction"))
sys.path.insert(0, HERE)

from config import MERGED_DL_TEST_PATH, MODEL_DIR  # noqa: E402

FILES = {"orig": MERGED_DL_TEST_PATH,
         "trainmed": os.path.join(HERE, "data", "merged_dl_test_trainmedian.parquet")}
PRED = os.path.join(HERE, "predictions")
FR_MODELS = os.path.join(REPO, "experiments", "field_reduction", "models")

# key -> inference_* module in out_of_sample/ (module reads TEST_PARQUET, writes OUTPUT_CSV)
OOS = {"ltae": "inference_ltae", "ltae_field": "inference_ltae_field",
       "tempcnn": "inference_tempcnn", "tempcnn_field": "inference_tempcnn_field",
       "transformer": "inference_transformer", "cnn_bilstm": "inference_cnn_bilstm",
       "tabnet_temporal_field": "inference_tabnet_temporal_field", "tabnet": "inference_tabnet"}
# key -> (predict_oos function, model dir); predict_oos reads its module-level MERGED_DL_TEST_PATH
PO = {"ltae_s": ("predict_ltae_sparse_pixel", os.path.join(FR_MODELS, "ltae_sparse_pixel", "frac_1.00")),
      "ltae_s_field": ("predict_ltae_sparse_field", os.path.join(FR_MODELS, "ltae_sparse_field", "frac_1.00")),
      "cnn_bilstm_field": ("predict_cnn_bilstm_field", MODEL_DIR)}
# key -> inference module for the patch CNNs (module reads TEST_PATCH_DATA, fills NaN with FILL,
# writes OUTPUT_CSV). 'orig' keeps FILL = None (0, as published); 'trainmed' fills each column with
# its median over the training patch pixels (patch_level_data.parquet), the patch analogue of the
# training-tile medians used for the pixel models.
PATCH = {"cnn3d": "inference_3d_cnn", "mc2d": "inference_multi_channel_cnn"}
PATCH_MEDIANS = os.path.join(HERE, "data", "train_patch_medians.csv")


def patch_medians():
    """Per-column medians of the training patch pixels, cached as CSV (index = column name)."""
    if os.path.exists(PATCH_MEDIANS):
        return pd.read_csv(PATCH_MEDIANS, index_col=0).iloc[:, 0]
    from config import PATCH_DATA_PATH
    import pyarrow.parquet as pq
    ignore = {"patch_id", "field_id", "crop_name", "row", "col"}
    cols = [c for c in pq.ParquetFile(PATCH_DATA_PATH).schema.names if c not in ignore]
    med = pd.read_parquet(PATCH_DATA_PATH, columns=cols).median(axis=0, skipna=True)
    med = med.dropna()  # May columns are all-NaN in training and are dropped by the inference scripts
    os.makedirs(os.path.dirname(PATCH_MEDIANS), exist_ok=True)
    med.rename("median").to_csv(PATCH_MEDIANS)
    return med


def tabnet_shim():
    """Old import path pytorch_tabnet.tab_model, with a checkpoint loader that tolerates
    the two feature-grouping buffers pytorch_tabnet 4.x added (identity when unused).
    Only these two keys may be missing; anything else stops the run."""
    import io
    import json
    import types
    import zipfile

    import torch
    import pytorch_tabnet
    from tabnet_fill import ALLOWED_MISSING

    class TabNetClassifier(pytorch_tabnet.TabNetClassifier):
        def load_model(self, filepath):
            with zipfile.ZipFile(filepath) as z:
                params = json.load(z.open("model_params.json"))
                params["init_params"]["device_name"] = self.device_name
                state = torch.load(io.BytesIO(z.open("network.pt").read()), map_location=self.device)
            self.__init__(**params["init_params"])
            self._set_network()
            res = self.network.load_state_dict(state, strict=False)
            if set(res.missing_keys) - ALLOWED_MISSING or res.unexpected_keys:
                raise RuntimeError(f"Unexpected checkpoint mismatch in {filepath}: {res}")
            for name, buf in self.network.named_buffers():
                if name in res.missing_keys and not torch.equal(buf, torch.eye(buf.shape[0], device=buf.device)):
                    raise RuntimeError(f"{name} is not the identity in {filepath}")
            self.network.eval()
            self.load_class_attrs(params["class_attrs"])

    shim = types.ModuleType("pytorch_tabnet.tab_model")
    shim.TabNetClassifier = TabNetClassifier
    sys.modules["pytorch_tabnet.tab_model"] = shim


def run(key, variant):
    out = os.path.join(PRED, f"{key}_{variant}.csv")
    if key == "tabnet_field":  # xr_fresh field TabNet: reads the rebuilt field-level features
        import importlib
        mod = importlib.import_module("inference_tabnet_field")
        mod.TEST_PARQUET = os.path.join(HERE, "data", f"combined_test_features_{variant}.parquet")
        mod.OUTPUT_CSV = out
        from joblib import load as _load

        def load(path):  # sklearn 1.5.2 SimpleImputer pickle under 1.8: add _fill_dtype (= fit dtype)
            obj = _load(path)
            if type(obj).__name__ == "SimpleImputer" and not hasattr(obj, "_fill_dtype"):
                obj._fill_dtype = obj._fit_dtype
            return obj
        mod.load = load
        mod.main()
    elif key in OOS:
        import importlib
        mod = importlib.import_module(OOS[key])
        mod.TEST_PARQUET = FILES[variant]
        mod.OUTPUT_CSV = out
        mod.main()
    elif key in PATCH:
        import importlib
        mod = importlib.import_module(PATCH[key])
        mod.FILL = None if variant == "orig" else patch_medians()
        mod.OUTPUT_CSV = out
        mod.main()
    else:
        import predict_oos as po
        po.MERGED_DL_TEST_PATH = FILES[variant]
        fn, model_dir = PO[key]
        getattr(po, fn)(model_dir, out)
    print(f"[done] {key} {variant} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(OOS) + list(PO) + ["tabnet_field"] + list(PATCH))
    ap.add_argument("--variants", nargs="+", default=list(FILES))
    args = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    tabnet_shim()
    for k in args.models:
        for v in args.variants:
            run(k, v)
