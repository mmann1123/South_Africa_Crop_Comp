"""Generate OOS predictions for all fraction models.

Loads trained model artifacts from models/{name}/frac_{F}/ and runs inference
on the holdout test data, producing prediction CSVs in results/predictions/.

DL models (tabnet, ltae) require torch — run with deep_field python.
ML models (xgboost, lgbm, lr) need only sklearn/joblib — run with ml_field python.
The orchestrator (run_experiment.py) dispatches each group with the correct env.

Usage:
    python predict_oos.py                          # all models, all fractions
    python predict_oos.py --models xgboost_field   # single model
    python predict_oos.py --fractions 0.50 0.75    # specific fractions
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from joblib import load as jl_load

from experiment_config import (
    MERGED_DL_TEST_PATH,
    COMBINED_TEST_FEATURES_PATH,
    FRACTIONS,
    SEEDS_ENSEMBLE,
    MODELS_DIR,
    PREDICTIONS_DIR,
)

sys.stdout.reconfigure(line_buffering=True)

# TabNet label_encoder uses crop_id, need this mapping
CROP_ID_TO_NAME = {
    1: "Wheat", 2: "Barley", 3: "Canola",
    4: "Lucerne/Medics", 5: "Small grain grazing",
}

# DL models that require torch
DL_MODELS = {"tabnet_pixel", "ltae_field", "ltae_pixel"}


def _get_torch_device():
    """Lazy import torch and return device."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================== Prediction functions per model ===================

def predict_tabnet(model_dir, output_csv):
    """TabNet pixel: 5-seed ensemble → avg proba → majority vote per field."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    # Load artifacts
    feature_columns = jl_load(os.path.join(model_dir, "tabnet_feature_columns.joblib"))
    scaler = jl_load(os.path.join(model_dir, "tabnet_scaler.joblib"))
    label_encoder = jl_load(os.path.join(model_dir, "tabnet_label_encoder.joblib"))

    # Load and preprocess test data (matches inference_tabnet.py)
    df = pd.read_parquet(MERGED_DL_TEST_PATH)
    one_hot_cols = [c for c in feature_columns if c.startswith("Type_")]
    numeric_cols = [c for c in feature_columns if c not in one_hot_cols]

    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    present_numeric = [c for c in numeric_cols if c in df.columns]
    df[present_numeric] = df[present_numeric].fillna(df[present_numeric].median())
    df[present_numeric] = df[present_numeric].fillna(0)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    features = df[feature_columns].astype(np.float32).values
    fids = df["fid"].values

    # Load models and predict
    preds_all = []
    for seed in SEEDS_ENSEMBLE:
        model_path = os.path.join(model_dir, f"tabnet_seed_{seed}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TabNet model not found: {model_path}")
        model = TabNetClassifier()
        model.load_model(model_path)
        preds_all.append(model.predict_proba(features))

    pred_mean = np.mean(preds_all, axis=0)
    preds = np.argmax(pred_mean, axis=1)

    # Field-level majority vote
    pred_df = pd.DataFrame({"fid": fids, "pred": preds})
    field_preds = pred_df.groupby("fid")["pred"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )

    # Convert label indices → crop_id → crop_name
    field_crop_ids = label_encoder.inverse_transform(field_preds.values)
    field_labels = [CROP_ID_TO_NAME[cid] for cid in field_crop_ids]

    df_out = pd.DataFrame({"fid": field_preds.index, "crop_name": field_labels})
    df_out.to_csv(output_csv, index=False)
    return len(df_out)


def predict_ltae_field(model_dir, output_csv):
    """L-TAE field: aggregate test pixels to field → 5-seed ensemble."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from models_arch import LTAE, T_SEQ, N_BANDS

    device = _get_torch_device()

    feature_cols = jl_load(os.path.join(model_dir, "feature_columns.joblib"))
    scaler = jl_load(os.path.join(model_dir, "scaler.joblib"))
    le = jl_load(os.path.join(model_dir, "label_encoder.joblib"))

    df = pd.read_parquet(MERGED_DL_TEST_PATH)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Aggregate to field level
    df_field = df.groupby('fid')[feature_cols].mean().reset_index()
    X = scaler.transform(df_field[feature_cols].values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    fids = df_field["fid"].values

    X_tensor = torch.tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32)
    fid_list = list(fids)

    class _Dataset(Dataset):
        def __len__(self): return len(fid_list)
        def __getitem__(self, idx): return X_tensor[idx], fid_list[idx]

    dataloader = DataLoader(_Dataset(), batch_size=256, shuffle=False)

    # 5-seed ensemble
    logits_all = []
    for seed in SEEDS_ENSEMBLE:
        model_path = os.path.join(model_dir, f"ltae_field_seed_{seed}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"L-TAE field model not found: {model_path}")
        model = LTAE(in_channels=N_BANDS, num_classes=len(le.classes_)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        seed_logits = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for X_batch, _ in dataloader:
                seed_logits.append(model(X_batch.to(device)).cpu())
        logits_all.append(torch.cat(seed_logits, dim=0).unsqueeze(0))

    avg_logits = torch.cat(logits_all, dim=0).float().mean(dim=0)
    preds = avg_logits.argmax(dim=1).tolist()
    labels = le.inverse_transform(preds)

    df_out = pd.DataFrame({"fid": fids, "crop_name": labels})
    df_out.to_csv(output_csv, index=False)
    return len(df_out)


def predict_ltae_pixel(model_dir, output_csv):
    """L-TAE pixel: 5-seed ensemble → avg logits → majority vote per field."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from models_arch import LTAE, T_SEQ, N_BANDS

    device = _get_torch_device()

    feature_cols = jl_load(os.path.join(model_dir, "feature_columns.joblib"))
    scaler = jl_load(os.path.join(model_dir, "scaler.joblib"))
    le = jl_load(os.path.join(model_dir, "label_encoder.joblib"))

    df = pd.read_parquet(MERGED_DL_TEST_PATH)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = scaler.transform(df[feature_cols].values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    fids = df["fid"].values

    X_tensor = torch.tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32)
    fid_list = list(fids)

    class _Dataset(Dataset):
        def __len__(self): return len(fid_list)
        def __getitem__(self, idx): return X_tensor[idx], fid_list[idx]

    dataloader = DataLoader(_Dataset(), batch_size=2048, shuffle=False)

    logits_all = []
    for seed in SEEDS_ENSEMBLE:
        model_path = os.path.join(model_dir, f"ltae_seed_{seed}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"L-TAE pixel model not found: {model_path}")
        model = LTAE(in_channels=N_BANDS, num_classes=len(le.classes_)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        seed_logits = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for X_batch, _ in dataloader:
                seed_logits.append(model(X_batch.to(device)).cpu())
        logits_all.append(torch.cat(seed_logits, dim=0).unsqueeze(0))

    avg_logits = torch.cat(logits_all, dim=0).float().mean(dim=0)
    preds = avg_logits.argmax(dim=1).tolist()

    # Field-level majority vote
    pred_df = pd.DataFrame({"fid": fids, "pred": preds})
    field_preds = pred_df.groupby("fid")["pred"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    labels = le.inverse_transform(field_preds.values)

    df_out = pd.DataFrame({"fid": field_preds.index, "crop_name": labels})
    df_out.to_csv(output_csv, index=False)
    return len(df_out)


def predict_xgboost_field(model_dir, output_csv):
    """XGBoost field: load model + imputer + scaler, predict on field-level test data."""
    model = jl_load(os.path.join(model_dir, "xgboost_model.joblib"))
    imputer = jl_load(os.path.join(model_dir, "imputer.joblib"))
    scaler = jl_load(os.path.join(model_dir, "scaler.joblib"))
    le = jl_load(os.path.join(model_dir, "label_encoder.joblib"))

    df = pd.read_parquet(COMBINED_TEST_FEATURES_PATH)
    fids = df["fid"].to_numpy()
    X = df.drop(columns=["fid", "crop_name"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Align columns with training
    train_cols = imputer.feature_names_in_ if hasattr(imputer, "feature_names_in_") else None
    if train_cols is not None:
        for col in set(train_cols) - set(X.columns):
            X[col] = 0.0
        X = X[train_cols]

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    codes = model.predict(X_scaled)
    labels = le.inverse_transform(codes)

    df_out = pd.DataFrame({"fid": fids, "crop_name": labels})
    df_out.to_csv(output_csv, index=False)
    return len(df_out)


def predict_base_ml(model_dir, model_file, output_csv):
    """Base ML (LightGBM or LR): load model, predict on field-level test data."""
    model = jl_load(os.path.join(model_dir, model_file))
    scaler = jl_load(os.path.join(model_dir, "scaler.joblib"))
    le = jl_load(os.path.join(model_dir, "label_encoder.joblib"))
    feature_cols = jl_load(os.path.join(model_dir, "feature_columns.joblib"))

    df = pd.read_parquet(COMBINED_TEST_FEATURES_PATH)
    df = df.drop(columns=["May"], errors="ignore")
    fids = df["fid"].values

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X).astype(np.float32)

    codes = model.predict(X_scaled)
    labels = le.inverse_transform(codes)

    df_out = pd.DataFrame({"fid": fids, "crop_name": labels})
    df_out.to_csv(output_csv, index=False)
    return len(df_out)


# =================== Dispatcher ===================

PREDICT_FNS = {
    "tabnet_pixel": lambda md, out: predict_tabnet(md, out),
    "ltae_field": lambda md, out: predict_ltae_field(md, out),
    "ltae_pixel": lambda md, out: predict_ltae_pixel(md, out),
    "xgboost_field": lambda md, out: predict_xgboost_field(md, out),
    "base_lgbm_pixel": lambda md, out: predict_base_ml(md, "lightgbm.joblib", out),
    "base_lr_pixel": lambda md, out: predict_base_ml(md, "logistic_regression.joblib", out),
    # L2 variants — same prediction logic, different model dirs
    "xgboost_field_l2": lambda md, out: predict_xgboost_field(md, out),
    "base_lgbm_pixel_l2": lambda md, out: predict_base_ml(md, "lightgbm.joblib", out),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=list(PREDICT_FNS.keys()))
    parser.add_argument('--fractions', nargs='*', type=float, default=FRACTIONS)
    args = parser.parse_args()

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Only show torch device for DL models
    has_dl = any(m in DL_MODELS for m in args.models)
    if has_dl:
        device = _get_torch_device()
        print(f"=== OOS Prediction === Device: {device}")
    else:
        print("=== OOS Prediction === (ML models, no torch needed)")

    print(f"Models: {args.models}")
    print(f"Fractions: {args.fractions}")

    for model_name in args.models:
        if model_name not in PREDICT_FNS:
            print(f"[SKIP] Unknown model: {model_name}")
            continue

        for frac in args.fractions:
            frac_str = f"{frac:.2f}"
            model_dir = os.path.join(MODELS_DIR, model_name, f"frac_{frac_str}")
            output_csv = os.path.join(PREDICTIONS_DIR, f"{model_name}_frac_{frac_str}.csv")

            if not os.path.isdir(model_dir):
                print(f"[SKIP] Model dir not found: {model_dir}")
                continue

            print(f"\n--- {model_name} frac={frac_str} ---")
            try:
                n_fields = PREDICT_FNS[model_name](model_dir, output_csv)
                print(f"  Saved {n_fields} field predictions -> {output_csv}")
            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
