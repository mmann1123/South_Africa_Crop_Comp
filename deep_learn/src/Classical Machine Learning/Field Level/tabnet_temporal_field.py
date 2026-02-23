"""
TabNet ensemble on field-level RAW TEMPORAL features.

Same TabNet architecture as tabnet_field.py, but trained on field-averaged
raw Sentinel-2 temporal data (6 bands x 10 months = 60 features) instead
of xr_fresh statistical features (~150 features).

Pixel values from merged_dl_train.parquet are averaged per field before
training, preserving the temporal band structure.

5-seed ensemble with averaged probabilities.

Input: data/merged_dl_train.parquet (pixel-level raw temporal)
Output: saved_models_tabnet_temporal_field/ (5 models + preprocessing artifacts)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, TABNET_TEMPORAL_FIELD_DIR

sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from report import ModelReport


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss -- class weights (alpha) + difficulty focusing (gamma)."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        alpha = self.alpha.to(input.device)
        ce = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = alpha[target]
        loss = alpha_t * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class F1MacroMetric(Metric):
    """F1 Macro metric for TabNet eval_metric."""
    def __init__(self):
        self._name = "f1_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return f1_score(y_true, preds, average='macro')


SEEDS = [42, 101, 202, 303, 404]

# Bands and months in chronological order (matching merged_dl data)
BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
MONTHS_CHRONO = [
    "January", "February", "March", "April",
    "July", "August", "September",
    "October", "November", "December",
]


def get_chrono_feature_cols(df):
    """Build feature column list in chronological order."""
    cols = []
    for month in MONTHS_CHRONO:
        for band in BANDS:
            col = f"{band}_{month}"
            if col in df.columns:
                cols.append(col)
    return cols


def aggregate_field(df, feature_cols):
    """Aggregate pixel-level data to field-level: mean for features, mode for label."""
    y = df.groupby('fid')['crop_label'].agg(lambda x: x.mode()[0])
    X = df.groupby('fid')[feature_cols].mean()
    return X, y


def main():
    t0 = time.time()
    os.makedirs(TABNET_TEMPORAL_FIELD_DIR, exist_ok=True)

    # =================== Load Data ===================
    print("[TIMER] Loading data...")
    data = pd.read_parquet(MERGED_DL_PATH, engine="pyarrow")
    data = data.drop(columns=['May'], errors='ignore')
    print(f"[TIMER] Data loaded: {len(data)} rows, {data.shape[1]} cols, {time.time()-t0:.1f}s")

    # Get feature columns in chronological order
    feature_cols = get_chrono_feature_cols(data)
    print(f"Temporal features: {len(feature_cols)} ({len(BANDS)} bands x {len(MONTHS_CHRONO)} months)")

    # Fill NaN with median, then remaining with 0
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
    data[feature_cols] = data[feature_cols].fillna(0)

    # Label encode
    le = LabelEncoder()
    data['crop_label'] = le.fit_transform(data['crop_name'])
    print(f"Classes: {list(le.classes_)}")

    # =================== Fid-Wise Split ===================
    print("Splitting by FID...")
    fids = data['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)

    train_data = data[data['fid'].isin(train_fids)].copy()
    val_data = data[data['fid'].isin(val_fids)].copy()
    test_data = data[data['fid'].isin(test_fids)].copy()

    # =================== Aggregate Per Field ===================
    print("Aggregating per field...")
    X_train, y_train = aggregate_field(train_data, feature_cols)
    X_val, y_val = aggregate_field(val_data, feature_cols)
    X_test, y_test = aggregate_field(test_data, feature_cols)
    print(f"Fields: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # =================== Scale ===================
    print("Preprocessing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Handle NaN/inf from scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    print(f"Features: {len(feature_cols)}")

    # Compute class weights for focal loss
    class_counts = np.bincount(y_train_arr, minlength=len(le.classes_)).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    alpha_weights = 1.0 / class_counts
    alpha_weights = alpha_weights / alpha_weights.sum() * len(le.classes_)
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32)
    print(f"Class weights (alpha): {alpha_weights.tolist()}")

    # =================== 5-Seed Ensemble ===================
    val_preds_all, test_preds_all = [], []

    for i, seed in enumerate(SEEDS):
        print(f"\n=== Model {i+1}/{len(SEEDS)} (seed={seed}) ===")
        t_model = time.time()
        model_path = os.path.join(TABNET_TEMPORAL_FIELD_DIR, f"tabnet_temporal_field_seed_{seed}")

        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=seed,
            verbose=1,
        )

        if os.path.exists(model_path) and os.environ.get("FORCE_RETRAIN") != "1":
            print(f"Loading saved model for seed {seed}...")
            model.load_model(model_path)
            print(f"[TIMER] Model loaded in {time.time()-t_model:.1f}s")
        else:
            print(f"Training model for seed {seed}...")
            t_train = time.time()
            model.fit(
                X_train=X_train_scaled, y_train=y_train_arr,
                eval_set=[(X_val_scaled, y_val_arr)],
                eval_metric=[F1MacroMetric],
                loss_fn=WeightedFocalLoss(alpha_tensor, gamma=2.0),
                max_epochs=100,
                patience=10,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )
            print(f"[TIMER] Training done in {time.time()-t_train:.1f}s")
            model.save_model(model_path)
            print(f"Saved model to {model_path}")

        val_preds_all.append(model.predict_proba(X_val_scaled))
        test_preds_all.append(model.predict_proba(X_test_scaled))
        print(f"[TIMER] Model {i+1} total: {time.time()-t_model:.1f}s")

    # Save preprocessing artifacts
    joblib.dump(le, os.path.join(TABNET_TEMPORAL_FIELD_DIR, "tabnet_temporal_field_label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(TABNET_TEMPORAL_FIELD_DIR, "tabnet_temporal_field_scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(TABNET_TEMPORAL_FIELD_DIR, "tabnet_temporal_field_feature_columns.joblib"))
    print(f"Saved preprocessing artifacts to {TABNET_TEMPORAL_FIELD_DIR}")

    # =================== Ensemble Predictions ===================
    val_pred_mean = np.mean(val_preds_all, axis=0)
    test_pred_mean = np.mean(test_preds_all, axis=0)
    y_val_pred = np.argmax(val_pred_mean, axis=1)
    y_test_pred = np.argmax(test_pred_mean, axis=1)

    # =================== Evaluation ===================
    for split_name, y_true, y_pred in [
        ("Validation", y_val_arr, y_val_pred),
        ("Test", y_test_arr, y_test_pred),
    ]:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"\n--- {split_name} ---")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1 weighted: {f1:.4f}")
        print(f"  Kappa:       {kappa:.4f}")

    # =================== Report ===================
    report = ModelReport("TabNet Field-Level (temporal)")
    report.set_hyperparameters({
        "n_d": 64, "n_a": 64, "n_steps": 5,
        "gamma": 1.5, "n_independent": 2, "n_shared": 2,
        "lr": 1e-3, "scheduler": "StepLR(step=10, gamma=0.9)",
        "loss": "WeightedFocalLoss(gamma=2.0, alpha=1/class_counts)",
        "model_selection": "val F1 macro (maximize)",
        "max_epochs": 100, "patience": 10, "batch_size": 1024,
        "n_models": len(SEEDS), "seeds": SEEDS,
        "features": "raw temporal (field-level mean, 6 bands x 10 months)",
    })
    report.set_split_info(
        train=len(X_train), val=len(X_val), test=len(X_test),
        seed=42, split_method="fid-wise (field-level)",
    )
    report.set_metrics(y_test_arr, y_test_pred, list(le.classes_))
    report.set_training_time(time.time() - t0)
    report.generate()

    print(f"\n[TIMER] Total: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
