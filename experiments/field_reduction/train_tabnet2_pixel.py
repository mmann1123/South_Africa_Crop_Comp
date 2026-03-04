"""Train TabNet2 (pixel-level) with a reduced fraction of training fields.

Uses pytorch-tabnet2 (DanielAvdar/tabnet) — speed-optimized fork with
vectorized data flow, GPU metrics, and loss-based weighting.
5-seed ensemble with averaged probabilities. Field-level evaluation via majority vote.

Usage:
    python train_tabnet2_pixel.py --fraction 0.50 --output-dir models/tabnet2_pixel/frac_0.50
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from pytorch_tabnet import TabNetClassifier
from pytorch_tabnet.metrics import Metric

from experiment_config import MERGED_DL_PATH, SEEDS_ENSEMBLE, SUBSAMPLE_SEED
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss for TabNet (non-buffer version for pickling)."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, weight=None):
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

    def __call__(self, y_true, y_score, weights=None):
        if isinstance(y_score, torch.Tensor):
            y_score = y_score.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        preds = np.argmax(y_score, axis=1)
        return f1_score(y_true, preds, average='macro')


def aggregate_field_preds(df_subset, y_preds, label_col='crop_label'):
    """Majority vote per field."""
    pred_df = pd.DataFrame({
        'fid': df_subset['fid'].values,
        'pred_label': y_preds,
        'true_label': df_subset[label_col].values,
    })
    field_pred = pred_df.groupby('fid')['pred_label'].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    field_true = pred_df.groupby('fid')['true_label'].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    return field_true, field_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    fraction = args.fraction
    print(f"=== TabNet2 Pixel Training === Fraction: {fraction}")

    # Load data
    print("Loading data...")
    df = pd.read_parquet(MERGED_DL_PATH)
    df = df.drop(columns=['May'], errors='ignore')
    print(f"Shape: {df.shape}")

    # Preprocess (matches original TabNet pipeline)
    exclude_cols = {'id', 'point', 'fid', 'crop_id', 'crop_name'}
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[numeric_cols] = df[numeric_cols].fillna(0)

    if 'Type' in df.columns:
        df = pd.get_dummies(df, columns=['Type'])
    one_hot_cols = [col for col in df.columns if col.startswith('Type_')]
    feature_columns = numeric_cols + one_hot_cols

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    features = df[feature_columns].astype(np.float32)
    label_encoder = LabelEncoder()
    df['crop_label'] = label_encoder.fit_transform(df['crop_id'])
    targets = df['crop_label'].values

    # FID split
    train_fids, val_fids, test_fids = get_fid_split_dl(df)

    # Subsample train FIDs
    sub_train_fids = subsample_train_fids(df, train_fids, fraction, seed=SUBSAMPLE_SEED)
    print(f"Train FIDs: {len(train_fids)} -> {len(sub_train_fids)} ({fraction*100:.0f}%)")

    train_mask = df['fid'].isin(sub_train_fids)
    val_mask = df['fid'].isin(val_fids)
    test_mask = df['fid'].isin(test_fids)

    X_train, y_train = features[train_mask].values, targets[train_mask]
    X_val, y_val = features[val_mask].values, targets[val_mask]
    X_test, y_test = features[test_mask].values, targets[test_mask]
    print(f"Pixels: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Class weights for focal loss
    class_counts = np.bincount(y_train, minlength=len(label_encoder.classes_)).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    alpha_weights = 1.0 / class_counts
    alpha_weights = alpha_weights / alpha_weights.sum() * len(label_encoder.classes_)
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32)

    # Train 5-seed ensemble
    test_preds_all = []
    t_train_start = time.time()

    for i, seed in enumerate(SEEDS_ENSEMBLE):
        print(f"\n=== Model {i+1}/{len(SEEDS_ENSEMBLE)} (seed={seed}) ===")
        t_model = time.time()

        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=1e-3, weight_decay=1e-4),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params={"T_max": 100, "eta_min": 1e-6},
            seed=seed,
            verbose=1,
        )

        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=[F1MacroMetric],
            loss_fn=WeightedFocalLoss(alpha_tensor, gamma=2.0),
            max_epochs=100,
            patience=15,
            batch_size=2048,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        print(f"Training done in {time.time()-t_model:.1f}s")

        # Save model
        model_path = os.path.join(args.output_dir, f"tabnet_seed_{seed}")
        model.save_model(model_path)

        test_preds_all.append(model.predict_proba(X_test))

    train_time = time.time() - t_train_start

    # Save preprocessing artifacts (same naming as original tabnet for predict_oos reuse)
    joblib.dump(scaler, os.path.join(args.output_dir, "tabnet_scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(args.output_dir, "tabnet_feature_columns.joblib"))
    joblib.dump(label_encoder, os.path.join(args.output_dir, "tabnet_label_encoder.joblib"))

    # Ensemble evaluation with field-level aggregation
    test_pred_mean = np.mean(test_preds_all, axis=0)
    y_test_pred = np.argmax(test_pred_mean, axis=1)

    test_field_true, test_field_pred = aggregate_field_preds(df[test_mask], y_test_pred)

    acc = accuracy_score(test_field_true, test_field_pred)
    f1_m = f1_score(test_field_true, test_field_pred, average='macro')
    f1_w = f1_score(test_field_true, test_field_pred, average='weighted')
    kappa = cohen_kappa_score(test_field_true, test_field_pred)

    print(f"\n--- Ensemble Field-Level Results ---")
    print(f"  acc={acc:.4f}, f1_macro={f1_m:.4f}, f1_weighted={f1_w:.4f}, kappa={kappa:.4f}")

    # Save metadata
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            "model": "tabnet2_pixel",
            "fraction": fraction,
            "train_fields": int(len(sub_train_fids)),
            "train_pixels": int(len(X_train)),
            "seeds": SEEDS_ENSEMBLE,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "batch_size": 2048,
            "patience": 15,
            "training_time_sec": round(train_time, 1),
            "metrics": {
                "accuracy": round(acc, 6),
                "f1_macro": round(f1_m, 6),
                "f1_weighted": round(f1_w, 6),
                "cohen_kappa": round(kappa, 6),
            },
        }, f, indent=2)

    print(f"\nArtifacts saved to {args.output_dir}")
    print(f"Total time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
