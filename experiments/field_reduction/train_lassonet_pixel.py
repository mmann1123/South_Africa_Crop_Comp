"""Train LassoNet (pixel-level) with a reduced fraction of training fields.

LassoNet: structured sparse neural network with automatic feature selection.
Residual architecture f(X) = theta^T X + g_W(X) with L1 on skip connection
and hierarchy constraint ||W^(1)_i||_inf <= M|theta_i|.
Uses the regularization path to find optimal sparsity on validation F1 macro.

Usage:
    python train_lassonet_pixel.py --fraction 0.50 --output-dir models/lassonet_pixel/frac_0.50
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from itertools import islice

import joblib
import numpy as np
import pandas as pd
import torch
from lassonet import LassoNetClassifier
from lassonet.interfaces import BaseLassoNet
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiment_config import MERGED_DL_PATH, SUBSAMPLE_SEED
from models_arch import get_chrono_feature_cols
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)

# LassoNet hyperparameters
HIDDEN_DIMS = (128,)
M = 10.0
DROPOUT = 0.3
PATH_MULTIPLIER = 1.02
BATCH_SIZE = 2048
N_ITERS = (1000, 100)
PATIENCE = (100, 10)


def compute_balanced_class_weight(y):
    """Compute sklearn-style balanced class weights as a list ordered by class."""
    classes = np.sort(np.unique(y))
    n_samples = len(y)
    n_classes = len(classes)
    return [n_samples / (n_classes * np.sum(y == c)) for c in classes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    fraction = args.fraction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== LassoNet Pixel Training === Fraction: {fraction}, Device: {device}")
    print(f"  hidden_dims={HIDDEN_DIMS}, M={M}, dropout={DROPOUT}, "
          f"batch_size={BATCH_SIZE}")

    # Load pixel-level data
    print("Loading data...")
    df = pd.read_parquet(MERGED_DL_PATH)
    print(f"Shape: {df.shape}")

    # Feature columns (flat 60-dim)
    feature_cols = get_chrono_feature_cols(df)
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    df[feature_cols] = df[feature_cols].fillna(0)
    in_dim = len(feature_cols)
    print(f"Feature cols: {in_dim}")

    # Labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['crop_name'])
    num_classes = len(le.classes_)

    # FID split
    train_fids, val_fids, test_fids = get_fid_split_dl(df)

    # Subsample train FIDs
    sub_train_fids = subsample_train_fids(df, train_fids, fraction, seed=SUBSAMPLE_SEED)
    print(f"Train FIDs: {len(train_fids)} -> {len(sub_train_fids)} ({fraction*100:.0f}%)")

    train_mask = df['fid'].isin(sub_train_fids)
    val_mask = df['fid'].isin(val_fids)
    test_mask = df['fid'].isin(test_fids)
    print(f"Pixels: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, feature_cols].values).astype(np.float32)
    X_val = scaler.transform(df.loc[val_mask, feature_cols].values).astype(np.float32)
    X_test = scaler.transform(df.loc[test_mask, feature_cols].values).astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = df.loc[train_mask, 'label'].values
    y_val = df.loc[val_mask, 'label'].values
    y_test = df.loc[test_mask, 'label'].values
    test_fid_arr = df.loc[test_mask, 'fid'].values

    del df
    gc.collect()

    # Balanced class weights
    class_weight = compute_balanced_class_weight(y_train)
    print(f"Class weights: {class_weight}")

    # Create LassoNet model
    model = LassoNetClassifier(
        hidden_dims=HIDDEN_DIMS,
        M=M,
        dropout=DROPOUT,
        gamma=0.0,
        gamma_skip=0.0,
        path_multiplier=PATH_MULTIPLIER,
        batch_size=BATCH_SIZE,
        n_iters=N_ITERS,
        patience=PATIENCE,
        class_weight=class_weight,
        device=device,
        verbose=2,
        random_state=42,
        torch_seed=42,
        val_size=0,
    )

    # Run the regularization path
    print("\n--- Running regularization path ---")
    t_train = time.time()
    path_result = model.path(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        return_state_dicts=True,
    )
    path_time = time.time() - t_train
    print(f"Path: {len(path_result)} steps in {path_time:.1f}s ({path_time/60:.1f} min)")

    # Select best model along the path using val F1 macro
    best_val_f1 = -1.0
    best_idx = 0

    for i, hist in enumerate(path_result):
        model.load(hist.state_dict)
        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average='macro')

        n_sel = hist.selected.sum().item()
        if i % 20 == 0 or val_f1 > best_val_f1:
            print(f"  Step {i}: lambda={hist.lambda_:.4f}, "
                  f"features={n_sel}/{in_dim}, val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_idx = i

    best_hist = path_result[best_idx]
    n_selected = int(best_hist.selected.sum().item())
    print(f"\nBest: step={best_idx}, lambda={best_hist.lambda_:.4f}, "
          f"features={n_selected}/{in_dim}, val_f1={best_val_f1:.4f}")

    # Load the best model
    model.load(best_hist.state_dict)

    # Test: pixel predictions -> field-level majority vote
    test_preds = model.predict(X_test)

    pred_df = pd.DataFrame({'fid': test_fid_arr, 'true': y_test, 'pred': test_preds})
    field_agg = pred_df.groupby('fid').agg(
        true=('true', lambda x: Counter(x).most_common(1)[0][0]),
        pred=('pred', lambda x: Counter(x).most_common(1)[0][0]),
    )
    acc = accuracy_score(field_agg['true'], field_agg['pred'])
    f1_m = f1_score(field_agg['true'], field_agg['pred'], average='macro')
    f1_w = f1_score(field_agg['true'], field_agg['pred'], average='weighted')
    kappa = cohen_kappa_score(field_agg['true'], field_agg['pred'])

    print(f"\n--- Field-Level Results ---")
    print(f"  acc={acc:.4f}, f1_macro={f1_m:.4f}, f1_weighted={f1_w:.4f}, kappa={kappa:.4f}")

    # Feature importances (lambda at which each feature is eliminated)
    feat_importances = BaseLassoNet._compute_feature_importances(path_result)

    # Save artifacts
    torch.save(best_hist.state_dict, os.path.join(args.output_dir, 'lassonet_state_dict.pt'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(args.output_dir, 'label_encoder.joblib'))
    joblib.dump(feature_cols, os.path.join(args.output_dir, 'feature_columns.joblib'))
    joblib.dump({
        "importances": feat_importances.cpu().numpy(),
        "feature_names": feature_cols,
        "selected_at_best": best_hist.selected.cpu().numpy(),
    }, os.path.join(args.output_dir, 'feature_importances.joblib'))

    # Selected features
    selected_names = [f for f, s in zip(feature_cols, best_hist.selected.cpu().numpy()) if s]
    print(f"\nSelected features ({n_selected}/{in_dim}):")
    for f in selected_names:
        print(f"  {f}")

    # Metadata
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            "model": "lassonet_pixel",
            "fraction": fraction,
            "train_fields": int(len(sub_train_fids)),
            "train_pixels": int(X_train.shape[0]),
            "val_pixels": int(X_val.shape[0]),
            "test_pixels": int(X_test.shape[0]),
            "test_fields": int(len(field_agg)),
            "hidden_dims": list(HIDDEN_DIMS),
            "M": M,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "path_multiplier": PATH_MULTIPLIER,
            "best_lambda": float(best_hist.lambda_),
            "best_path_step": best_idx,
            "total_path_steps": len(path_result),
            "features_selected": n_selected,
            "features_total": in_dim,
            "best_val_f1_macro": round(best_val_f1, 6),
            "training_time_sec": round(path_time, 1),
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
