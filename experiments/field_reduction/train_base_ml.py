"""Train Base LightGBM (pixel) and Base LR (pixel) with a reduced fraction of training fields.

Usage:
    python train_base_ml.py --fraction 0.50 --output-dir-lgbm models/base_lgbm_pixel/frac_0.50 \
                            --output-dir-lr models/base_lr_pixel/frac_0.50
"""

import argparse
import gc
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

from experiment_config import (
    FINAL_DATA_PATH,
    SUBSAMPLE_SEED,
)
from subsample import get_fid_split_base_ml, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)


def prepare_data(df, label_encoder, exclude_cols=None):
    """Replicate base_ml_models.py prepare_data: drop NaN columns, encode labels."""
    if exclude_cols is None:
        exclude_cols = ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN']
    feature_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col not in exclude_cols and not df[col].isna().any()]
    X = df[feature_cols]
    y = label_encoder.transform(df['crop_name'])
    return X, y, feature_cols


def evaluate_field_level(fids, y_true, y_pred, label_encoder):
    """Aggregate pixel predictions to field level via majority vote."""
    preds_df = pd.DataFrame({
        'fid': fids,
        'true_label': label_encoder.inverse_transform(y_true),
        'pred_label': label_encoder.inverse_transform(y_pred),
    })
    field_preds = (
        preds_df.groupby('fid')
        .agg(lambda x: Counter(x).most_common(1)[0][0])
        .reset_index()
    )
    true_f = field_preds['true_label']
    pred_f = field_preds['pred_label']
    acc = accuracy_score(true_f, pred_f)
    f1_m = f1_score(true_f, pred_f, average='macro')
    f1_w = f1_score(true_f, pred_f, average='weighted')
    kappa = cohen_kappa_score(true_f, pred_f)
    return acc, f1_m, f1_w, kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--output-dir-lgbm', type=str, required=True)
    parser.add_argument('--output-dir-lr', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir_lgbm, exist_ok=True)
    os.makedirs(args.output_dir_lr, exist_ok=True)
    t0 = time.time()
    fraction = args.fraction
    print(f"=== Base ML (LightGBM + LR) Training === Fraction: {fraction}")

    # Load data
    print("Loading data...")
    data = pd.read_parquet(FINAL_DATA_PATH)

    # FID split (base_ml uses different val split: test_size=0.15)
    train_fids, val_fids, test_fids = get_fid_split_base_ml(data)

    # Subsample train FIDs
    sub_train_fids = subsample_train_fids(data, train_fids, fraction, seed=SUBSAMPLE_SEED)
    print(f"Train FIDs: {len(train_fids)} -> {len(sub_train_fids)} ({fraction*100:.0f}%)")

    train_df = data[data['fid'].isin(sub_train_fids)].copy()
    val_df = data[data['fid'].isin(val_fids)].copy()
    test_df = data[data['fid'].isin(test_fids)].copy()

    # Prepare data — compute feature_cols from FULL training data for consistency
    label_encoder = LabelEncoder()
    label_encoder.fit(data['crop_name'])

    # Use full train fids to determine feature columns (same cols as original)
    full_train_df = data[data['fid'].isin(train_fids)]
    _, _, feature_cols = prepare_data(full_train_df, label_encoder)
    del full_train_df

    # Now extract with those columns
    X_train = train_df[feature_cols]
    y_train = label_encoder.transform(train_df['crop_name'])
    X_val = val_df[feature_cols]
    y_val = label_encoder.transform(val_df['crop_name'])
    X_test = test_df[feature_cols]
    y_test = label_encoder.transform(test_df['crop_name'])

    print(f"Pixels: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    del data
    gc.collect()

    # ========== LightGBM ==========
    print("\n--- Training LightGBM ---")
    t_lgbm = time.time()
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=1000, device="cuda", is_unbalance=True,
        verbose=-1, random_state=42,
    )
    lgbm_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgbm_time = time.time() - t_lgbm
    print(f"LightGBM best iteration: {lgbm_model.best_iteration_}, time: {lgbm_time:.1f}s")

    y_pred_lgbm = lgbm_model.predict(X_test_scaled)
    acc_l, f1m_l, f1w_l, kappa_l = evaluate_field_level(
        test_df['fid'].values, y_test, y_pred_lgbm, label_encoder
    )
    print(f"LightGBM field: acc={acc_l:.4f}, f1_macro={f1m_l:.4f}, kappa={kappa_l:.4f}")

    # Save LightGBM
    joblib.dump(lgbm_model, os.path.join(args.output_dir_lgbm, 'lightgbm.joblib'))
    joblib.dump(scaler, os.path.join(args.output_dir_lgbm, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(args.output_dir_lgbm, 'label_encoder.joblib'))
    joblib.dump(feature_cols, os.path.join(args.output_dir_lgbm, 'feature_columns.joblib'))
    with open(os.path.join(args.output_dir_lgbm, 'metadata.json'), 'w') as f:
        json.dump({
            "model": "base_lgbm_pixel",
            "fraction": fraction,
            "train_fields": int(len(sub_train_fids)),
            "train_pixels": int(len(X_train)),
            "training_time_sec": round(lgbm_time, 1),
            "best_iteration": int(lgbm_model.best_iteration_),
            "metrics": {
                "accuracy": round(acc_l, 6),
                "f1_macro": round(f1m_l, 6),
                "f1_weighted": round(f1w_l, 6),
                "cohen_kappa": round(kappa_l, 6),
            },
        }, f, indent=2)

    # ========== Logistic Regression ==========
    print("\n--- Training Logistic Regression ---")
    t_lr = time.time()
    lr_model = LogisticRegression(max_iter=500, n_jobs=4, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    lr_time = time.time() - t_lr
    print(f"LR training time: {lr_time:.1f}s")

    y_pred_lr = lr_model.predict(X_test_scaled)
    acc_r, f1m_r, f1w_r, kappa_r = evaluate_field_level(
        test_df['fid'].values, y_test, y_pred_lr, label_encoder
    )
    print(f"LR field: acc={acc_r:.4f}, f1_macro={f1m_r:.4f}, kappa={kappa_r:.4f}")

    # Save LR
    joblib.dump(lr_model, os.path.join(args.output_dir_lr, 'logistic_regression.joblib'))
    joblib.dump(scaler, os.path.join(args.output_dir_lr, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(args.output_dir_lr, 'label_encoder.joblib'))
    joblib.dump(feature_cols, os.path.join(args.output_dir_lr, 'feature_columns.joblib'))
    with open(os.path.join(args.output_dir_lr, 'metadata.json'), 'w') as f:
        json.dump({
            "model": "base_lr_pixel",
            "fraction": fraction,
            "train_fields": int(len(sub_train_fids)),
            "train_pixels": int(len(X_train)),
            "training_time_sec": round(lr_time, 1),
            "metrics": {
                "accuracy": round(acc_r, 6),
                "f1_macro": round(f1m_r, 6),
                "f1_weighted": round(f1w_r, 6),
                "cohen_kappa": round(kappa_r, 6),
            },
        }, f, indent=2)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
