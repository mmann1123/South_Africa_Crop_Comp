"""Train XGBoost (field-level) with a reduced fraction of training fields.

Uses pre-tuned Optuna hyperparameters (no re-tuning).

Usage:
    python train_xgboost_field.py --fraction 0.50 --output-dir models/xgboost_field/frac_0.50
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from experiment_config import (
    FINAL_DATA_PATH,
    XGB_TUNER_DIR,
    SUBSAMPLE_SEED,
)
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)


def aggregate_field(df):
    """Aggregate pixel-level data to field level (mean features, mode label)."""
    y = df.groupby('fid')['crop_name_encoded'].agg(lambda x: x.mode()[0])
    df_feat = df.drop(columns=['crop_name_encoded', 'crop_name'], errors='ignore')
    X = (
        df_feat.groupby('fid')
        .mean(numeric_only=True)
        .drop(columns=['crop_id', 'SHAPE_AREA', 'SHAPE_LEN'], errors='ignore')
    )
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    fraction = args.fraction
    print(f"=== XGBoost Field Training === Fraction: {fraction}")

    # Load data
    print("Loading data...")
    data = pd.read_parquet(FINAL_DATA_PATH, engine="pyarrow")

    # Label encode
    le = LabelEncoder()
    data['crop_name_encoded'] = le.fit_transform(data['crop_name'])

    # FID split (same as original xg_boost_hyper.py)
    train_fids, val_fids, test_fids = get_fid_split_dl(data)

    # Subsample train FIDs
    sub_train_fids = subsample_train_fids(data, train_fids, fraction, seed=SUBSAMPLE_SEED)
    print(f"Train FIDs: {len(train_fids)} -> {len(sub_train_fids)} ({fraction*100:.0f}%)")

    train_data = data[data['fid'].isin(sub_train_fids)].copy()
    val_data = data[data['fid'].isin(val_fids)].copy()
    test_data = data[data['fid'].isin(test_fids)].copy()

    # Aggregate to field level
    print("Aggregating per field...")
    X_train, y_train = aggregate_field(train_data)
    X_val, y_val = aggregate_field(val_data)
    X_test, y_test = aggregate_field(test_data)
    print(f"Fields: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Impute + scale
    X_train = X_train.dropna(axis=1, how='all')
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    imputer = SimpleImputer(strategy='mean')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)

    # Load pre-tuned params
    params_path = os.path.join(XGB_TUNER_DIR, 'best_xgb_params.joblib')
    best_params = joblib.load(params_path)
    best_params['n_estimators'] = 2000
    print(f"Loaded pre-tuned params from {params_path}")

    # Train
    print("Training XGBoost...")
    t_train = time.time()
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        device="cuda",
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        **best_params,
    )
    model.fit(
        X_train_scaled, y_train,
        sample_weight=compute_sample_weight('balanced', y_train),
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )
    train_time = time.time() - t_train
    print(f"Best iteration: {model.best_iteration}, training time: {train_time:.1f}s")

    # Evaluate on internal test set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average='macro')
    f1_w = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Test: acc={acc:.4f}, f1_macro={f1_m:.4f}, f1_weighted={f1_w:.4f}, kappa={kappa:.4f}")

    # Save artifacts
    joblib.dump(model, os.path.join(args.output_dir, 'xgboost_model.joblib'), compress=3)
    joblib.dump(imputer, os.path.join(args.output_dir, 'imputer.joblib'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(args.output_dir, 'label_encoder.joblib'))
    joblib.dump(list(X_train.columns), os.path.join(args.output_dir, 'feature_columns.joblib'))

    metadata = {
        "model": "xgboost_field",
        "fraction": fraction,
        "train_fields": int(len(sub_train_fids)),
        "val_fields": int(len(X_val)),
        "test_fields": int(len(X_test)),
        "best_iteration": int(model.best_iteration),
        "training_time_sec": round(train_time, 1),
        "metrics": {
            "accuracy": round(acc, 6),
            "f1_macro": round(f1_m, 6),
            "f1_weighted": round(f1_w, 6),
            "cohen_kappa": round(kappa, 6),
        },
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to {args.output_dir}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
