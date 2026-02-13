"""
TabNet ensemble on field-level xr_fresh time-series features.

Same data as XGBoost/SMOTE/Voting/Stacking (final_data.parquet aggregated per field),
but using TabNet architecture instead of tree-based models.

5-seed ensemble with averaged probabilities.

Input: data/final_data.parquet (pixel-level xr_fresh features)
Output: saved_models_tabnet_field/ (5 models + preprocessing artifacts)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import FINAL_DATA_PATH, TABNET_FIELD_DIR

sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from pytorch_tabnet.tab_model import TabNetClassifier
from report import ModelReport

SEEDS = [42, 101, 202, 303, 404]


def aggregate_field(df):
    """Aggregate pixel-level data to field-level: mean for features, mode for label."""
    y = df.groupby('fid')['crop_name_encoded'].agg(lambda x: x.mode()[0])
    df_features = df.drop(columns=['crop_name_encoded', 'crop_name'], errors='ignore')
    X = df_features.groupby('fid').mean(numeric_only=True).drop(
        columns=['crop_id', 'SHAPE_AREA', 'SHAPE_LEN'], errors='ignore'
    )
    return X, y


def main():
    t0 = time.time()
    os.makedirs(TABNET_FIELD_DIR, exist_ok=True)

    # =================== Load Data ===================
    print("[TIMER] Loading data...")
    data = pd.read_parquet(FINAL_DATA_PATH, engine="pyarrow")
    print(f"[TIMER] Data loaded: {len(data)} rows, {data.shape[1]} cols, {time.time()-t0:.1f}s")

    # Label encode
    le = LabelEncoder()
    data['crop_name_encoded'] = le.fit_transform(data['crop_name'])
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
    X_train, y_train = aggregate_field(train_data)
    X_val, y_val = aggregate_field(val_data)
    X_test, y_test = aggregate_field(test_data)
    print(f"Fields: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # =================== Impute + Scale ===================
    print("Preprocessing features...")
    X_train = X_train.dropna(axis=1, how='all')
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    feature_columns = list(X_train.columns)

    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_imp).astype(np.float32)

    # Handle NaN/inf from scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    print(f"Features: {len(feature_columns)}")

    # =================== 5-Seed Ensemble ===================
    val_preds_all, test_preds_all = [], []

    for i, seed in enumerate(SEEDS):
        print(f"\n=== Model {i+1}/{len(SEEDS)} (seed={seed}) ===")
        t_model = time.time()
        model_path = os.path.join(TABNET_FIELD_DIR, f"tabnet_field_seed_{seed}")

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

        if os.path.exists(model_path):
            print(f"Loading saved model for seed {seed}...")
            model.load_model(model_path)
            print(f"[TIMER] Model loaded in {time.time()-t_model:.1f}s")
        else:
            print(f"Training model for seed {seed}...")
            t_train = time.time()
            model.fit(
                X_train=X_train_scaled, y_train=y_train_arr,
                eval_set=[(X_val_scaled, y_val_arr)],
                eval_metric=["accuracy"],
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
    joblib.dump(le, os.path.join(TABNET_FIELD_DIR, "tabnet_field_label_encoder.joblib"))
    joblib.dump(imputer, os.path.join(TABNET_FIELD_DIR, "tabnet_field_imputer.joblib"))
    joblib.dump(scaler, os.path.join(TABNET_FIELD_DIR, "tabnet_field_scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(TABNET_FIELD_DIR, "tabnet_field_feature_columns.joblib"))
    print(f"Saved preprocessing artifacts to {TABNET_FIELD_DIR}")

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
    report = ModelReport("TabNet Field-Level (xr_fresh)")
    report.set_hyperparameters({
        "n_d": 64, "n_a": 64, "n_steps": 5,
        "gamma": 1.5, "n_independent": 2, "n_shared": 2,
        "lr": 1e-3, "scheduler": "StepLR(step=10, gamma=0.9)",
        "max_epochs": 100, "patience": 10, "batch_size": 1024,
        "n_models": len(SEEDS), "seeds": SEEDS,
        "features": "xr_fresh time-series (field-level mean)",
    })
    report.set_split_info(
        train=len(X_train), val=len(X_val), test=len(X_test),
        seed=42, split_method="fid-wise (field-level)",
    )
    report.set_metrics(y_test_arr, y_test_pred, list(le.classes_))
    report.generate()

    print(f"\n[TIMER] Total: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
