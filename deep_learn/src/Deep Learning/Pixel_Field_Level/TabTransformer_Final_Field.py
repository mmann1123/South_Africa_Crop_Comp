import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, TABNET_DIR

# Force unbuffered stdout so prints appear immediately in subprocess
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import torch
import random
import time
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# =================== Step 1: Load Data ===================
t0 = time.time()
print("[TIMER] Loading data...")
df = pd.read_parquet(MERGED_DL_PATH)
df = df.drop(columns=['May'], errors='ignore')
print(f"[TIMER] Data loaded: {len(df)} rows, {time.time()-t0:.1f}s")

# =================== Step 2: Preprocess ===================
t1 = time.time()
print("[TIMER] Preprocessing...")
exclude_cols = {'id', 'point', 'fid', 'crop_id', 'crop_name'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
# Fill NaN with median, then fill remaining NaN (all-NaN columns) with 0
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[numeric_cols] = df[numeric_cols].fillna(0)

if 'Type' in df.columns:
    df = pd.get_dummies(df, columns=['Type'])

one_hot_cols = [col for col in df.columns if col.startswith('Type_')]
feature_columns = numeric_cols + one_hot_cols

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# Handle any NaN/inf from zero-variance columns after scaling
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

features = df[feature_columns].astype(np.float32)
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['crop_id'])
targets = df['crop_label'].values
print(f"[TIMER] Preprocessing done: {len(feature_columns)} features, {time.time()-t1:.1f}s")

# =================== Step 3: Fid-Wise Split ===================
t2 = time.time()
unique_fids = df['fid'].unique()
trainval_fids, test_fids = train_test_split(unique_fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(trainval_fids, test_size=0.2, random_state=42)

train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)
test_mask = df['fid'].isin(test_fids)

X_train, y_train = features[train_mask].values, targets[train_mask]
X_val, y_val = features[val_mask].values, targets[val_mask]
X_test, y_test = features[test_mask].values, targets[test_mask]
print(f"[TIMER] Split done: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, {time.time()-t2:.1f}s")

# =================== Step 4: Ensemble of TabTransformer ===================
n_models = 5
seeds = [42, 101, 202, 303, 404]
val_preds_all, test_preds_all = [], []
model_dir = TABNET_DIR
os.makedirs(model_dir, exist_ok=True)

for i, seed in enumerate(seeds):
    print(f"\n[TIMER] === Model {i+1}/{n_models} (seed={seed}) ===")
    t_model = time.time()
    model_path = os.path.join(model_dir, f"tabnet_seed_{seed}")
    model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=seed,
        verbose=1
    )

    if os.path.exists(model_path):
        print(f"🔁 Loading saved model for seed {seed}...")
        model.load_model(model_path)
        print(f"[TIMER] Model loaded in {time.time()-t_model:.1f}s")
    else:
        print(f"🚀 Training new model for seed {seed}...")
        t_train = time.time()
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        print(f"[TIMER] Training done in {time.time()-t_train:.1f}s")
        model.save_model(model_path)
        print(f"Saved model to {model_path}")

    t_pred = time.time()
    print(f"[TIMER] Predicting on val ({len(X_val)} samples)...")
    val_preds_all.append(model.predict_proba(X_val))
    print(f"[TIMER] Val prediction: {time.time()-t_pred:.1f}s")
    t_pred2 = time.time()
    print(f"[TIMER] Predicting on test ({len(X_test)} samples)...")
    test_preds_all.append(model.predict_proba(X_test))
    print(f"[TIMER] Test prediction: {time.time()-t_pred2:.1f}s")
    print(f"[TIMER] Model {i+1} total: {time.time()-t_model:.1f}s")

# Save preprocessing artifacts for inference
joblib.dump(scaler, os.path.join(model_dir, "tabnet_scaler.joblib"))
joblib.dump(feature_columns, os.path.join(model_dir, "tabnet_feature_columns.joblib"))
joblib.dump(label_encoder, os.path.join(model_dir, "tabnet_label_encoder.joblib"))
print(f"Saved preprocessing artifacts to {model_dir}")

# =================== Step 5: Aggregated Predictions ===================
print(f"\n[TIMER] Aggregating ensemble predictions...")
t_agg = time.time()
val_pred_mean = np.mean(val_preds_all, axis=0)
test_pred_mean = np.mean(test_preds_all, axis=0)

y_val_pred = np.argmax(val_pred_mean, axis=1)
y_test_pred = np.argmax(test_pred_mean, axis=1)

print(f"[TIMER] Ensemble aggregation done: {time.time()-t_agg:.1f}s")

# =================== Step 6: Field-Level Aggregation ===================
print("[TIMER] Field-level aggregation...")
t_field = time.time()
def aggregate_field_preds(df_subset, y_preds, label_col='crop_label'):
    pred_df = pd.DataFrame({
        'fid': df_subset['fid'].values,
        'pred_label': y_preds,
        'true_label': df_subset[label_col].values
    })
    field_pred = pred_df.groupby('fid')['pred_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
    field_true = pred_df.groupby('fid')['true_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
    return field_true, field_pred

train_field_true, train_field_pred = aggregate_field_preds(df[train_mask], targets[train_mask])
val_field_true, val_field_pred = aggregate_field_preds(df[val_mask], y_val_pred)
test_field_true, test_field_pred = aggregate_field_preds(df[test_mask], y_test_pred)
print(f"[TIMER] Field-level aggregation done: {time.time()-t_field:.1f}s")

# =================== Step 7: Evaluation Function ===================
def evaluate_field_level(true_labels, pred_labels, title="Confusion Matrix"):
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("F1 Score:", f1_score(true_labels, pred_labels, average='weighted'))
    print("Cohen Kappa:", cohen_kappa_score(true_labels, pred_labels))
    print("Classification Report:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.close()

# =================== Step 8: Results ===================
print("========= [Train Field-Level Evaluation] =========")
evaluate_field_level(train_field_true, train_field_pred, title="Train Confusion Matrix")

print("========= [Validation Field-Level Evaluation] =========")
evaluate_field_level(val_field_true, val_field_pred, title="Validation Confusion Matrix")

print("========= [Test Field-Level Evaluation] =========")
evaluate_field_level(test_field_true, test_field_pred, title="Test Confusion Matrix")

# ===================== REPORT =====================
from report import ModelReport

report = ModelReport("TabTransformer Ensemble (Field-Level)")
report.set_hyperparameters({
    "n_d": 64, "n_a": 64, "n_steps": 5,
    "gamma": 1.5, "n_independent": 2, "n_shared": 2,
    "lr": 1e-3, "scheduler": "StepLR(step=10, gamma=0.9)",
    "max_epochs": 100, "patience": 10, "batch_size": 1024,
    "n_models": n_models, "seeds": seeds,
})
report.set_split_info(
    train=int(train_mask.sum()), val=int(val_mask.sum()), test=int(test_mask.sum()),
    seed=42,
)
report.set_metrics(test_field_true.values, test_field_pred.values,
                   [str(c) for c in label_encoder.classes_])
report.set_predictions(test_field_pred.index, test_field_true.values, test_field_pred.values,
                       [str(c) for c in label_encoder.classes_])
report.generate()
print(f"\n[TIMER] Total script time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
