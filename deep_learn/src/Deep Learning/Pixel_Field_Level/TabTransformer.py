import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------------
# 1. Load Data
# --------------------------
print("Loading data...")
df = pd.read_parquet(MERGED_DL_PATH)
df = df.drop(columns=['May'], errors='ignore')

# --------------------------
# 2. Preprocess Data
# --------------------------
print("Preprocessing data...")
exclude_cols = {'id', 'point', 'fid', 'crop_id', 'crop_name'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

if 'Type' in df.columns:
    df = pd.get_dummies(df, columns=['Type'])

one_hot_cols = [col for col in df.columns if col.startswith('Type_')]
feature_columns = numeric_cols + one_hot_cols

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

features = df[feature_columns].astype(np.float32)
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['crop_id'])
targets = df['crop_label'].values

# --------------------------
# 3. Train/Val Split (no fid overlap)
# --------------------------
print("Splitting data...")
unique_fids = df['fid'].unique()
train_fids, val_fids = train_test_split(unique_fids, test_size=0.2, random_state=42)

train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)

X_train, y_train = features[train_mask].values, targets[train_mask]
X_val, y_val = features[val_mask].values, targets[val_mask]

# --------------------------
# 4. Ensemble of 5 TabNet Models
# --------------------------
n_models = 5
seeds = [42, 101, 202, 303, 404]
predictions = []

for idx, seed in enumerate(seeds):
    print(f"Training model {idx + 1}/{n_models} with seed {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["accuracy"],
        max_epochs=100,
        patience=30,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    preds = model.predict_proba(X_val)
    predictions.append(preds)

# --------------------------
# 5. Average Ensemble Predictions
# --------------------------
print("Averaging predictions from ensemble...")
ensemble_preds = np.mean(predictions, axis=0)
y_pred_ensemble = np.argmax(ensemble_preds, axis=1)

# --------------------------
# 6. Evaluation
# --------------------------
print("Evaluating ensemble predictions...")
acc = accuracy_score(y_val, y_pred_ensemble)
kappa = cohen_kappa_score(y_val, y_pred_ensemble)
print(f"Ensemble Accuracy: {acc:.4f} | Cohen Kappa: {kappa:.4f}")

# --------------------------
# 7. Confusion Matrix
# --------------------------
print("Generating confusion matrix...")
cm = confusion_matrix(y_val, y_pred_ensemble, normalize="true") * 100
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

plt.title("Confusion Matrix (TabNet Ensemble)")
plt.xlabel("Predicted Crop")
plt.ylabel("True Crop")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
