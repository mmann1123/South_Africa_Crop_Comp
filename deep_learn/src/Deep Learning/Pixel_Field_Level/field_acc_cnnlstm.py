# ==================== Step 1: Imports ====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, MODEL_DIR

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torchtoolbox.nn import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Libraries imported. Using device: {device}")

# ==================== Step 2: Dataset Loading ====================
df = pd.read_parquet(MERGED_DL_PATH)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['crop_name'])

feature_cols = [col for col in df.columns if any(b in col for b in ['B2_', 'B6_', 'B11_', 'B12_', 'hue_', 'EVI_'])]
feature_cols = [col for col in feature_cols if not df[col].isna().all()]
df[feature_cols] = df[feature_cols].fillna(0)
print("[INFO] Dataset loaded and labels encoded.")

# ==================== Step 3: Dataset ====================
class CropDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.X = df[feature_cols].values.astype(np.float32)
        self.fids = df['fid'].values

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), self.fids[idx]

# ==================== Step 4: Model Definition ====================
class CropCNNBiLSTM(nn.Module):
    def __init__(self, input_size, num_classes, conv_filters=64, lstm_hidden=64, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(6, conv_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 6, -1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

# ==================== Step 5: Load Models & Predict ====================
print("[INFO] Loading saved models...")
models = []
for seed in range(5):
    model = CropCNNBiLSTM(len(feature_cols), df['label'].nunique()).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"new_model_seed_{seed}_25epochs.pt"), map_location=device))
    model.eval()
    models.append(model)
print("[INFO] All models loaded.")

# ==================== Step 6: Prepare Data ====================
fids = df['fid'].unique()
train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.1, random_state=42)
val_df = df[df['fid'].isin(val_fids)].reset_index(drop=True)
test_df = df[df['fid'].isin(test_fids)].reset_index(drop=True)

val_dataset = CropDataset(val_df, feature_cols)
test_dataset = CropDataset(test_df, feature_cols)

val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ==================== Step 7: Predict on Validation and Test ====================
val_preds, val_fids_list = [], []
all_preds, all_fids = [], []

with torch.no_grad():
    for X_batch, fid_batch in val_loader:
        X_batch = X_batch.to(device)
        logits_ensemble = torch.stack([m(X_batch) for m in models])
        mean_logits = logits_ensemble.mean(dim=0)
        preds = torch.argmax(mean_logits, dim=1)
        val_preds.extend(preds.cpu().tolist())
        val_fids_list.extend(fid_batch.tolist())

    for X_batch, fid_batch in test_loader:
        X_batch = X_batch.to(device)
        logits_ensemble = torch.stack([m(X_batch) for m in models])
        mean_logits = logits_ensemble.mean(dim=0)
        preds = torch.argmax(mean_logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_fids.extend(fid_batch.tolist())

# ==================== Step 8: Aggregate Predictions to Field-Level ====================
val_preds_df = pd.DataFrame({'fid': val_fids_list, 'pred_label': val_preds})
val_fid_label_map = val_df.groupby('fid')['label'].agg(lambda x: Counter(x).most_common(1)[0][0])
val_field_preds = val_preds_df.groupby('fid')['pred_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
val_field_true = val_fid_label_map.loc[val_field_preds.index]

pixel_preds_df = pd.DataFrame({'fid': all_fids, 'pred_label': all_preds})
fid_label_map = test_df.groupby('fid')['label'].agg(lambda x: Counter(x).most_common(1)[0][0])
field_preds = pixel_preds_df.groupby('fid')['pred_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
field_true = fid_label_map.loc[field_preds.index]

# ==================== Step 9: Save Test Set Predictions ====================
results_df = pd.DataFrame({
    'field_id': field_preds.index,
    'true_crop_name': label_encoder.inverse_transform(field_true.values),
    'predicted_crop_name': label_encoder.inverse_transform(field_preds.values)
})
results_df.to_csv(os.path.join(MODEL_DIR, "test_field_level_predictions_latest.csv"), index=False)
print("[INFO] Saved test field-level predictions to 'test_field_level_predictions.csv'")

# ==================== Step 10: Evaluation ====================
print("\n[VALIDATION FIELD-LEVEL EVALUATION]")
print("Accuracy:", accuracy_score(val_field_true, val_field_preds))
print("F1 Score:", f1_score(val_field_true, val_field_preds, average='weighted'))
print("Cohen Kappa:", cohen_kappa_score(val_field_true, val_field_preds))
print("Classification Report:\n", classification_report(val_field_true, val_field_preds, target_names=label_encoder.classes_))

val_cm = confusion_matrix(val_field_true, val_field_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Greens', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Validation Field-Level Confusion Matrix')
plt.tight_layout()
plt.close()

print("\n[TEST FIELD-LEVEL EVALUATION]")
print("Accuracy:", accuracy_score(field_true, field_preds))
print("F1 Score:", f1_score(field_true, field_preds, average='weighted'))
print("Cohen Kappa:", cohen_kappa_score(field_true, field_preds))
print("Classification Report:\n", classification_report(field_true, field_preds, target_names=label_encoder.classes_))

cm = confusion_matrix(field_true, field_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Field-Level Confusion Matrix')
plt.tight_layout()
plt.close()

# ===================== REPORT =====================
from report import ModelReport

report = ModelReport("CNN-BiLSTM Ensemble Field-Level (Inference)")
report.set_hyperparameters({
    "num_models": 5,
    "conv_filters": 64,
    "lstm_hidden": 64,
    "kernel_size": 5,
    "dropout": 0.3,
    "aggregation": "majority vote by FID",
})
report.set_split_info(train=len(train_fids), val=len(val_field_true), test=len(field_true), seed=42, split_method="fid-wise (field-level aggregation)")
report.set_metrics(field_true, field_preds, label_encoder.classes_)
report.set_predictions(field_preds.index, field_true, field_preds, label_encoder.classes_)
report.generate()
