# ==================== Step 1: Imports ====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, MODEL_DIR

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==================== Step 2: Dataset Loading & Split ====================
df = pd.read_parquet(MERGED_DL_PATH)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['crop_name'])

fids = df['fid'].unique()
train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.1, random_state=42)

train_df = df[df['fid'].isin(train_fids)].reset_index(drop=True)
val_df = df[df['fid'].isin(val_fids)].reset_index(drop=True)
test_df = df[df['fid'].isin(test_fids)].reset_index(drop=True)

# ==================== Step 3: Dataset Class ====================
feature_cols = [col for col in df.columns if any(b in col for b in ['B2_', 'B6_', 'B11_', 'B12_', 'hue_', 'EVI_'])]
feature_cols = [col for col in feature_cols if not df[col].isna().all()]
df[feature_cols] = df[feature_cols].fillna(0)

class CropDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df['label'].values.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

train_dataset = CropDataset(train_df, feature_cols)
val_dataset = CropDataset(val_df, feature_cols)
test_dataset = CropDataset(test_df, feature_cols)

# ==================== Step 4: Sampler ====================
class_counts = np.bincount(train_df['label'])
class_weights = 1. / class_counts
sample_weights = class_weights[train_df['label'].values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ==================== Step 5: Model Definition ====================
class CropCNN1D(nn.Module):
    def __init__(self, input_size, num_classes, conv_filters=64, kernel_size=7, dropout=0.4939):
        super().__init__()
        self.conv1 = nn.Conv1d(6, conv_filters, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_filters, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), 6, -1)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# ==================== Step 6: Train & Evaluate ====================
def train(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_preds

# ==================== Step 7: Train Final Model ====================
model = CropCNN1D(len(feature_cols), len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001162)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

for epoch in range(50):
    loss = train(model, optimizer, criterion, train_loader)
    print(f"[FINAL MODEL] Epoch {epoch+1}/50 - Loss: {loss:.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "crop_cnn1d_model.pt"))

# ==================== Step 8: Evaluation ====================
train_labels, train_preds = evaluate_model(model, DataLoader(train_dataset, batch_size=128))
val_labels, val_preds = evaluate_model(model, val_loader)
test_labels, test_preds = evaluate_model(model, test_loader)

def print_metrics(y_true, y_pred, name):
    print(f"\n====== {name.upper()} RESULTS ======")
    print(f" Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    print(f" Cohen Kappa : {cohen_kappa_score(y_true, y_pred):.4f}")
    print(" Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

print_metrics(train_labels, train_preds, "Train")
print_metrics(val_labels, val_preds, "Validation")
print_metrics(test_labels, test_preds, "Test")

# ==================== Step 9: Field-Level Aggregation & Confusion Matrix ====================
def aggregate_field_predictions(df, preds):
    pred_df = pd.DataFrame({'fid': df['fid'], 'pred_label': preds, 'true_label': df['label'].values})
    field_preds = pred_df.groupby('fid')['pred_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
    field_true = pred_df.groupby('fid')['true_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
    return field_true, field_preds

def plot_confusion_matrix(true, pred, title, filename):
    cm = confusion_matrix(true, pred)
    target_names = label_encoder.classes_
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.xlabel("Predicted Crop")
    plt.ylabel("True Crop")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to: {filename}")

train_field_true, train_field_pred = aggregate_field_predictions(train_df, train_preds)
val_field_true, val_field_pred = aggregate_field_predictions(val_df, val_preds)
test_field_true, test_field_pred = aggregate_field_predictions(test_df, test_preds)

print_metrics(train_field_true, train_field_pred, "Train Field-Level")
print_metrics(val_field_true, val_field_pred, "Validation Field-Level")
print_metrics(test_field_true, test_field_pred, "Test Field-Level")

plot_confusion_matrix(train_field_true, train_field_pred, "Train Field-Level Confusion", "train_field_confusion.png")
plot_confusion_matrix(val_field_true, val_field_pred, "Validation Field-Level Confusion", "val_field_confusion.png")
plot_confusion_matrix(test_field_true, test_field_pred, "Test Field-Level Confusion", "test_field_confusion.png")
