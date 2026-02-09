# ==================== Step 1: Imports ====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Libraries imported. Using device: {device}")

# ==================== Step 2: Dataset Loading & Split ====================
print("[INFO] Loading dataset...")
df = pd.read_parquet(MERGED_DL_PATH)

print("[INFO] Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['crop_name'])

fids = df['fid'].unique()
train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.1, random_state=42)

train_df = df[df['fid'].isin(train_fids)].reset_index(drop=True)
val_df = df[df['fid'].isin(val_fids)].reset_index(drop=True)
test_df = df[df['fid'].isin(test_fids)].reset_index(drop=True)

print("[INFO] Train/Val/Test split complete.")

# ==================== Step 3: Dataset Class ====================
class CropDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ==================== Step 4: Feature Columns ====================
feature_cols = [col for col in df.columns if any(b in col for b in ['B2_', 'B6_', 'B11_', 'B12_', 'hue_', 'EVI_'])]
feature_cols = [col for col in feature_cols if not df[col].isna().all()]
df[feature_cols] = df[feature_cols].fillna(0)

train_dataset = CropDataset(train_df, feature_cols)
val_dataset = CropDataset(val_df, feature_cols)
test_dataset = CropDataset(test_df, feature_cols)

print("[INFO] Dataset classes created.")

# ==================== Step 5: Weighted Sampler ====================
class_counts = np.bincount(train_df['label'])
class_weights = 1. / class_counts
sample_weights = class_weights[train_df['label'].values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

print("[INFO] Weighted sampler initialized.")

# ==================== Step 6: Model Definition ====================
class CropCNN1D(nn.Module):
    def __init__(self, input_size, num_classes, conv_filters=64, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(6, conv_filters, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_filters, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 6, -1)  # Reshape: (B, Channels, Time)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

print("[INFO] CNN model class defined.")

# ==================== Step 7: Training Utilities ====================
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

def evaluate(model, criterion, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ==================== Step 8: Optuna Tuning ====================
def objective(trial):
    print("[INFO] Starting a new trial...")
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    conv_filters = trial.suggest_categorical('conv_filters', [32, 64, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])

    print(f"[INFO] Trial {trial.number} Hyperparameters: lr={lr}, dropout={dropout}, conv_filters={conv_filters}, kernel_size={kernel_size}")

    model = CropCNN1D(len(feature_cols), len(label_encoder.classes_), conv_filters, kernel_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    for epoch in range(25):
        loss = train(model, optimizer, criterion, train_loader)
        print(f"[Trial {trial.number}] Epoch {epoch+1}/25 - Loss: {loss:.4f}")

    val_acc = evaluate(model, criterion, val_loader)
    print(f"[Trial {trial.number}] Validation Accuracy: {val_acc:.4f}")
    return val_acc

print("[INFO] Starting hyperparameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

print("[INFO] Best Hyperparameters:", study.best_params)
