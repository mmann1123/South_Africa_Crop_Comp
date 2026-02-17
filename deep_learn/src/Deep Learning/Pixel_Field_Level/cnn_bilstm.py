# ==================== Step 1: Imports ====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, MODEL_DIR

sys.stdout.reconfigure(line_buffering=True)

import time
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
import random
import joblib
from report import ModelReport

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"[INFO] Libraries imported. Using device: {device}")

# ==================== Step 2: Dataset Loading & Split ====================
print("[INFO] Loading dataset...")
df = pd.read_parquet(MERGED_DL_PATH)

print("[INFO] Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['crop_name'])
num_classes = len(label_encoder.classes_)

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
# Drop columns that are entirely NaN (May has no Sentinel-2 data due to cloud cover)
feature_cols = [col for col in feature_cols if not df[col].isna().all()]
# Fill any remaining sparse NaN values with 0
df[feature_cols] = df[feature_cols].fillna(0)

train_dataset = CropDataset(train_df, feature_cols)
val_dataset = CropDataset(val_df, feature_cols)
test_dataset = CropDataset(test_df, feature_cols)

print("[INFO] Dataset classes created.")

# ==================== Step 5: Weighted Focal Loss ====================
class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss — class weights (alpha) + difficulty focusing (gamma)."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha[target]
        loss = alpha_t * ((1 - pt) ** self.gamma) * ce
        return loss.mean()

class_counts = np.bincount(train_df['label'], minlength=num_classes).astype(np.float64)
class_counts = np.maximum(class_counts, 1.0)
alpha = 1.0 / class_counts
alpha = alpha / alpha.sum() * num_classes
alpha = torch.tensor(alpha, dtype=torch.float32)
print(f"[INFO] Class weights (alpha): {alpha.tolist()}")

# ==================== Step 6: CNN + BiLSTM Model ====================
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

print("[INFO] CNN + BiLSTM model class defined.")

# ==================== Step 7: Training & Evaluation ====================
scaler_amp = torch.amp.GradScaler("cuda")

def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(X)
            loss = criterion(outputs, y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def get_logits_and_labels(model, dataloader):
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            outputs = model(X)
            logits_list.append(outputs.cpu())
            labels_list.extend(y.tolist())
    return torch.cat(logits_list, dim=0), labels_list

t_train_start = time.time()
print("[INFO] Starting ensemble with CNN + BiLSTM + Weighted Focal Loss...")
num_models = 5
PATIENCE = 15
N_EPOCHS = 100
logits_ensemble = []

val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

for seed in range(num_models):
    print(f"\n[ENSEMBLE] Training model with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = CropCNNBiLSTM(len(feature_cols), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = WeightedFocalLoss(alpha, gamma=2.0).to(device)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model_path = os.path.join(MODEL_DIR, f"new_model_seed_{seed}_25epochs.pt")
    torch.save(model.state_dict(), model_path)  # initial save

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        t_ep = time.time()
        loss = train_epoch(model, optimizer, criterion, train_loader)

        # Validation — select by F1 macro
        val_logits, val_labels = get_logits_and_labels(model, val_loader)
        val_preds = val_logits.argmax(dim=1).tolist()
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"[SEED {seed}] Epoch {epoch+1}/{N_EPOCHS} - "
              f"loss={loss:.4f}, val_f1_macro={val_f1:.4f}, "
              f"val_acc={val_acc:.4f}, {time.time()-t_ep:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[SEED {seed}] Early stopping at epoch {epoch+1}")
                break

    # Load best model for evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))
    logits, test_labels = get_logits_and_labels(model, test_loader)
    print(f"[SEED {seed}] Model saved: {model_path}")
    logits_ensemble.append(logits.unsqueeze(0))

# Save label encoder and feature columns for inference
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "cnn_bilstm_label_encoder.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "cnn_bilstm_feature_cols.pkl"))
print(f"[INFO] Saved label_encoder and feature_cols to {MODEL_DIR}")

print("\n[INFO] Averaging ensemble logits...")
avg_logits = torch.cat(logits_ensemble, dim=0).float().mean(dim=0)
pred_labels = torch.argmax(avg_logits, dim=1).tolist()

print("[ENSEMBLE + CNN+BiLSTM + WeightedFocal] Accuracy:", accuracy_score(test_labels, pred_labels))
print("[ENSEMBLE + CNN+BiLSTM + WeightedFocal] F1 Score:", f1_score(test_labels, pred_labels, average='weighted'))
print("[ENSEMBLE + CNN+BiLSTM + WeightedFocal] F1 Macro:", f1_score(test_labels, pred_labels, average='macro'))
print("[ENSEMBLE + CNN+BiLSTM + WeightedFocal] Cohen Kappa:", cohen_kappa_score(test_labels, pred_labels))
print("[ENSEMBLE + CNN+BiLSTM + WeightedFocal] Classification Report:\n", classification_report(test_labels, pred_labels, target_names=label_encoder.classes_))

# ===================== REPORT =====================
report = ModelReport("CNN-BiLSTM Ensemble (5-seed)")
report.set_hyperparameters({
    "num_models": num_models,
    "conv_filters": 64,
    "lstm_hidden": 64,
    "kernel_size": 5,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": N_EPOCHS,
    "patience": PATIENCE,
    "batch_size": 1024,
    "loss": "WeightedFocalLoss(gamma=2.0, alpha=1/class_counts)",
    "model_selection": "val F1 macro (maximize)",
    "gradient_clipping": 1.0,
})
report.set_split_info(train=len(train_df), val=len(val_df), test=len(test_df), seed=42)
report.set_metrics(test_labels, pred_labels, label_encoder.classes_)
report.set_training_time(time.time() - t_train_start)
report.generate()
