"""
L-TAE (Lightweight Temporal Attention Encoder) for crop classification.

Processes pixel-level Sentinel-2 time series with chronologically-ordered
temporal sequences and multi-head attention to learn which time steps
matter most for each crop class.

5-seed ensemble with averaged logits, field-level aggregation via majority vote.

Input: data/merged_dl_train.parquet (pixel-level, 6 bands x 10 months)
Output: models/ltae_seed_*.pt + preprocessing artifacts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, MODEL_DIR

sys.stdout.reconfigure(line_buffering=True)

import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import joblib
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from report import ModelReport

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Bands and months in chronological order
BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
MONTHS_CHRONO = [
    "January", "February", "March", "April",
    "July", "August", "September",
    "October", "November", "December",
]
# Month positions (1-indexed) for positional encoding — note gap at May(5), June(6)
MONTH_POSITIONS = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]

SEEDS = [42, 101, 202, 303, 404]
N_EPOCHS = 30
BATCH_SIZE = 2048
LR = 1e-3
PATIENCE = 15
T_SEQ = len(MONTHS_CHRONO)  # 10
N_BANDS = len(BANDS)         # 6


def get_chrono_feature_cols(df):
    """Build feature column list in chronological order: [B2_Jan, B6_Jan, ..., hue_Jan, B2_Feb, ...]."""
    cols = []
    for month in MONTHS_CHRONO:
        for band in BANDS:
            col = f"{band}_{month}"
            if col in df.columns:
                cols.append(col)
    return cols


class TemporalDataset(Dataset):
    """Dataset returning (T, C) temporal tensors."""

    def __init__(self, X, y):
        # X is (N, T*C) flat, reshape to (N, T, C)
        self.X = torch.tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =================== FocalLoss ===================

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) — down-weights well-classified examples."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# =================== L-TAE Architecture ===================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with actual month positions."""

    def __init__(self, d_model, positions=None):
        super().__init__()
        if positions is None:
            positions = list(range(T_SEQ))
        pe = torch.zeros(len(positions), d_model)
        pos = torch.tensor(positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder."""

    def __init__(self, in_channels=6, d_model=128, n_head=16, d_k=8, dropout=0.3, num_classes=5):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k

        # Input embedding: project spectral bands to d_model
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding with actual month positions
        self.pos_enc = PositionalEncoding(d_model, positions=MONTH_POSITIONS)

        # Multi-head attention components
        # Learnable master query per head
        self.query = nn.Parameter(torch.randn(n_head, d_k))
        nn.init.normal_(self.query, mean=0, std=0.5)

        self.key_proj = nn.Linear(d_model, n_head * d_k)
        self.value_proj = nn.Linear(d_model, n_head * d_k)

        self.attention_dropout = nn.Dropout(dropout)

        # Output MLP
        self.norm = nn.LayerNorm(n_head * d_k)
        self.mlp = nn.Sequential(
            nn.Linear(n_head * d_k, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, C=6)
        B, T, C = x.shape

        # Embed input
        x = self.embedding(x)  # (B, T, d_model)
        x = self.pos_enc(x)     # (B, T, d_model)

        # Multi-head attention
        keys = self.key_proj(x).view(B, T, self.n_head, self.d_k)     # (B, T, H, d_k)
        values = self.value_proj(x).view(B, T, self.n_head, self.d_k) # (B, T, H, d_k)

        keys = keys.permute(2, 0, 1, 3)   # (H, B, T, d_k)
        values = values.permute(2, 0, 1, 3) # (H, B, T, d_k)

        query = self.query.unsqueeze(1).unsqueeze(2)  # (H, 1, 1, d_k)
        query = query.expand(-1, B, -1, -1)            # (H, B, 1, d_k)

        # Attention scores
        attn = torch.matmul(query, keys.transpose(-2, -1))  # (H, B, 1, T)
        attn = attn / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, values)  # (H, B, 1, d_k)
        out = out.squeeze(2)               # (H, B, d_k)
        out = out.permute(1, 0, 2)         # (B, H, d_k)
        out = out.reshape(B, -1)           # (B, H*d_k)

        # MLP
        out = self.norm(out)
        out = self.mlp(out)  # (B, 128)

        return self.classifier(out)  # (B, num_classes)


# =================== Training ===================

def train_epoch(model, optimizer, criterion, dataloader, scaler_amp):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(X)
            loss = criterion(out, y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    logits_all, labels_all = [], []
    with torch.amp.autocast("cuda"):
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            logits_all.append(model(X).cpu())
            labels_all.extend(y.tolist())
    return torch.cat(logits_all, dim=0), labels_all


def aggregate_field_preds(fids, y_true, y_pred):
    """Majority vote per field."""
    df = pd.DataFrame({"fid": fids, "true": y_true, "pred": y_pred})
    field_true = df.groupby("fid")["true"].agg(lambda x: Counter(x).most_common(1)[0][0])
    field_pred = df.groupby("fid")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])
    return field_true, field_pred


def main():
    t0 = time.time()
    print(f"=== L-TAE Training === Device: {device}")

    # Load data
    print("[TIMER] Loading data...")
    df = pd.read_parquet(MERGED_DL_PATH)
    print(f"Shape: {df.shape}")

    # Feature columns in chronological order
    feature_cols = get_chrono_feature_cols(df)
    # Drop any that are all-NaN
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"Feature cols: {len(feature_cols)} (expected {T_SEQ * N_BANDS})")

    # Labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['crop_name'])
    num_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    # Fid-wise split
    fids = df['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)

    train_mask = df['fid'].isin(train_fids)
    val_mask = df['fid'].isin(val_fids)
    test_mask = df['fid'].isin(test_fids)

    print(f"Split: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

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

    # Datasets
    train_ds = TemporalDataset(X_train, y_train)
    val_ds = TemporalDataset(X_val, y_val)
    test_ds = TemporalDataset(X_test, y_test)

    # Weighted sampler for class imbalance (same as CNN-BiLSTM)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = (1.0 / class_counts)[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # FocalLoss — same as CNN-BiLSTM, no class weights needed
    criterion = FocalLoss(gamma=2.0).to(device)

    # 5-seed ensemble
    test_logits_ensemble = []

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = LTAE(in_channels=N_BANDS, d_model=128, n_head=16, d_k=8,
                     dropout=0.3, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
        scaler_amp = torch.amp.GradScaler("cuda")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, pin_memory=True, persistent_workers=True)

        best_val_loss = float('inf')
        patience_counter = 0
        model_path = os.path.join(MODEL_DIR, f"ltae_seed_{seed}.pt")

        # Save initial model so file always exists
        torch.save(model.state_dict(), model_path)

        for epoch in range(N_EPOCHS):
            t_ep = time.time()
            train_loss = train_epoch(model, optimizer, criterion, train_loader, scaler_amp)

            # Validation
            val_logits, val_labels = evaluate(model, val_loader)
            val_loss = F.cross_entropy(val_logits.float(), torch.tensor(val_labels)).item()
            val_acc = accuracy_score(val_labels, val_logits.argmax(dim=1).tolist())
            scheduler.step()

            print(f"  Epoch {epoch+1}/{N_EPOCHS} - "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}, "
                  f"{time.time()-t_ep:.1f}s")

            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best and evaluate
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_logits, test_labels = evaluate(model, test_loader)
        test_logits_ensemble.append(test_logits.unsqueeze(0))

        preds = test_logits.argmax(dim=1).tolist()
        print(f"  Test acc={accuracy_score(test_labels, preds):.4f}, "
              f"kappa={cohen_kappa_score(test_labels, preds):.4f}")

    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, "ltae_scaler.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "ltae_label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "ltae_feature_columns.joblib"))
    print(f"\nSaved artifacts to {MODEL_DIR}")

    # Ensemble evaluation
    avg_logits = torch.cat(test_logits_ensemble, dim=0).float().mean(dim=0)
    pred_labels = avg_logits.argmax(dim=1).tolist()

    # Field-level aggregation
    field_true, field_pred = aggregate_field_preds(test_fid_arr, test_labels, pred_labels)
    acc = accuracy_score(field_true, field_pred)
    f1 = f1_score(field_true, field_pred, average='weighted')
    kappa = cohen_kappa_score(field_true, field_pred)

    print(f"\n--- Ensemble Field-Level Results ---")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 weighted: {f1:.4f}")
    print(f"  Kappa:       {kappa:.4f}")
    print(classification_report(field_true, field_pred, target_names=le.classes_))

    # Report
    report = ModelReport("L-TAE Temporal Attention")
    report.set_hyperparameters({
        "architecture": "L-TAE (Lightweight Temporal Attention Encoder)",
        "d_model": 128, "n_head": 16, "d_k": 8,
        "dropout": 0.3, "lr": LR,
        "optimizer": "AdamW (wd=1e-4)",
        "scheduler": "CosineAnnealingLR",
        "loss": "FocalLoss(gamma=2.0)",
        "gradient_clipping": 1.0,
        "epochs": N_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "n_models": len(SEEDS), "seeds": SEEDS,
        "temporal_order": "chronological",
        "positional_encoding": "sinusoidal (actual month positions)",
    })
    report.set_split_info(
        train=int(train_mask.sum()), val=int(val_mask.sum()), test=int(test_mask.sum()),
        seed=42,
    )
    report.set_metrics(field_true.values, field_pred.values, list(le.classes_))
    report.generate()

    print(f"\n[TIMER] Total: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
