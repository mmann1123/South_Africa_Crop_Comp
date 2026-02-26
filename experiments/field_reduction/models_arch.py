"""Shared model architectures for the field reduction experiment.

Extracted verbatim from the original training scripts:
- LTAE, PositionalEncoding, WeightedFocalLoss, TemporalDataset from ltae_field.py / ltae_model.py
- F1MacroMetric from TabTransformer_Final_Field.py
- train_epoch, evaluate, aggregate_field_preds helpers
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
from sklearn.metrics import f1_score

# =================== Constants ===================

BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
MONTHS_CHRONO = [
    "January", "February", "March", "April",
    "July", "August", "September",
    "October", "November", "December",
]
MONTH_POSITIONS = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
T_SEQ = len(MONTHS_CHRONO)  # 10
N_BANDS = len(BANDS)         # 6


# =================== Feature helpers ===================

def get_chrono_feature_cols(df):
    """Build feature column list in chronological order: [B2_Jan, B6_Jan, ..., hue_Dec]."""
    cols = []
    for month in MONTHS_CHRONO:
        for band in BANDS:
            col = f"{band}_{month}"
            if col in df.columns:
                cols.append(col)
    return cols


# =================== Datasets ===================

class TemporalDataset(Dataset):
    """Dataset returning (T, C) temporal tensors for L-TAE."""

    def __init__(self, X, y):
        self.X = torch.tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =================== Weighted Focal Loss ===================

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


# =================== L-TAE Architecture ===================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with actual month positions."""

    def __init__(self, d_model, positions=None):
        super().__init__()
        if positions is None:
            positions = list(range(T_SEQ))
        pe = torch.zeros(len(positions), d_model)
        pos = torch.tensor(positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder."""

    def __init__(self, in_channels=6, d_model=128, n_head=16, d_k=8,
                 dropout=0.3, num_classes=5):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, positions=MONTH_POSITIONS)
        self.query = nn.Parameter(torch.randn(n_head, d_k))
        nn.init.normal_(self.query, mean=0, std=0.5)
        self.key_proj = nn.Linear(d_model, n_head * d_k)
        self.value_proj = nn.Linear(d_model, n_head * d_k)
        self.attention_dropout = nn.Dropout(dropout)

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
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C = x.shape
        x = self.embedding(x)
        x = self.pos_enc(x)
        keys = self.key_proj(x).view(B, T, self.n_head, self.d_k)
        values = self.value_proj(x).view(B, T, self.n_head, self.d_k)
        keys = keys.permute(2, 0, 1, 3)
        values = values.permute(2, 0, 1, 3)
        query = self.query.unsqueeze(1).unsqueeze(2).expand(-1, B, -1, -1)
        attn = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)
        out = torch.matmul(attn, values).squeeze(2).permute(1, 0, 2).reshape(B, -1)
        out = self.norm(out)
        out = self.mlp(out)
        return self.classifier(out)


# =================== TabNet metric ===================

try:
    from pytorch_tabnet.metrics import Metric

    class F1MacroMetric(Metric):
        """F1 Macro metric for TabNet eval_metric."""
        def __init__(self):
            self._name = "f1_macro"
            self._maximize = True

        def __call__(self, y_true, y_score):
            preds = np.argmax(y_score, axis=1)
            return f1_score(y_true, preds, average='macro')

except ImportError:
    F1MacroMetric = None


# =================== Training helpers ===================

def get_device():
    """Return the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, optimizer, criterion, dataloader, scaler_amp, device):
    """Run one training epoch with mixed precision."""
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
def evaluate(model, dataloader, device):
    """Evaluate model, returning logits and labels."""
    model.eval()
    logits_all, labels_all = [], []
    with torch.amp.autocast("cuda"):
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            logits_all.append(model(X).float().cpu())
            labels_all.extend(y.tolist())
    return torch.cat(logits_all, dim=0), labels_all


def aggregate_field_preds(fids, y_true, y_pred):
    """Majority vote per field."""
    df = pd.DataFrame({"fid": fids, "true": y_true, "pred": y_pred})
    field_true = df.groupby("fid")["true"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    field_pred = df.groupby("fid")["pred"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    return field_true, field_pred


def compute_focal_loss_weights(y_train, num_classes):
    """Compute class weights for WeightedFocalLoss from training labels."""
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum() * num_classes
    return torch.tensor(alpha, dtype=torch.float32)
