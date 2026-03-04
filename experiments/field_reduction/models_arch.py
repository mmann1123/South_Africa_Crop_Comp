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

# =================== Sparsemax ===================

def sparsemax(z, dim=-1):
    """Sparsemax activation (Martins & Astudillo, 2016).

    Projects input onto the probability simplex, producing exact zeros
    for small values (unlike softmax which is always dense).

    Args:
        z: input tensor of any shape
        dim: dimension along which to apply sparsemax

    Returns:
        Tensor of same shape with values in [0,1] summing to 1 along dim,
        with many exact zeros.
    """
    z = z - z.max(dim=dim, keepdim=True).values  # numerical stability
    z_sorted, _ = z.sort(dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)

    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    # Reshape k for broadcasting across all other dims
    shape = [1] * z.dim()
    shape[dim] = -1
    k = k.view(shape)

    support = (z_sorted - (z_cumsum - 1) / k) > 0
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1).float()
    # Gather the cumulative sum at the support boundary
    idx = (k_z - 1).long().clamp(min=0)
    tau = (z_cumsum.gather(dim, idx) - 1) / k_z
    return torch.clamp(z - tau, min=0)


# =================== L-TAE-S Components ===================

class SparseFeatureGate(nn.Module):
    """TabNet-inspired sparse channel selection gate for multi-head attention.

    For each head h, produces a sparse mask over the embedding dimension E
    using sparsemax. A prior scale mechanism discourages redundant selections
    across heads.

    Args:
        d_model: embedding dimension E
        n_head: number of attention heads
        gamma: prior scale coefficient (higher = more feature reuse allowed)
        time_varying: if True, gate varies per timestep; if False, uses time-averaged input
    """

    def __init__(self, d_model, n_head, gamma=1.5, time_varying=True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.gamma = gamma
        self.time_varying = time_varying

        self.gate_fc = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_head)
        ])
        self.gate_bn = nn.ModuleList([
            nn.BatchNorm1d(d_model) for _ in range(n_head)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, E) embedded + positionally-encoded input

        Returns:
            masked_xs: list of n_head tensors, each (B, T, E)
            masks: (B, n_head, T, E) for sparsity loss and interpretability
        """
        B, T, E = x.shape

        if self.time_varying:
            gate_input = x
        else:
            gate_input = x.mean(dim=1, keepdim=True).expand(-1, T, -1)

        prior = torch.ones(B, T, E, device=x.device)

        masked_xs = []
        all_masks = []

        for h in range(self.n_head):
            gi_flat = gate_input.reshape(B * T, E)
            h_logits = self.gate_fc[h](gi_flat)
            h_logits = self.gate_bn[h](h_logits)
            h_logits = h_logits.view(B, T, E)

            h_logits = h_logits * prior
            mask_h = sparsemax(h_logits, dim=-1)

            prior = prior * torch.clamp(self.gamma - mask_h, min=0)

            masked_xs.append(x * mask_h)
            all_masks.append(mask_h)

        masks = torch.stack(all_masks, dim=1)  # (B, n_head, T, E)
        return masked_xs, masks


class LTAESparse(nn.Module):
    """L-TAE with Sparse Channel Attention (L-TAE-S).

    Extends LTAE with a SparseFeatureGate that gives each attention head
    a learned, sparse, instance-dependent view of the embedding channels.

    Args:
        in_channels: number of input bands (6)
        d_model: embedding dimension (128)
        n_head: number of attention heads (16)
        d_k: key dimension per head (8)
        dropout: dropout rate (0.3)
        num_classes: number of output classes (5)
        gamma: prior scale coefficient for feature gate (1.5)
        time_varying_gate: whether gate varies per timestep (True)
    """

    def __init__(self, in_channels=6, d_model=128, n_head=16, d_k=8,
                 dropout=0.3, num_classes=5, gamma=1.5, time_varying_gate=True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k

        # Same embedding as LTAE
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, positions=MONTH_POSITIONS)

        # Sparse feature gate (novel component)
        self.feature_gate = SparseFeatureGate(
            d_model, n_head, gamma=gamma, time_varying=time_varying_gate
        )

        # Per-head key and value projections
        self.key_projs = nn.ModuleList([
            nn.Linear(d_model, d_k) for _ in range(n_head)
        ])
        self.value_projs = nn.ModuleList([
            nn.Linear(d_model, d_k) for _ in range(n_head)
        ])

        # Learnable query per head (same as LTAE)
        self.query = nn.Parameter(torch.randn(n_head, d_k))
        nn.init.normal_(self.query, mean=0, std=0.5)
        self.attention_dropout = nn.Dropout(dropout)

        # Same downstream as LTAE
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
        """
        Args:
            x: (B, T, C) raw temporal input

        Returns:
            logits: (B, num_classes)
            aux: dict with 'sparsity_loss' (scalar) and 'masks' (B, H, T, E)
        """
        B, T, C = x.shape
        x = self.embedding(x)       # (B, T, E)
        x = self.pos_enc(x)         # (B, T, E)

        # Sparse feature gating
        masked_xs, masks = self.feature_gate(x)

        # Per-head temporal attention
        head_outputs = []
        for h in range(self.n_head):
            x_h = masked_xs[h]  # (B, T, E)

            keys_h = self.key_projs[h](x_h)      # (B, T, d_k)
            values_h = self.value_projs[h](x_h)   # (B, T, d_k)

            query_h = self.query[h].unsqueeze(0).unsqueeze(0)  # (1, 1, d_k)
            query_h = query_h.expand(B, -1, -1)                # (B, 1, d_k)

            attn = torch.matmul(query_h, keys_h.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(attn, dim=-1)      # (B, 1, T)
            attn = self.attention_dropout(attn)

            out_h = torch.matmul(attn, values_h)     # (B, 1, d_k)
            head_outputs.append(out_h.squeeze(1))     # (B, d_k)

        # Concatenate heads: (B, n_head * d_k) = (B, 128)
        out = torch.cat(head_outputs, dim=-1)
        out = self.norm(out)
        out = self.mlp(out)
        logits = self.classifier(out)

        sparsity_loss = masks.mean()
        aux = {"sparsity_loss": sparsity_loss, "masks": masks}

        return logits, aux


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


class LTAELinear(nn.Module):
    """L-TAE with a single linear classification head (no MLP decoder).

    Same temporal attention encoder as LTAE, but replaces the 2-layer MLP
    decoder with a single nn.Linear layer. Designed to be paired with
    balanced cross-entropy loss for improved macro F1 on minority classes.
    """

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
        self.classifier = nn.Linear(n_head * d_k, num_classes)

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
        return self.classifier(out)


# =================== TabNet metric ===================

try:
    from pytorch_tabnet.metrics import Metric

    class F1MacroMetric(Metric):
        """F1 Macro metric for TabNet eval_metric."""
        def __init__(self):
            self._name = "f1_macro"
            self._maximize = True

        def __call__(self, y_true, y_score, weights=None):
            if isinstance(y_score, torch.Tensor):
                y_score = y_score.detach().cpu().numpy()
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
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


def train_epoch_sparse(model, optimizer, criterion, dataloader, scaler_amp, device,
                       lambda_sparse=1e-3):
    """Run one training epoch for L-TAE-S with sparsity regularization."""
    model.train()
    total_loss = 0
    total_sparse = 0
    for X, y in dataloader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits, aux = model(X)
            task_loss = criterion(logits, y)
            sparse_loss = aux["sparsity_loss"]
            loss = task_loss + lambda_sparse * sparse_loss
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item()
        total_sparse += sparse_loss.item()
    n = len(dataloader)
    return total_loss / n, total_sparse / n


@torch.no_grad()
def evaluate_sparse(model, dataloader, device):
    """Evaluate L-TAE-S model, returning logits and labels (ignoring aux output)."""
    model.eval()
    logits_all, labels_all = [], []
    with torch.amp.autocast("cuda"):
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            logits, _ = model(X)
            logits_all.append(logits.float().cpu())
            labels_all.extend(y.tolist())
    return torch.cat(logits_all, dim=0), labels_all


# =================== FastTabNet Architecture ===================


class FlatDataset(Dataset):
    """Dataset returning flat feature tensors + labels (no temporal reshape)."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GhostBN(nn.Module):
    """Ghost Batch Normalization — splits batch into virtual sub-batches for BN.

    Matches pytorch_tabnet's GBN. Each virtual chunk is normalized independently
    using the same BN parameters, reducing sensitivity to batch composition.
    """

    def __init__(self, dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.vbs = virtual_batch_size
        self.bn = nn.BatchNorm1d(dim, momentum=momentum)

    def forward(self, x):
        if self.training and x.shape[0] > self.vbs:
            n_chunks = max(1, (x.shape[0] + self.vbs - 1) // self.vbs)
            chunks = x.chunk(n_chunks, dim=0)
            return torch.cat([self.bn(c) for c in chunks], dim=0)
        return self.bn(x)


def _init_glu(linear, input_dim, output_dim):
    """TabNet-style Xavier init with custom gain for GLU layers."""
    gain = math.sqrt((input_dim + output_dim) / math.sqrt(input_dim))
    nn.init.xavier_normal_(linear.weight, gain=gain)


def _init_non_glu(linear, input_dim, output_dim):
    """TabNet-style Xavier init for non-GLU layers (attention transforms)."""
    gain = math.sqrt((input_dim + output_dim) / math.sqrt(4 * input_dim))
    nn.init.xavier_normal_(linear.weight, gain=gain)


class FastTabNet(nn.Module):
    """Custom TabNet faithfully reimplemented with L-TAE training optimizations.

    Matches pytorch_tabnet's architecture: sequential decision steps with
    sparsemax feature masks, shared+independent GLU layers with residual
    connections (sqrt(0.5) scaling), Ghost Batch Normalization, and
    per-step BN for shared linear weights.

    Returns (logits, aux) — compatible with train_epoch_sparse()/evaluate_sparse().

    Args:
        in_dim: number of input features (60 for 10 months x 6 bands)
        n_d: decision step output dimension
        n_a: attention dimension (feeds next step's mask)
        n_steps: number of sequential decision steps
        gamma: prior scale coefficient (controls feature reuse across steps)
        n_shared: number of shared GLU layers (weights reused, BN per-step)
        n_independent: number of step-specific GLU layers per step
        virtual_batch_size: chunk size for Ghost Batch Normalization
        momentum: BatchNorm momentum
        num_classes: number of output classes
    """

    def __init__(self, in_dim=60, n_d=64, n_a=64, n_steps=5, gamma=1.5,
                 n_shared=2, n_independent=2, virtual_batch_size=128,
                 momentum=0.02, num_classes=5):
        super().__init__()
        self.in_dim = in_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.gamma = gamma
        self.epsilon = 1e-15
        feat_dim = n_d + n_a

        # Initial batch normalization on raw input
        self.initial_bn = nn.BatchNorm1d(in_dim, momentum=momentum)

        # Shared linear weights (reused across initial splitter + all steps)
        # Each step gets its own BN but shares these Linear modules.
        self.shared_linears = nn.ModuleList()
        for i in range(n_shared):
            in_d = in_dim if i == 0 else feat_dim
            linear = nn.Linear(in_d, feat_dim * 2, bias=False)
            _init_glu(linear, in_d, feat_dim * 2)
            self.shared_linears.append(linear)

        # Initial splitter: own GhostBN for shared layers + own independent layers
        self.init_shared_bns = nn.ModuleList([
            GhostBN(feat_dim * 2, virtual_batch_size, momentum)
            for _ in range(n_shared)
        ])
        self.init_indep_linears = nn.ModuleList()
        self.init_indep_bns = nn.ModuleList()
        for _ in range(n_independent):
            linear = nn.Linear(feat_dim, feat_dim * 2, bias=False)
            _init_glu(linear, feat_dim, feat_dim * 2)
            self.init_indep_linears.append(linear)
            self.init_indep_bns.append(
                GhostBN(feat_dim * 2, virtual_batch_size, momentum))

        # Per-step: own GhostBN for shared layers + own independent layers
        self.step_shared_bns = nn.ModuleList()
        self.step_indep_linears = nn.ModuleList()
        self.step_indep_bns = nn.ModuleList()
        for _ in range(n_steps):
            self.step_shared_bns.append(nn.ModuleList([
                GhostBN(feat_dim * 2, virtual_batch_size, momentum)
                for _ in range(n_shared)
            ]))
            s_linears = nn.ModuleList()
            s_bns = nn.ModuleList()
            for _ in range(n_independent):
                linear = nn.Linear(feat_dim, feat_dim * 2, bias=False)
                _init_glu(linear, feat_dim, feat_dim * 2)
                s_linears.append(linear)
                s_bns.append(
                    GhostBN(feat_dim * 2, virtual_batch_size, momentum))
            self.step_indep_linears.append(s_linears)
            self.step_indep_bns.append(s_bns)

        # Attention transforms (one per step)
        self.attn_linears = nn.ModuleList()
        self.attn_bns = nn.ModuleList()
        for _ in range(n_steps):
            linear = nn.Linear(n_a, in_dim, bias=False)
            _init_non_glu(linear, n_a, in_dim)
            self.attn_linears.append(linear)
            self.attn_bns.append(
                GhostBN(in_dim, virtual_batch_size, momentum))

        # Final classifier
        self.final_fc = nn.Linear(n_d, num_classes)

    @staticmethod
    def _glu(x):
        """GLU activation: split in half, value * sigmoid(gate)."""
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

    def _feat_transform(self, x, shared_bns, indep_linears, indep_bns):
        """Shared + independent GLU layers with residual connections.

        Matches pytorch_tabnet's GLU_Block: first shared layer has no residual
        (dimension change in_dim → feat_dim), all subsequent layers use
        residual + sqrt(0.5) scaling.
        """
        SCALE = math.sqrt(0.5)

        # Shared layers (reuse self.shared_linears weights, per-step BN)
        for i, (linear, bn) in enumerate(zip(self.shared_linears, shared_bns)):
            h = self._glu(bn(linear(x)))
            if i == 0:
                x = h  # First layer: no residual (dimension change)
            else:
                x = (x + h) * SCALE  # Residual + scaling

        # Independent layers (always residual since shared already transformed)
        for linear, bn in zip(indep_linears, indep_bns):
            h = self._glu(bn(linear(x)))
            x = (x + h) * SCALE

        return x

    def forward(self, x):
        """
        Args:
            x: (B, in_dim) flat input features

        Returns:
            logits: (B, num_classes)
            aux: dict with 'sparsity_loss' scalar and 'masks' list
        """
        B = x.shape[0]

        # Batch normalize input (saved for masking at each step)
        x_bn = self.initial_bn(x)

        # Initial splitter: full shared+independent transform to get first h_a
        h = self._feat_transform(
            x_bn, self.init_shared_bns,
            self.init_indep_linears, self.init_indep_bns)
        h_a = h[:, self.n_d:]  # (B, n_a) — feeds first attention transform

        # Prior: starts uniform, gets reduced as features are used
        prior = torch.ones(B, self.in_dim, device=x.device)

        # Accumulated decision output
        output_agg = torch.zeros(B, self.n_d, device=x.device)

        masks = []

        for step in range(self.n_steps):
            # 1. Attention mask from previous step's h_a
            a = self.attn_linears[step](h_a)
            a = self.attn_bns[step](a)
            a = a * prior
            # Run sparsemax in float32 for numerical stability under AMP
            a = sparsemax(a.float(), dim=-1).to(x_bn.dtype)
            masks.append(a)

            # 2. Update prior (discourage reuse)
            prior = prior * (self.gamma - a)

            # 3. Apply mask to batch-normalized input
            masked_x = a * x_bn

            # 4. Feature transform (shared + independent with residuals)
            h = self._feat_transform(
                masked_x, self.step_shared_bns[step],
                self.step_indep_linears[step], self.step_indep_bns[step])

            # 5. Split and accumulate
            output_agg = output_agg + F.relu(h[:, :self.n_d])
            h_a = h[:, self.n_d:]

        # Classification
        logits = self.final_fc(output_agg)

        # Sparsity loss: entropy of masks (matches TabNet's M_loss / n_steps)
        mask_stack = torch.stack(masks, dim=1)  # (B, n_steps, in_dim)
        sparsity_loss = (
            -mask_stack * torch.log(mask_stack + self.epsilon)
        ).sum(dim=-1).mean()

        aux = {"sparsity_loss": sparsity_loss, "masks": masks}
        return logits, aux


def compute_focal_loss_weights(y_train, num_classes):
    """Compute class weights for WeightedFocalLoss from training labels."""
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum() * num_classes
    return torch.tensor(alpha, dtype=torch.float32)


def compute_balanced_ce_weights(y_train, num_classes):
    """Compute sklearn-style balanced class weights: n_samples / (n_classes * count_i)."""
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    n_samples = class_counts.sum()
    weights = n_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)
