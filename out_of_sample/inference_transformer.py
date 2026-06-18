"""
Transformer encoder ensemble inference on holdout test data (34S_20E_259N).

Uses 5-seed Transformer-encoder models to generate pixel-level predictions,
then aggregates to field level via majority vote.

Input: merged_dl_test.parquet (pixel-level)
Output: predictions_transformer.csv (field-level)

Required artifacts in models/:
  - transformer_seed_{42,101,202,303,404}.pt
  - transformer_scaler.joblib
  - transformer_label_encoder.joblib
  - transformer_feature_columns.joblib
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from joblib import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import MERGED_DL_TEST_PATH, MODEL_DIR

TEST_PARQUET = MERGED_DL_TEST_PATH
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_transformer.csv")
SEEDS = [42, 101, 202, 303, 404]

MONTH_POSITIONS = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
T_SEQ = 10
N_BANDS = 6

# Transformer hyperparameters (must match training)
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 3
DIM_FF = 256
DROPOUT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalDataset(Dataset):
    def __init__(self, X, fids):
        self.X = torch.tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32)
        self.fids = fids

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, idx):
        return self.X[idx], self.fids[idx]


# =================== Transformer Architecture (must match training) ===================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, positions=None):
        super().__init__()
        if positions is None:
            positions = list(range(T_SEQ))
        pe = torch.zeros(len(positions), d_model)
        pos = torch.tensor(positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SITSTransformer(nn.Module):
    def __init__(self, in_channels=6, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
                 dim_ff=DIM_FF, dropout=DROPOUT, num_classes=5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, positions=MONTH_POSITIONS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        x = x.mean(dim=1)
        x = self.mlp(x)
        return self.classifier(x)


def main():
    print(f"=== Transformer Encoder Ensemble Inference === Device: {device}")

    # Check artifacts
    for artifact in ["transformer_scaler.joblib", "transformer_label_encoder.joblib", "transformer_feature_columns.joblib"]:
        if not os.path.exists(os.path.join(MODEL_DIR, artifact)):
            print(f"Error: Missing {artifact}")
            return

    feature_cols = load(os.path.join(MODEL_DIR, "transformer_feature_columns.joblib"))
    scaler = load(os.path.join(MODEL_DIR, "transformer_scaler.joblib"))
    le = load(os.path.join(MODEL_DIR, "transformer_label_encoder.joblib"))
    print(f"Classes: {list(le.classes_)}, Features: {len(feature_cols)}")

    # Load test data
    print(f"\nLoading: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape: {df.shape}, Fields: {df['fid'].nunique()}")

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"  Warning: missing column '{col}', filling with 0")
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = scaler.transform(df[feature_cols].values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    fids = df["fid"].values

    dataset = TemporalDataset(X, fids)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

    # Load and run models
    print("\nLoading models...")
    logits_all = []
    for seed in SEEDS:
        model_path = os.path.join(MODEL_DIR, f"transformer_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(f"Error: Model not found: {model_path}")
            return
        model = SITSTransformer(in_channels=N_BANDS, num_classes=len(le.classes_)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"  Loaded transformer_seed_{seed}.pt")

        seed_logits = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(device)
                seed_logits.append(model(X_batch).cpu())
        logits_all.append(torch.cat(seed_logits, dim=0).unsqueeze(0))

    # Ensemble average (cast to float32 — autocast produces float16)
    avg_logits = torch.cat(logits_all, dim=0).float().mean(dim=0)
    preds = avg_logits.argmax(dim=1).tolist()
    print(f"Total pixel predictions: {len(preds)}")

    # Field-level majority vote
    print("\nAggregating to field level...")
    pred_df = pd.DataFrame({"fid": fids, "pred": preds})
    field_preds = pred_df.groupby("fid")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])
    labels = le.inverse_transform(field_preds.values)
    print(f"Total fields: {len(field_preds)}")

    df_out = pd.DataFrame({"fid": field_preds.index, "crop_name": labels})
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"\n{df_out['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()
