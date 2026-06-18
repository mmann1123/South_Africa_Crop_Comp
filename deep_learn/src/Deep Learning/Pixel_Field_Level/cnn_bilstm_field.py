"""
CNN-BiLSTM trained on FIELD-LEVEL data (field-averaged temporal sequences).

Field-level analog of cnn_bilstm.py: pixel time series are averaged per field
(~3300 train fields) before training, exactly as ltae_field.py / tempcnn_field.py
do for their architectures. This yields a genuine field-trained CNN-BiLSTM with its
own independent spatial-transfer score (not a re-scoring of the pixel model).

Same architecture as cnn_bilstm.py (1D conv over 6 bands -> BiLSTM), raw reflectance
(no scaling, matching the pixel CNN-BiLSTM), weighted focal loss, 5-seed ensemble.

Runs on CPU by design (tiny dataset) so it never competes with GPU jobs; set
CUDA_VISIBLE_DEVICES="" before launching.

Input:  data/merged_dl_train.parquet (pixel-level, aggregated to field)
Outputs:
  - models/cnn_bilstm_field_seed_*.pt + cnn_bilstm_field_{feature_cols,label_encoder}.joblib
  - in-region field-level report (reports/)
  - out_of_sample/predictions_cnn_bilstm_field.csv  (OOS holdout predictions)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MERGED_DL_PATH, MERGED_DL_TEST_PATH, MODEL_DIR

sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from report import ModelReport

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
PRED_CSV = os.path.join(REPO_ROOT, "out_of_sample", "predictions_cnn_bilstm_field.csv")

SEEDS = [42, 101, 202, 303, 404]
N_EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 15
BAND_PREFIXES = ['B2_', 'B6_', 'B11_', 'B12_', 'hue_', 'EVI_']


def get_feature_cols(df):
    """Band-major feature columns (df-column order), matching cnn_bilstm.py so the
    model's view(B, 6, -1) groups each band's full month series contiguously."""
    cols = [c for c in df.columns if any(b in c for b in BAND_PREFIXES)]
    return [c for c in cols if not df[c].isna().all()]


def aggregate_to_field(df, feature_cols, with_label=True):
    df[feature_cols] = df[feature_cols].fillna(0)
    agg = {c: 'mean' for c in feature_cols}
    if with_label:
        agg['crop_name'] = lambda x: x.mode()[0]
    out = df.groupby('fid').agg(agg).reset_index()
    out[feature_cols] = out[feature_cols].fillna(0)
    return out


class CropDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.astype(np.float32)
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        if self.y is None:
            return x, idx
        return x, torch.tensor(self.y[idx], dtype=torch.long)


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha[target] * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class CropCNNBiLSTM(nn.Module):
    """Identical to cnn_bilstm.py: 1D conv over 6 band-channels -> BiLSTM -> last step."""
    def __init__(self, num_classes, conv_filters=64, lstm_hidden=64, kernel_size=5, dropout=0.3):
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


def train_epoch(model, optimizer, criterion, loader):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    out = []
    for X, _ in loader:
        out.append(model(X.to(device)).cpu())
    return torch.cat(out, dim=0)


def main():
    t0 = time.time()
    print(f"=== CNN-BiLSTM Field-Level Training === Device: {device}")

    df = pd.read_parquet(MERGED_DL_PATH)
    feature_cols = get_feature_cols(df)
    print(f"Feature cols: {len(feature_cols)} (band-major; expect multiple of 6)")
    assert len(feature_cols) % 6 == 0, "feature count must be divisible by 6"

    df = aggregate_to_field(df[['fid', 'crop_name'] + feature_cols].copy(), feature_cols)
    print(f"Field-level shape: {df.shape} ({len(df)} fields)")

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['crop_name'])
    num_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    fids = df['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)
    tr, va, te = (df[df['fid'].isin(s)] for s in (train_fids, val_fids, test_fids))
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")

    X_tr = tr[feature_cols].values; y_tr = tr['label'].values
    X_va = va[feature_cols].values; y_va = va['label'].values
    X_te = te[feature_cols].values; y_te = te['label'].values

    counts = np.maximum(np.bincount(y_tr, minlength=num_classes).astype(np.float64), 1.0)
    alpha = (1.0 / counts); alpha = alpha / alpha.sum() * num_classes
    criterion = WeightedFocalLoss(torch.tensor(alpha, dtype=torch.float32)).to(device)

    val_loader = DataLoader(CropDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(CropDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    test_logits_ens = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = CropCNNBiLSTM(num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
        train_loader = DataLoader(CropDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

        best_f1, patience, mpath = 0.0, 0, os.path.join(MODEL_DIR, f"cnn_bilstm_field_seed_{seed}.pt")
        torch.save(model.state_dict(), mpath)
        for epoch in range(N_EPOCHS):
            loss = train_epoch(model, optimizer, criterion, train_loader)
            vp = predict_logits(model, val_loader).argmax(1).tolist()
            vf1 = f1_score(y_va, vp, average='macro')
            scheduler.step()
            if vf1 > best_f1:
                best_f1 = vf1; patience = 0; torch.save(model.state_dict(), mpath)
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"  Early stop epoch {epoch+1} (best val F1 {best_f1:.4f})")
                    break
        model.load_state_dict(torch.load(mpath, map_location=device))
        test_logits_ens.append(predict_logits(model, test_loader).unsqueeze(0))
        print(f"  seed {seed} done, best val F1 macro={best_f1:.4f}")

    joblib.dump(le, os.path.join(MODEL_DIR, "cnn_bilstm_field_label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "cnn_bilstm_field_feature_columns.joblib"))

    avg = torch.cat(test_logits_ens, 0).mean(0)
    pred = avg.argmax(1).tolist()
    print("\n--- Ensemble Field-Level (in-region) ---")
    print(f"  Acc={accuracy_score(y_te, pred):.4f}  F1_macro={f1_score(y_te, pred, average='macro'):.4f}  "
          f"F1_w={f1_score(y_te, pred, average='weighted'):.4f}  kappa={cohen_kappa_score(y_te, pred):.4f}")
    print(classification_report(y_te, pred, target_names=le.classes_))

    report = ModelReport("CNN-BiLSTM Field-Level (temporal)")
    report.set_hyperparameters({
        "architecture": "CNN(1D, 6->64, k=5) + BiLSTM(64) ", "training_level": "field (pixel means per FID)",
        "conv_filters": 64, "lstm_hidden": 64, "kernel_size": 5, "dropout": 0.3, "lr": LR,
        "optimizer": "AdamW (wd=1e-4)", "scheduler": "CosineAnnealingLR",
        "loss": "WeightedFocalLoss(gamma=2.0)", "features": "raw reflectance (unscaled)",
        "epochs": N_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "n_models": len(SEEDS), "seeds": SEEDS,
    })
    report.set_split_info(train=len(tr), val=len(va), test=len(te), seed=42,
                          split_method="fid-wise (field-level training)")
    report.set_metrics(y_te, np.array(pred), list(le.classes_))
    report.set_training_time(time.time() - t0)
    report.generate()

    # ---- OOS holdout predictions ----
    print(f"\n=== OOS inference on holdout ({MERGED_DL_TEST_PATH}) ===")
    dft = pd.read_parquet(MERGED_DL_TEST_PATH)
    for c in feature_cols:
        if c not in dft.columns:
            dft[c] = 0
    dft_field = aggregate_to_field(dft[['fid'] + [c for c in feature_cols if c in dft.columns]].copy(),
                                   feature_cols, with_label=False)
    Xo = dft_field[feature_cols].values
    oos_fids = dft_field['fid'].values
    oos_loader = DataLoader(CropDataset(Xo), batch_size=256, shuffle=False)
    logits_all = []
    for seed in SEEDS:
        model = CropCNNBiLSTM(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"cnn_bilstm_field_seed_{seed}.pt"),
                                         map_location=device))
        logits_all.append(predict_logits(model, oos_loader).unsqueeze(0))
    oos_pred = torch.cat(logits_all, 0).mean(0).argmax(1).tolist()
    pd.DataFrame({"fid": oos_fids, "crop_name": le.inverse_transform(oos_pred)}).to_csv(PRED_CSV, index=False)
    print(f"Saved OOS predictions: {PRED_CSV}  ({len(oos_fids)} fields)")
    print(f"\n[TIMER] Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
