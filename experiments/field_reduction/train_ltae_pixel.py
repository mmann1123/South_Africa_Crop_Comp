"""Train L-TAE (pixel-level) with a reduced fraction of training fields.

5-seed ensemble with averaged logits. Field-level evaluation via majority vote.

Usage:
    python train_ltae_pixel.py --fraction 0.50 --output-dir models/ltae_pixel/frac_0.50
"""

import argparse
import json
import os
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiment_config import MERGED_DL_PATH, SEEDS_ENSEMBLE, SUBSAMPLE_SEED
from models_arch import (
    LTAE, WeightedFocalLoss, TemporalDataset,
    get_chrono_feature_cols, compute_focal_loss_weights,
    train_epoch, evaluate, aggregate_field_preds, get_device,
    T_SEQ, N_BANDS,
)
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 100
BATCH_SIZE = 2048
LR = 1e-3
PATIENCE = 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    fraction = args.fraction
    print(f"=== L-TAE Pixel Training === Fraction: {fraction}, Device: {device}")

    # Load pixel-level data
    print("Loading data...")
    df = pd.read_parquet(MERGED_DL_PATH)
    print(f"Shape: {df.shape}")

    # Feature columns
    feature_cols = get_chrono_feature_cols(df)
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"Feature cols: {len(feature_cols)}")

    # Labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['crop_name'])
    num_classes = len(le.classes_)

    # FID split
    train_fids, val_fids, test_fids = get_fid_split_dl(df)

    # Subsample train FIDs
    sub_train_fids = subsample_train_fids(df, train_fids, fraction, seed=SUBSAMPLE_SEED)
    print(f"Train FIDs: {len(train_fids)} -> {len(sub_train_fids)} ({fraction*100:.0f}%)")

    train_mask = df['fid'].isin(sub_train_fids)
    val_mask = df['fid'].isin(val_fids)
    test_mask = df['fid'].isin(test_fids)
    print(f"Pixels: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # Scale features (fitted on subsampled training data)
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

    # Weighted Focal Loss
    alpha = compute_focal_loss_weights(y_train, num_classes)
    criterion = WeightedFocalLoss(alpha, gamma=2.0).to(device)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # 5-seed ensemble
    test_logits_ensemble = []
    t_train_start = time.time()

    for seed in SEEDS_ENSEMBLE:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = LTAE(in_channels=N_BANDS, d_model=128, n_head=16, d_k=8,
                     dropout=0.3, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
        scaler_amp = torch.amp.GradScaler("cuda")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)

        best_val_f1 = 0.0
        patience_counter = 0
        model_path = os.path.join(args.output_dir, f"ltae_seed_{seed}.pt")
        torch.save(model.state_dict(), model_path)

        for epoch in range(N_EPOCHS):
            t_ep = time.time()
            train_loss = train_epoch(model, optimizer, criterion, train_loader, scaler_amp, device)

            val_logits, val_labels = evaluate(model, val_loader, device)
            val_preds = val_logits.argmax(dim=1).tolist()
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{N_EPOCHS} - "
                      f"loss={train_loss:.4f}, val_f1={val_f1:.4f}, {time.time()-t_ep:.1f}s")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best and evaluate
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        test_logits, test_labels = evaluate(model, test_loader, device)
        test_logits_ensemble.append(test_logits.unsqueeze(0))

    train_time = time.time() - t_train_start

    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(args.output_dir, "feature_columns.joblib"))

    # Ensemble evaluation with field-level aggregation
    avg_logits = torch.cat(test_logits_ensemble, dim=0).float().mean(dim=0)
    pred_labels = avg_logits.argmax(dim=1).tolist()

    field_true, field_pred = aggregate_field_preds(test_fid_arr, test_labels, pred_labels)
    acc = accuracy_score(field_true, field_pred)
    f1_m = f1_score(field_true, field_pred, average='macro')
    f1_w = f1_score(field_true, field_pred, average='weighted')
    kappa = cohen_kappa_score(field_true, field_pred)

    print(f"\n--- Ensemble Field-Level Results ---")
    print(f"  acc={acc:.4f}, f1_macro={f1_m:.4f}, f1_weighted={f1_w:.4f}, kappa={kappa:.4f}")

    # Save metadata
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            "model": "ltae_pixel",
            "fraction": fraction,
            "train_fields": int(len(sub_train_fids)),
            "train_pixels": int(train_mask.sum()),
            "val_pixels": int(val_mask.sum()),
            "test_pixels": int(test_mask.sum()),
            "test_fields": int(len(test_fids)),
            "seeds": SEEDS_ENSEMBLE,
            "training_time_sec": round(train_time, 1),
            "metrics": {
                "accuracy": round(acc, 6),
                "f1_macro": round(f1_m, 6),
                "f1_weighted": round(f1_w, 6),
                "cohen_kappa": round(kappa, 6),
            },
        }, f, indent=2)

    print(f"Artifacts saved to {args.output_dir}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
