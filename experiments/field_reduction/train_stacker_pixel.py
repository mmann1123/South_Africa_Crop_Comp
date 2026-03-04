"""Train L-TAE + LR Stacker (pixel-level) with GridSearchCV on val set.

Uses existing pre-trained L-TAE pixel models and a newly trained LR to generate
out-of-fold probability features on the validation set. A meta-learner LR is
tuned via GridSearchCV (FID-aware stratified K-fold) on the val set, then
trained on the full val set with the best hyperparameters.

DEPENDENCY: Requires trained L-TAE pixel models at models/ltae_pixel/frac_{F}/.
Run train_ltae_pixel.py first.

Usage:
    python train_stacker_pixel.py --fraction 0.50 --output-dir models/ltae_lr_stack/frac_0.50
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torchF
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, make_scorer,
)

from experiment_config import MERGED_DL_PATH, SEEDS_ENSEMBLE, SUBSAMPLE_SEED, MODELS_DIR
from models_arch import (
    LTAE, get_chrono_feature_cols, aggregate_field_preds, get_device,
    T_SEQ, N_BANDS,
)
from subsample import get_fid_split_dl, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)

BATCH_SIZE = 2048
N_FOLDS = 5


class _SimpleDataset(torch.utils.data.Dataset):
    """Dataset returning temporal tensors without labels (for inference only)."""

    def __init__(self, X):
        self.X = torch.tensor(X.reshape(-1, T_SEQ, N_BANDS), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def get_ltae_probs(ltae_model_dir, X_scaled, num_classes, device):
    """Load L-TAE 5-seed ensemble and return pixel-level softmax probabilities."""
    ds = _SimpleDataset(X_scaled)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    logits_all = []
    for seed in SEEDS_ENSEMBLE:
        model_path = os.path.join(ltae_model_dir, f"ltae_seed_{seed}.pt")
        model = LTAE(in_channels=N_BANDS, d_model=128, n_head=16, d_k=8,
                     dropout=0.3, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        seed_logits = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for X_batch in loader:
                seed_logits.append(model(X_batch.to(device)).float().cpu())
        logits_all.append(torch.cat(seed_logits, dim=0).unsqueeze(0))

    avg_logits = torch.cat(logits_all, dim=0).mean(dim=0)
    probs = torchF.softmax(avg_logits, dim=1).numpy()
    return probs


def build_fid_cv_splits(fid_arr, crop_labels_per_fid, n_splits=N_FOLDS):
    """Build FID-aware CV splits: StratifiedKFold on FIDs, expanded to pixel indices.

    Args:
        fid_arr: array of FID values per pixel (length = n_pixels)
        crop_labels_per_fid: Series mapping FID -> crop label (for stratification)
        n_splits: number of CV folds

    Returns:
        list of (train_pixel_indices, test_pixel_indices) tuples
    """
    fid_order = crop_labels_per_fid.index.values
    fid_labels = crop_labels_per_fid.values

    # Map FID → pixel indices
    fid_to_px = {}
    for i, fid in enumerate(fid_arr):
        fid_to_px.setdefault(fid, []).append(i)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = []
    for train_fid_idx, test_fid_idx in skf.split(fid_order, fid_labels):
        train_px = [i for fid in fid_order[train_fid_idx] for i in fid_to_px[fid]]
        test_px = [i for fid in fid_order[test_fid_idx] for i in fid_to_px[fid]]
        cv_splits.append((train_px, test_px))

    return cv_splits


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
    print(f"=== L-TAE+LR GridSearchCV Stacker === Fraction: {fraction}, Device: {device}")

    # --- Dependency check: L-TAE models must exist ---
    frac_str = f"{fraction:.2f}"
    ltae_model_dir = os.path.join(MODELS_DIR, "ltae_pixel", f"frac_{frac_str}")
    required_files = (
        [f"ltae_seed_{s}.pt" for s in SEEDS_ENSEMBLE]
        + ["scaler.joblib", "label_encoder.joblib", "feature_columns.joblib"]
    )
    for fname in required_files:
        fpath = os.path.join(ltae_model_dir, fname)
        if not os.path.exists(fpath):
            print(f"[ERROR] L-TAE dependency missing: {fpath}")
            print("Train L-TAE pixel models first:")
            print(f"  python train_ltae_pixel.py --fraction {fraction} "
                  f"--output-dir models/ltae_pixel/frac_{frac_str}")
            sys.exit(1)
    print(f"L-TAE models found at: {ltae_model_dir}")

    # --- Load data ---
    print("Loading data...")
    df = pd.read_parquet(MERGED_DL_PATH)
    print(f"Shape: {df.shape}")

    feature_cols = get_chrono_feature_cols(df)
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"Feature cols: {len(feature_cols)}")

    # Labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['crop_name'])
    num_classes = len(le.classes_)

    # --- Load L-TAE's scaler and verify consistency ---
    ltae_scaler = joblib.load(os.path.join(ltae_model_dir, "scaler.joblib"))
    ltae_le = joblib.load(os.path.join(ltae_model_dir, "label_encoder.joblib"))
    ltae_feature_cols = joblib.load(os.path.join(ltae_model_dir, "feature_columns.joblib"))

    assert ltae_feature_cols == feature_cols, (
        f"Feature column mismatch: stacker has {len(feature_cols)} cols, "
        f"L-TAE has {len(ltae_feature_cols)}"
    )
    assert list(ltae_le.classes_) == list(le.classes_), (
        f"Label encoder mismatch: stacker {le.classes_} vs L-TAE {ltae_le.classes_}"
    )

    # --- FID split (same as L-TAE) ---
    train_fids, val_fids, test_fids = get_fid_split_dl(df)
    sub_train_fids = subsample_train_fids(df, train_fids, fraction, seed=SUBSAMPLE_SEED)
    print(f"Train FIDs: {len(train_fids)} -> {len(sub_train_fids)} ({fraction*100:.0f}%)")

    train_mask = df['fid'].isin(sub_train_fids)
    val_mask = df['fid'].isin(val_fids)
    test_mask = df['fid'].isin(test_fids)
    print(f"Pixels: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # --- Scale data using L-TAE's scaler ---
    X_train = ltae_scaler.transform(df.loc[train_mask, feature_cols].values).astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = df.loc[train_mask, 'label'].values

    X_val = ltae_scaler.transform(df.loc[val_mask, feature_cols].values).astype(np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    y_val = df.loc[val_mask, 'label'].values
    val_fid_arr = df.loc[val_mask, 'fid'].values

    X_test = ltae_scaler.transform(df.loc[test_mask, feature_cols].values).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = df.loc[test_mask, 'label'].values
    test_fid_arr = df.loc[test_mask, 'fid'].values

    # ====================================================================
    # GENERATE VAL-SET META-FEATURES
    # ====================================================================

    # --- L-TAE probs on val set (existing 5-seed models, inference only) ---
    print("\n--- L-TAE inference on val set ---")
    t_ltae = time.time()
    ltae_val_probs = get_ltae_probs(ltae_model_dir, X_val, num_classes, device)
    print(f"  L-TAE val probs shape: {ltae_val_probs.shape} ({time.time()-t_ltae:.1f}s)")

    # --- Train base LR on subsampled train set ---
    print("\n--- Training base LR on train set ---")
    t_lr = time.time()
    base_lr = LogisticRegression(max_iter=500, n_jobs=4, class_weight='balanced')
    base_lr.fit(X_train, y_train)
    lr_train_time = time.time() - t_lr
    print(f"  LR training time: {lr_train_time:.1f}s")

    assert np.array_equal(base_lr.classes_, np.arange(num_classes)), (
        f"LR classes {base_lr.classes_} don't match expected range [0, {num_classes})"
    )

    # --- LR probs on val set ---
    lr_val_probs = base_lr.predict_proba(X_val)
    print(f"  LR val probs shape: {lr_val_probs.shape}")

    # --- Concatenate meta-features ---
    X_meta_val = np.hstack([ltae_val_probs, lr_val_probs])
    print(f"  Meta-feature shape: {X_meta_val.shape} (2 * {num_classes} classes)")

    # ====================================================================
    # GRIDSEARCHCV ON VAL SET (FID-aware stratified K-fold)
    # ====================================================================
    print(f"\n--- GridSearchCV ({N_FOLDS}-fold, FID-aware) ---")
    t_gs = time.time()

    # Build FID-aware CV splits
    val_fid_crop = (df.loc[val_mask].groupby('fid')['crop_name']
                    .agg(lambda x: x.mode()[0]))
    cv_splits = build_fid_cv_splits(val_fid_arr, val_fid_crop, n_splits=N_FOLDS)
    print(f"  Val FIDs: {len(val_fid_crop)}, CV folds: {len(cv_splits)}")

    # Stacker pipeline: StandardScaler → LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=500, solver='saga', n_jobs=4)),
    ])

    param_grid = {
        'lr__C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'lr__penalty': ['l1', 'l2'],
        'lr__class_weight': ['balanced', None],
    }

    f1_macro_scorer = make_scorer(f1_score, average='macro')

    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=cv_splits,
        scoring=f1_macro_scorer,
        refit=True,
        n_jobs=1,  # pipeline already uses n_jobs=4
        verbose=1,
    )
    grid_search.fit(X_meta_val, y_val)

    gs_time = time.time() - t_gs
    print(f"\n  GridSearch time: {gs_time:.1f}s")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1 macro: {grid_search.best_score_:.4f}")

    # Extract fitted stacker and scaler from best pipeline
    best_pipeline = grid_search.best_estimator_
    stacker_scaler = best_pipeline.named_steps['scaler']
    stacker = best_pipeline.named_steps['lr']

    # ====================================================================
    # EVALUATE ON INTERNAL TEST SET
    # ====================================================================
    print("\n--- Evaluating on test set ---")

    # L-TAE test probs (existing 5-seed models)
    print("  L-TAE inference on test set...")
    ltae_test_probs = get_ltae_probs(ltae_model_dir, X_test, num_classes, device)

    # LR test probs
    lr_test_probs = base_lr.predict_proba(X_test)

    # Stacker predict
    X_meta_test = np.hstack([ltae_test_probs, lr_test_probs])
    X_meta_test_scaled = stacker_scaler.transform(X_meta_test)
    stacker_preds = stacker.predict(X_meta_test_scaled)

    # Field-level majority vote
    field_true, field_pred = aggregate_field_preds(test_fid_arr, y_test, stacker_preds)
    acc = accuracy_score(field_true, field_pred)
    f1_m = f1_score(field_true, field_pred, average='macro')
    f1_w = f1_score(field_true, field_pred, average='weighted')
    kappa = cohen_kappa_score(field_true, field_pred)

    print(f"\n--- Stacker Field-Level Results ---")
    print(f"  acc={acc:.4f}, f1_macro={f1_m:.4f}, f1_weighted={f1_w:.4f}, kappa={kappa:.4f}")

    # --- Save artifacts ---
    joblib.dump(base_lr, os.path.join(args.output_dir, "base_lr.joblib"))
    joblib.dump(stacker, os.path.join(args.output_dir, "stacker.joblib"))
    joblib.dump(stacker_scaler, os.path.join(args.output_dir, "stacker_scaler.joblib"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(args.output_dir, "feature_columns.joblib"))

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            "model": "ltae_lr_stack",
            "fraction": fraction,
            "fusion": "grid_search_stacked",
            "n_folds": N_FOLDS,
            "best_params": grid_search.best_params_,
            "best_cv_f1_macro": round(grid_search.best_score_, 6),
            "train_fields": int(len(sub_train_fids)),
            "train_pixels": int(train_mask.sum()),
            "val_pixels": int(val_mask.sum()),
            "test_pixels": int(test_mask.sum()),
            "test_fields": int(len(test_fids)),
            "num_classes": num_classes,
            "ltae_model_dir": ltae_model_dir,
            "ltae_seeds": SEEDS_ENSEMBLE,
            "base_lr_train_time_sec": round(lr_train_time, 1),
            "grid_search_time_sec": round(gs_time, 1),
            "training_time_sec": round(time.time() - t0, 1),
            "metrics": {
                "accuracy": round(acc, 6),
                "f1_macro": round(f1_m, 6),
                "f1_weighted": round(f1_w, 6),
                "cohen_kappa": round(kappa, 6),
            },
        }, f, indent=2)

    print(f"\nArtifacts saved to {args.output_dir}")
    print(f"Total time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
