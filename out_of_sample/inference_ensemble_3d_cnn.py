"""
Ensemble 3D CNN inference on holdout test patches (34S_20E_259N).

Uses one-vs-all binary 3D CNN models (one per crop class) + LogisticRegression
meta-learner to generate predictions.

Input: test_patch_data.parquet
Output: predictions_ensemble_3d_cnn.csv (field-level)

Required model files:
  - conv3d_model_class_<crop>.h5 (one per class)
  - meta_model.joblib (LogisticRegression meta-learner)
  - ensemble_3d_cnn_label_encoder.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

try:
    import tensorflow as tf
    from tensorflow.keras import models
    HAS_TF = True
except ImportError:
    HAS_TF = False

from joblib import load as joblib_load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import TEST_PATCH_DATA_PATH, PATCH_DATA_PATH, MODEL_DIR

TEST_PATCH_DATA = TEST_PATCH_DATA_PATH
META_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "ensemble_3d_cnn_label_encoder.joblib")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_ensemble_3d_cnn.csv")

BAND_PREFIXES = ["SA_B11", "SA_B12", "SA_B2", "SA_B6", "SA_EVI", "SA_hue"]
TARGET_SIZE = (128, 128)
BATCH_SIZE = 16


def group_band_columns(channel_cols, band_prefixes):
    """Group columns by band prefix, sorted by month."""
    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        if not matching:
            print(f"Warning: No columns for band prefix: {prefix}")
            continue
        matching_sorted = sorted(matching, key=lambda x: int(x.split("_")[-1]))
        band_mapping[prefix] = matching_sorted
    return band_mapping


def reconstruct_patch_3d(df_patch, band_mapping, target_size):
    """Reconstruct 3D patch tensor (T, H, W, num_bands) from pixel data."""
    band_prefixes = sorted(band_mapping.keys())
    T = min(len(band_mapping[p]) for p in band_prefixes if p in band_mapping)

    rows = df_patch["row"].values
    cols = df_patch["col"].values
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    H = (max_r - min_r) + 1
    W = (max_c - min_c) + 1
    rr = rows - min_r
    cc = cols - min_c

    band_images = []
    for prefix in band_prefixes:
        if prefix not in band_mapping:
            continue
        col_names = band_mapping[prefix][:T]
        vals = df_patch[col_names].values  # (num_pixels, T)
        img = np.zeros((H, W, T), dtype=np.float32)
        img[rr, cc, :] = vals[:, :T]
        band_images.append(img)

    stacked = np.stack(band_images, axis=-1)  # (H, W, T, num_bands)
    patch_3d = np.transpose(stacked, (2, 0, 1, 3))  # (T, H, W, num_bands)

    resized = tf.image.resize(patch_3d, target_size)  # (T, 128, 128, num_bands)
    return resized.numpy()


def main():
    print("=== Ensemble 3D CNN Inference ===")

    if not HAS_TF:
        print("Error: TensorFlow is required")
        return

    # Load label encoder to get class names
    if not os.path.exists(LABEL_ENCODER_PATH):
        print(f"\nError: Label encoder not found: {LABEL_ENCODER_PATH}")
        print("Run 'Ensemble - 3D CNN.py' first to train the models.")
        return

    le = joblib_load(LABEL_ENCODER_PATH)
    classes = le.classes_
    print(f"Classes: {list(classes)}")

    # Load binary base models (one per class)
    base_models = {}
    for crop in classes:
        model_path = os.path.join(MODEL_DIR, f"conv3d_model_class_{crop}.h5")
        if not os.path.exists(model_path):
            print(f"\nError: Base model not found: {model_path}")
            return
        base_models[crop] = models.load_model(model_path)
        print(f"Loaded base model: {crop}")

    # Load meta-learner
    if not os.path.exists(META_MODEL_PATH):
        print(f"\nError: Meta-model not found: {META_MODEL_PATH}")
        return
    meta_model = joblib_load(META_MODEL_PATH)
    print("Loaded meta-model (LogisticRegression)")

    # Check patch data
    if not os.path.exists(TEST_PATCH_DATA):
        print(f"\nError: Test patch data not found: {TEST_PATCH_DATA}")
        print("Run create_test_patches.py first.")
        return

    # Load patch data
    print(f"\nLoading patch data: {TEST_PATCH_DATA}")
    df = pd.read_parquet(TEST_PATCH_DATA)
    print(f"Shape: {df.shape}")

    # Match training preprocessing
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])

    print("Loading training column info for NaN alignment...")
    df_train_cols = pd.read_parquet(PATCH_DATA_PATH, columns=channel_cols)
    train_nan_cols = df_train_cols.columns[df_train_cols.isna().any()].tolist()
    clean_cols = [c for c in channel_cols if c not in train_nan_cols]
    print(f"Training dropped {len(train_nan_cols)} NaN columns: {sorted(train_nan_cols)}")
    print(f"Using {len(clean_cols)} clean columns")
    del df_train_cols

    nan_count = df[clean_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"Filling {nan_count} NaN values with 0")
        df[clean_cols] = df[clean_cols].fillna(0)

    # Band mapping
    band_mapping = group_band_columns(clean_cols, BAND_PREFIXES)
    print(f"Band prefixes found: {list(band_mapping.keys())}")

    if not band_mapping:
        print("Error: No valid band columns found")
        return

    T = min(len(v) for v in band_mapping.values())
    print(f"Time steps: {T}")

    patch_ids = df["patch_id"].unique()
    print(f"Unique patches: {len(patch_ids)}")
    print(f"Unique fields: {df['field_id'].nunique()}")

    # Pre-group
    print("\nPre-grouping patches...")
    grouped = dict(list(df.groupby("patch_id")))
    patch_to_field = df.groupby("patch_id")["field_id"].first().to_dict()

    # For each patch, get probability from each binary base model
    print(f"\nRunning inference (batch_size={BATCH_SIZE})...")
    all_patch_ids_ordered = []
    all_meta_features = []

    batch_tensors = []
    batch_pids = []

    for i, patch_id in enumerate(patch_ids):
        df_patch = grouped[patch_id]
        patch_tensor = reconstruct_patch_3d(df_patch, band_mapping, TARGET_SIZE)
        batch_tensors.append(patch_tensor)
        batch_pids.append(patch_id)

        if len(batch_tensors) == BATCH_SIZE or i == len(patch_ids) - 1:
            X_batch = np.stack(batch_tensors, axis=0)

            # Get probability from each binary model
            probs_per_class = []
            for crop in classes:
                preds = base_models[crop].predict(X_batch, verbose=0)
                probs_per_class.append(preds.flatten())

            # Stack to (batch_size, num_classes) — each column is P(class=1)
            meta_features = np.column_stack(probs_per_class)

            all_patch_ids_ordered.extend(batch_pids)
            all_meta_features.append(meta_features)

            batch_tensors = []
            batch_pids = []

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(patch_ids)} patches...")

    # Concatenate all meta features
    X_meta = np.vstack(all_meta_features)
    print(f"\nMeta-features shape: {X_meta.shape}")

    # Predict with meta-learner
    meta_preds = meta_model.predict(X_meta)
    print(f"Total patch predictions: {len(meta_preds)}")

    # Convert label indices to class names
    pred_class_names = le.inverse_transform(meta_preds)

    # Build patch-level results
    patch_results = pd.DataFrame({
        "patch_id": all_patch_ids_ordered,
        "field_id": [patch_to_field[pid] for pid in all_patch_ids_ordered],
        "pred_label": pred_class_names,
    })

    # Aggregate to field level by majority vote
    print("\nAggregating to field level...")
    field_preds = patch_results.groupby("field_id")["pred_label"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    print(f"Total fields: {len(field_preds)}")

    df_out = pd.DataFrame({
        "fid": field_preds.index,
        "crop_name": field_preds.values,
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    print("\n=== Prediction Distribution ===")
    print(df_out["crop_name"].value_counts())


if __name__ == "__main__":
    main()
