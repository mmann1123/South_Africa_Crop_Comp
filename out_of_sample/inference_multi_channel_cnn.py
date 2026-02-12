"""
Multi-Channel CNN inference on holdout test patches (34S_20E_259N).

Uses saved Keras model (patch_level_cnn.h5) to generate predictions.
This model treats all band×month columns as channels in a 2D CNN.

Input: test_patch_data.parquet
Output: predictions_multi_channel_cnn.csv (field-level)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import TEST_PATCH_DATA_PATH, PATCH_DATA_PATH, MODEL_DIR

TEST_PATCH_DATA = TEST_PATCH_DATA_PATH
MODEL_PATH = os.path.join(MODEL_DIR, "patch_level_cnn.h5")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_multi_channel_cnn.csv")

TARGET_SIZE = (128, 128)
BATCH_SIZE = 16

# Alphabetical order (LabelEncoder default)
CLASS_NAMES = ["Barley", "Canola", "Lucerne/Medics", "Small grain grazing", "Wheat"]


def reconstruct_patch_2d(df_patch, channel_cols, target_size):
    """Reconstruct 2D multi-channel image from pixel data."""
    rows = df_patch["row"].values
    cols = df_patch["col"].values
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    H = (max_r - min_r) + 1
    W = (max_c - min_c) + 1
    rr = rows - min_r
    cc = cols - min_c

    vals = df_patch[channel_cols].values  # (num_pixels, n_channels)
    img = np.zeros((H, W, len(channel_cols)), dtype=np.float32)
    img[rr, cc, :] = vals

    # Resize to target size
    resized = tf.image.resize(
        tf.convert_to_tensor(img[np.newaxis], dtype=tf.float32), target_size
    )
    return resized[0].numpy()


def main():
    print("=== Multi-Channel CNN Inference ===")

    if not HAS_TF:
        print("Error: TensorFlow is required")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found: {MODEL_PATH}")
        print("Run Multi_Channel_CNN.py first to train the model.")
        return

    if not os.path.exists(TEST_PATCH_DATA):
        print(f"\nError: Test patch data not found: {TEST_PATCH_DATA}")
        print("Run create_test_patches.py first.")
        return

    # Load patch data
    print(f"\nLoading patch data: {TEST_PATCH_DATA}")
    df = pd.read_parquet(TEST_PATCH_DATA)
    print(f"Shape: {df.shape}")

    # Match training preprocessing: drop columns with ANY NaN
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])

    print("Loading training column info for NaN alignment...")
    df_train_cols = pd.read_parquet(PATCH_DATA_PATH, columns=channel_cols)
    train_nan_cols = df_train_cols.columns[df_train_cols.isna().any()].tolist()
    clean_cols = [c for c in channel_cols if c not in train_nan_cols]
    print(f"Training dropped {len(train_nan_cols)} NaN columns: {sorted(train_nan_cols)}")
    print(f"Using {len(clean_cols)} clean columns")
    del df_train_cols

    # Fill remaining NaN
    nan_count = df[clean_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"Filling {nan_count} NaN values with 0")
        df[clean_cols] = df[clean_cols].fillna(0)

    patch_ids = df["patch_id"].unique()
    print(f"Unique patches: {len(patch_ids)}")
    print(f"Unique fields: {df['field_id'].nunique()}")

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)
    print("Model loaded successfully")

    # Check input shape matches
    expected_channels = model.input_shape[-1]
    if len(clean_cols) != expected_channels:
        print(f"Warning: Model expects {expected_channels} channels but data has {len(clean_cols)}")
        print("Adjusting to match model input...")
        clean_cols = clean_cols[:expected_channels]

    # Pre-group by patch_id
    print("\nPre-grouping patches...")
    grouped = dict(list(df.groupby("patch_id")))
    patch_to_field = df.groupby("patch_id")["field_id"].first().to_dict()

    # Run inference
    print(f"\nRunning inference (batch_size={BATCH_SIZE})...")
    patch_predictions = []
    batch_tensors = []
    batch_patch_ids = []

    for i, patch_id in enumerate(patch_ids):
        df_patch = grouped[patch_id]
        patch_tensor = reconstruct_patch_2d(df_patch, clean_cols, TARGET_SIZE)
        batch_tensors.append(patch_tensor)
        batch_patch_ids.append(patch_id)

        if len(batch_tensors) == BATCH_SIZE or i == len(patch_ids) - 1:
            X_batch = np.stack(batch_tensors, axis=0)
            probs = model.predict(X_batch, verbose=0)
            pred_labels = np.argmax(probs, axis=1)

            for pid, pred in zip(batch_patch_ids, pred_labels):
                patch_predictions.append({
                    "patch_id": pid,
                    "field_id": patch_to_field[pid],
                    "pred_label": pred,
                })

            batch_tensors = []
            batch_patch_ids = []

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(patch_ids)} patches...")

    print(f"Total patch predictions: {len(patch_predictions)}")

    # Aggregate to field level
    print("\nAggregating to field level...")
    patch_df = pd.DataFrame(patch_predictions)
    field_preds = patch_df.groupby("field_id")["pred_label"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    print(f"Total fields: {len(field_preds)}")

    field_labels = [CLASS_NAMES[p] for p in field_preds.values]

    df_out = pd.DataFrame({
        "fid": field_preds.index,
        "crop_name": field_labels,
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    print("\n=== Prediction Distribution ===")
    print(df_out["crop_name"].value_counts())


if __name__ == "__main__":
    main()
