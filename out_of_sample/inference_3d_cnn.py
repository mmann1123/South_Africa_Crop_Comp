"""
3D CNN inference on holdout test patches (34S_20E_259N).

Uses saved Keras model to generate predictions from patch data.

Input: test_patch_data.parquet (patch-level pixel data)
Output: predictions_3d_cnn.csv (field-level)

Note: Requires test patch data to be generated first via create_test_patches.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from collections import Counter

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import models
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Import config for shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import TEST_PATCH_DATA_PATH, MODEL_DIR

# Input
TEST_PATCH_DATA = TEST_PATCH_DATA_PATH

# Model
MODEL_PATH = os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5")

# Output
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_3d_cnn.csv")

# Model parameters (must match training)
BAND_PREFIXES = ["SA_B11", "SA_B12", "SA_B2", "SA_B6", "SA_EVI", "SA_hue"]
TARGET_SIZE = (128, 128)
BATCH_SIZE = 8

# Class names (alphabetical order as LabelEncoder encodes them)
CLASS_NAMES = ["Barley", "Canola", "Lucerne/Medics", "Small grain grazing", "Wheat"]


def group_band_columns(channel_cols, band_prefixes):
    """Group columns by band prefix, sorted by month."""
    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        if len(matching) == 0:
            print(f"Warning: No columns found for band prefix: {prefix}")
            continue

        def extract_month(col):
            parts = col.split("_")
            try:
                return int(parts[-1])
            except:
                return 0

        matching_sorted = sorted(matching, key=extract_month)
        band_mapping[prefix] = matching_sorted

    return band_mapping


def patch_pixels_to_image(df_patch, cols):
    """Reconstruct patch image from pixel rows."""
    min_r = df_patch["row"].min()
    max_r = df_patch["row"].max()
    min_c = df_patch["col"].min()
    max_c = df_patch["col"].max()
    H = (max_r - min_r) + 1
    W = (max_c - min_c) + 1
    C = len(cols)
    img = np.zeros((H, W, C), dtype=np.float32)

    for _, px in df_patch.iterrows():
        rr = int(px["row"] - min_r)
        cc = int(px["col"] - min_c)
        vals = [px[c] for c in cols]
        img[rr, cc, :] = vals

    return img


def resize_image(image, target_size=(128, 128)):
    """Resize image to target size."""
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    resized = tf.image.resize(tensor, target_size)
    return tf.squeeze(resized, axis=0).numpy()


def reconstruct_patch(df, patch_id, band_mapping, target_size):
    """Reconstruct 3D patch tensor from pixel data."""
    df_patch = df[df["patch_id"] == patch_id]
    band_prefixes = sorted(band_mapping.keys())
    T = min(len(band_mapping[p]) for p in band_prefixes if p in band_mapping)

    band_images = []
    for prefix in band_prefixes:
        if prefix not in band_mapping:
            continue
        cols = band_mapping[prefix][:T]
        band_img = patch_pixels_to_image(df_patch, cols)

        # Ensure consistent time dimension
        if band_img.shape[-1] > T:
            band_img = band_img[..., :T]
        elif band_img.shape[-1] < T:
            pad_width = T - band_img.shape[-1]
            band_img = np.pad(band_img, ((0, 0), (0, 0), (0, pad_width)), mode="constant")

        band_images.append(band_img)

    # Stack to (H, W, T, num_bands)
    stacked = np.stack(band_images, axis=-1)
    # Transpose to (T, H, W, num_bands)
    patch_3d = np.transpose(stacked, (2, 0, 1, 3))

    # Resize each time slice
    resized_slices = []
    for t_idx in range(patch_3d.shape[0]):
        slice_img = patch_3d[t_idx]
        resized_slice = resize_image(slice_img, target_size)
        resized_slices.append(resized_slice)

    # Final shape: (T, H, W, num_bands)
    final_tensor = np.stack(resized_slices, axis=0)
    return final_tensor


def main():
    print("=== 3D CNN Inference ===")

    if not HAS_TF:
        print("Error: TensorFlow is required for 3D CNN inference")
        return

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found: {MODEL_PATH}")
        print("Run 3D_CNN.py first to train the model.")
        return

    # Check patch data exists
    if not os.path.exists(TEST_PATCH_DATA):
        print(f"\nError: Test patch data not found: {TEST_PATCH_DATA}")
        print("Run create_test_patches.py first to generate patch data.")
        return

    # Load patch data
    print(f"\nLoading patch data: {TEST_PATCH_DATA}")
    df = pd.read_parquet(TEST_PATCH_DATA)
    print(f"Shape: {df.shape}")

    # Get patch IDs
    patch_ids = df["patch_id"].unique()
    print(f"Unique patches: {len(patch_ids)}")
    print(f"Unique fields: {df['field_id'].nunique()}")

    # Get band mapping
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    channel_cols = [c for c in df.columns if c not in ignore_cols]
    band_mapping = group_band_columns(channel_cols, BAND_PREFIXES)
    print(f"Band prefixes found: {list(band_mapping.keys())}")

    if not band_mapping:
        print("Error: No valid band columns found in patch data")
        return

    T = min(len(v) for v in band_mapping.values())
    print(f"Time steps: {T}")

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)
    print("Model loaded successfully")

    # Run inference
    print("\nRunning inference...")
    patch_predictions = []

    for i, patch_id in enumerate(patch_ids):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(patch_ids)} patches...")

        # Reconstruct patch
        patch_tensor = reconstruct_patch(df, patch_id, band_mapping, TARGET_SIZE)
        X_input = np.expand_dims(patch_tensor, axis=0)

        # Predict
        probs = model.predict(X_input, verbose=0)
        pred_label = np.argmax(probs, axis=1)[0]

        # Get field ID
        field_id = df.loc[df["patch_id"] == patch_id, "field_id"].iloc[0]

        patch_predictions.append({
            "patch_id": patch_id,
            "field_id": field_id,
            "pred_label": pred_label,
        })

    print(f"Total patch predictions: {len(patch_predictions)}")

    # Aggregate to field level by majority vote
    print("\nAggregating to field level...")
    patch_df = pd.DataFrame(patch_predictions)

    field_preds = patch_df.groupby("field_id")["pred_label"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    print(f"Total fields: {len(field_preds)}")

    # Convert label indices to class names
    field_labels = [CLASS_NAMES[p] for p in field_preds.values]

    # Save predictions
    df_out = pd.DataFrame({
        "fid": field_preds.index,
        "crop_name": field_labels,
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Summary
    print("\n=== Prediction Distribution ===")
    print(df_out["crop_name"].value_counts())


if __name__ == "__main__":
    main()
