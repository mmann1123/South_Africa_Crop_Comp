"""
Ensemble of 3D CNN Models for Multi-Temporal Crop Classification
1. Creates 1 binary 3D CNN model for each class (one-vs-all)
2. Combines the predicted probabilities from each model to get meta features
3. Uses LogisticRegression ensemble to get final classes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_DATA_PATH, MODEL_DIR

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from joblib import dump
from report import ModelReport

PATCH_PARQUET = PATCH_DATA_PATH
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
BASE_MODEL_SAVE_DIR = MODEL_DIR
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model.joblib")

BAND_PREFIXES = ['SA_B11', 'SA_B12', 'SA_B2', 'SA_B6', 'SA_EVI', 'SA_hue']


def group_band_columns(channel_cols, band_prefixes):
    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        if not matching:
            raise ValueError(f"No columns found for band prefix: {prefix}")
        matching_sorted = sorted(matching, key=lambda x: int(x.split('_')[-1]))
        band_mapping[prefix] = matching_sorted
    return band_mapping


def precompute_3d_patches(df, patch_ids, band_mapping, label_encoder, target_size):
    """Vectorized: convert all patches to 3D tensors (T, H, W, bands). Returns X, y arrays."""
    band_prefixes = sorted(band_mapping.keys())
    T = min(len(band_mapping[p]) for p in band_prefixes)
    num_bands = len(band_prefixes)
    grouped = df.groupby("patch_id")

    images = []
    labels = []

    for pid in patch_ids:
        grp = grouped.get_group(pid)
        crops = grp["crop_name"].unique()
        if len(crops) == 0:
            continue
        crop_str = crops[0]
        if crop_str not in label_encoder.classes_:
            continue

        # Vectorized pixel placement
        rows = grp["row"].values
        cols_arr = grp["col"].values
        min_r, min_c = rows.min(), cols_arr.min()
        H = (rows.max() - min_r) + 1
        W = (cols_arr.max() - min_c) + 1
        rr = rows - min_r
        cc = cols_arr - min_c

        # Build per-band images and stack
        band_images = []
        for prefix in band_prefixes:
            col_names = band_mapping[prefix][:T]
            vals = grp[col_names].values  # (num_pixels, T)
            img = np.zeros((H, W, T), dtype=np.float32)
            img[rr, cc, :] = vals[:, :T]
            band_images.append(img)

        stacked = np.stack(band_images, axis=-1)  # (H, W, T, num_bands)
        patch_3d = np.transpose(stacked, (2, 0, 1, 3))  # (T, H, W, num_bands)

        images.append(patch_3d)
        labels.append(label_encoder.transform([crop_str])[0])

        if len(images) % 2000 == 0:
            print(f"  Reconstructed {len(images)} patches...")

    print(f"  Total: {len(images)} patches")

    # Batch-resize all patches
    print("  Resizing to target size...")
    resized = np.zeros((len(images), T, target_size[0], target_size[1], num_bands), dtype=np.float32)
    batch_sz = 64
    for i in range(0, len(images), batch_sz):
        batch = images[i:i+batch_sz]
        for j, patch_3d in enumerate(batch):
            # patch_3d shape: (T, H, W, bands) — resize spatial dims
            r = tf.image.resize(patch_3d, target_size).numpy()  # (T, 128, 128, bands)
            resized[i+j] = r

    return resized, np.array(labels, dtype=np.int64)


def create_binary_model(input_shape):
    """Build a 3D CNN model for binary classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling3D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)

    # Load and preprocess data
    df = pd.read_parquet(PATCH_PARQUET)
    print("Loaded DF shape:", df.shape)
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    df_ignore = df[list(ignore_cols)]
    channel_candidates = [c for c in df.columns if c not in ignore_cols]
    df_channel = df[channel_candidates].dropna(axis=1)
    df = pd.concat([df_ignore, df_channel], axis=1)
    print("After dropping NaN band columns, shape:", df.shape)

    df = df.dropna(subset=["crop_name"])
    df = df[df["crop_name"].str.lower() != "none"]
    fields = df["field_id"].dropna().unique()
    if len(fields) == 0:
        raise ValueError("No valid field_id found.")
    f_train, f_test = train_test_split(fields, test_size=0.2, random_state=42)
    train_ids = df.loc[df["field_id"].isin(f_train), "patch_id"].unique()
    test_ids = df.loc[df["field_id"].isin(f_test), "patch_id"].unique()
    print(f"Train patches: {len(train_ids)}, Test patches: {len(test_ids)}")

    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])
    band_mapping = group_band_columns(channel_cols, BAND_PREFIXES)
    T = min(len(v) for v in band_mapping.values())
    print(f"Using {T} time steps for each band.")

    le = LabelEncoder()
    le.fit(df[df["patch_id"].isin(train_ids)]["crop_name"].unique())
    classes = le.classes_
    print("Crop classes:", classes)

    # Precompute all patches into numpy arrays (vectorized)
    df_train = df[df["patch_id"].isin(train_ids)].copy()
    df_test = df[df["patch_id"].isin(test_ids)].copy()

    print("\nPrecomputing training patches...")
    X_train, y_train = precompute_3d_patches(df_train, train_ids, band_mapping, le, TARGET_SIZE)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("\nPrecomputing test patches...")
    X_test, y_test = precompute_3d_patches(df_test, test_ids, band_mapping, le, TARGET_SIZE)
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    input_shape = (T, TARGET_SIZE[0], TARGET_SIZE[1], len(BAND_PREFIXES))

    # Train one binary model per class
    base_models = {}
    for crop in classes:
        print(f"\nTraining base model for class '{crop}' (one-vs-all)...")
        target_label = le.transform([crop])[0]

        # Binary labels
        y_train_bin = (y_train == target_label).astype(np.int32)
        y_test_bin = (y_test == target_label).astype(np.int32)

        # tf.data datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_bin))
        train_ds = train_ds.shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_bin))
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = create_binary_model(input_shape)
        model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

        model_path = os.path.join(BASE_MODEL_SAVE_DIR, f"conv3d_model_class_{crop}.h5")
        model.save(model_path)
        print(f"Saved base model for class '{crop}' to {model_path}")
        base_models[crop] = model

    # Generate ensemble meta-features: batch predict all patches through each base model
    print("\nGenerating ensemble features (batch mode)...")

    meta_train = np.zeros((len(X_train), len(classes)), dtype=np.float32)
    for i, crop in enumerate(classes):
        probs = base_models[crop].predict(X_train, batch_size=BATCH_SIZE, verbose=0)
        meta_train[:, i] = probs.flatten()
        print(f"  Train meta-features for '{crop}': done")

    meta_test = np.zeros((len(X_test), len(classes)), dtype=np.float32)
    for i, crop in enumerate(classes):
        probs = base_models[crop].predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        meta_test[:, i] = probs.flatten()
        print(f"  Test meta-features for '{crop}': done")

    # Train meta-classifier
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_train, y_train)
    print("Meta-classifier trained.")

    y_pred = meta_model.predict(meta_test)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- ENSEMBLE TEST EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save artifacts
    dump(meta_model, ENSEMBLE_MODEL_PATH)
    print(f"Ensemble meta-model saved to {ENSEMBLE_MODEL_PATH}")
    dump(le, os.path.join(BASE_MODEL_SAVE_DIR, "ensemble_3d_cnn_label_encoder.joblib"))
    print(f"Label encoder saved")

    # Generate report
    report = ModelReport("Ensemble 3D CNN Patch-Level", os.path.abspath(__file__))
    report.set_hyperparameters({
        "target_size": list(TARGET_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam(lr=1e-4)",
        "loss": "binary_crossentropy (per-class) + LogisticRegression meta",
        "bands": BAND_PREFIXES,
        "time_steps": T,
        "num_base_models": len(classes),
    })
    report.set_split_info(train=len(train_ids), test=len(test_ids),
                          seed=42, split_method="field-based (patch-level)")
    report.set_metrics(y_test, y_pred, le.classes_)
    report.generate()


if __name__ == "__main__":
    main()
