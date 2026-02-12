import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_DATA_PATH, MODEL_DIR, REPORTS_DIR

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from report import ModelReport

PATCH_PARQUET = PATCH_DATA_PATH
TARGET_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 20


def precompute_patches(df, patch_ids, channel_cols, label_encoder, target_size):
    """Vectorized: convert all patches to images in one pass. Returns X, y arrays."""
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
        cols = grp["col"].values
        min_r, min_c = rows.min(), cols.min()
        H = (rows.max() - min_r) + 1
        W = (cols.max() - min_c) + 1

        img = np.zeros((H, W, len(channel_cols)), dtype=np.float32)
        img[rows - min_r, cols - min_c, :] = grp[channel_cols].values

        images.append(img)
        labels.append(label_encoder.transform([crop_str])[0])

        if len(images) % 2000 == 0:
            print(f"  Reconstructed {len(images)} patches...")

    print(f"  Total: {len(images)} patches")

    # Batch-resize all images
    print("  Resizing to target size...")
    resized = np.zeros((len(images), target_size[0], target_size[1], len(channel_cols)), dtype=np.float32)
    batch_sz = 256
    for i in range(0, len(images), batch_sz):
        batch = images[i:i+batch_sz]
        # Pad to uniform size for batch resize
        max_h = max(img.shape[0] for img in batch)
        max_w = max(img.shape[1] for img in batch)
        padded = np.zeros((len(batch), max_h, max_w, len(channel_cols)), dtype=np.float32)
        for j, img in enumerate(batch):
            padded[j, :img.shape[0], :img.shape[1], :] = img
        resized_batch = tf.image.resize(padded, target_size).numpy()
        resized[i:i+len(batch)] = resized_batch

    return resized, np.array(labels, dtype=np.int64)


def main():
    # Load data
    df = pd.read_parquet(PATCH_PARQUET)

    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    df_ignore = df[list(ignore_cols)]
    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])
    df_channel = df[channel_cols].dropna(axis=1)
    df = pd.concat([df_ignore, df_channel], axis=1)

    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])
    print("Loaded patch-level parquet.")
    print("Shape:", df.shape)
    print(f"Channel columns: {len(channel_cols)}")

    df = df.dropna(subset=["crop_name"])
    df = df[df["crop_name"].str.lower() != "none"]

    unique_fields = df["field_id"].dropna().unique()
    if len(unique_fields) == 0:
        print("No valid field_id found. Exiting.")
        return

    fields_train, fields_test = train_test_split(unique_fields, test_size=0.2, random_state=42)
    train_patch_ids = df.loc[df["field_id"].isin(fields_train), "patch_id"].unique()
    test_patch_ids = df.loc[df["field_id"].isin(fields_test), "patch_id"].unique()
    print(f"#fields train: {len(fields_train)}, #fields test: {len(fields_test)}")
    print(f"#patches train: {len(train_patch_ids)}, #patches test: {len(test_patch_ids)}")

    df_train = df[df["patch_id"].isin(train_patch_ids)].copy()
    df_test = df[df["patch_id"].isin(test_patch_ids)].copy()

    le = LabelEncoder()
    le.fit(df_train["crop_name"].unique())
    df_test = df_test[df_test["crop_name"].isin(le.classes_)]

    print(f"\nChannel/Band columns used: {len(channel_cols)}")
    print(f"Classes: {list(le.classes_)}")

    # Precompute all patches into numpy arrays (vectorized)
    print("\nPrecomputing training patches...")
    X_train, y_train = precompute_patches(df_train, train_patch_ids, channel_cols, le, TARGET_SIZE)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("\nPrecomputing test patches...")
    X_test, y_test = precompute_patches(df_test, test_patch_ids, channel_cols, le, TARGET_SIZE)
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Create tf.data datasets with prefetching
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    n_channels = X_train.shape[-1]
    n_classes = len(le.classes_)

    model = models.Sequential([
        layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], n_channels)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=1)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- TEST EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "patch_level_cnn.h5"))
    print("CNN model saved to", os.path.join(MODEL_DIR, "patch_level_cnn.h5"))

    # Save label encoder
    import joblib
    joblib.dump(le, os.path.join(MODEL_DIR, "multi_channel_cnn_label_encoder.joblib"))

    # Generate report
    report = ModelReport("Multi-Channel CNN Patch-Level", os.path.abspath(__file__))
    report.set_hyperparameters({
        "target_size": list(TARGET_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "channels": n_channels,
    })
    report.set_split_info(train=len(train_patch_ids), test=len(test_patch_ids),
                          seed=42, split_method="field-based (patch-level)")
    report.set_metrics(y_test, y_pred, le.classes_)
    report.set_training_history(history.history)
    report.generate()


if __name__ == "__main__":
    main()
