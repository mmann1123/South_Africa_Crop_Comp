"""
Multi-temporal, multi-spectral crop classification using a 3D CNN.
The input patches are restructured to incorporate time.
Instead of using 60 channels (6 bands x ~10 months) as flat channels,
we group them into 6 spectral bands and T time steps.
The final input to the network is of shape (T, 128, 128, 6).
Train/test split is based on field_id.
Oversampling and sample weighting are applied to handle class imbalance.
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
from collections import Counter
import joblib

PATCH_PARQUET = PATCH_DATA_PATH
TARGET_SIZE = (128, 128)  
BATCH_SIZE = 8
EPOCHS = 20

BAND_PREFIXES = ['SA_B11', 'SA_B12', 'SA_B2', 'SA_B6', 'SA_EVI', 'SA_hue']

def group_band_columns(channel_cols, band_prefixes):
    """
    Returns a dictionary mapping each band prefix to a sorted list of columns.
    Sorting is done based on the month number extracted from the column name.
    """
    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        if len(matching) == 0:
            raise ValueError(f"No columns found for band prefix: {prefix}")

        def extract_month(col):

            parts = col.split('_')
            try:
                return int(parts[-1])
            except:
                return 0

        matching_sorted = sorted(matching, key=extract_month)
        band_mapping[prefix] = matching_sorted
    return band_mapping


def patch_pixels_to_image(df_patch, cols):
    """
    Reconstructs a patch image from pixel-level rows for the given list of columns.
    Returns an array of shape [H, W, len(cols)].
    """
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
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)  
    resized = tf.image.resize(tensor, target_size)
    return tf.squeeze(resized, axis=0).numpy() 


def patch_data_generator_time(df, patch_ids, band_mapping, label_encoder,
                              batch_size=8, infinite=True, target_size=(128, 128)):
    """
    The generator:
      1. For each patch, for each band prefix, it reconstructs an image
         using the corresponding columns (sorted by month).
      2. It then crops/pads each image to target spatial size.
      3. It uses the minimum common number of time steps T across bands.
      4. Stacks the band images along a new axis so that final shape becomes
         (H, W, num_bands, T) and then transposes to (T, H, W, num_bands).
    """
    T = min(len(v) for v in band_mapping.values())
    num_bands = len(band_mapping)
    band_prefixes = sorted(band_mapping.keys())

    df_sub = df[df["patch_id"].isin(patch_ids)]
    unique_patches = df_sub["patch_id"].unique()
    rng = np.random.default_rng()
    while True:
        shuffled = rng.permutation(unique_patches)
        X_batch, y_batch = [], []
        for pid in shuffled:
            df_patch = df_sub[df_sub["patch_id"] == pid]
            crops = df_patch["crop_name"].unique()
            if len(crops) == 0:
                continue
            crop_str = crops[0]
            if crop_str not in label_encoder.classes_:
                continue
            y_val = label_encoder.transform([crop_str])[0]
            band_images = []
            for prefix in band_prefixes:
                cols = band_mapping[prefix][:T]  
                band_img = patch_pixels_to_image(df_patch, cols)  
                if band_img.shape[-1] > T:
                    band_img = band_img[..., :T]

                elif band_img.shape[-1] < T:
                    pad_width = T - band_img.shape[-1]
                    band_img = np.pad(band_img, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                band_images.append(band_img)
            stacked = np.stack(band_images, axis=-1)
            patch_img_final = np.transpose(stacked, (3, 0, 1, 2))
            T_current = patch_img_final.shape[0]
            resized_slices = []
            for t in range(T_current):
                slice_img = patch_img_final[t]  
                resized_slice = resize_image(slice_img,
                                             target_size=target_size) 
                resized_slices.append(resized_slice)

            # Stack back along time dimension: final shape (T, target_size[0], target_size[1], num_bands)
            final_patch = np.stack(resized_slices, axis=0)
            X_batch.append(final_patch)
            y_batch.append(y_val)
            if len(X_batch) == batch_size:
                yield (np.array(X_batch), np.array(y_batch))
                X_batch, y_batch = [], []
        if len(X_batch) > 0:
            yield (np.array(X_batch), np.array(y_batch))
            X_batch, y_batch = [], []
        if not infinite:
            break


def main():

    df = pd.read_parquet(PATCH_PARQUET)
    print("Loaded DF shape:", df.shape)

    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    df_ignore = df[list(ignore_cols)]
    channel_candidates = [c for c in df.columns if c not in ignore_cols]
    df_channel = df[channel_candidates].dropna(axis=1)
    df = pd.concat([df_ignore, df_channel], axis=1)
    print("After dropping NaN band columns, shape:", df.shape)

    # Filter invalid crops. This is not needed when the data is processed correctly
    df = df.dropna(subset=["crop_name"])
    df = df[df["crop_name"].str.lower() != "none"]

    # Field-based split. Avoids any data leakage
    fields = df["field_id"].dropna().unique()
    if len(fields) == 0:
        print("No valid field_id found. Exiting.")
        return
    f_train, f_test = train_test_split(fields, test_size=0.2, random_state=42)
    train_ids = df.loc[df["field_id"].isin(f_train), "patch_id"].unique()
    test_ids = df.loc[df["field_id"].isin(f_test), "patch_id"].unique()
    print(f"Train patches: {len(train_ids)}, Test patches: {len(test_ids)}")

    df_train = df[df["patch_id"].isin(train_ids)].copy()
    df_test = df[df["patch_id"].isin(test_ids)].copy()

    le = LabelEncoder()
    le.fit(df_train["crop_name"].unique())
    df_test = df_test[df_test["crop_name"].isin(le.classes_)]
    if df_test.empty:
        print("No valid test patches after filtering unknown classes. Exiting.")
        return
    num_classes = len(le.classes_)

    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])
    c_count = len(channel_cols)
    print("Channel columns used:", channel_cols, "=> count:", c_count)

    # Group channels by band prefix.
    band_mapping = group_band_columns(channel_cols, BAND_PREFIXES)
    # Determine common time steps T: use minimum number across bands. In our case its 10 months
    T = min(len(v) for v in band_mapping.values())
    print(f"Using {T} time steps for each band.")

    patch_crop_map = {}
    for pid in train_ids:
        subset = df_train[df_train["patch_id"] == pid]
        crops = subset["crop_name"].unique()
        if len(crops) == 1:
            patch_crop_map[pid] = crops[0]
    patch_class_counts = Counter(patch_crop_map.values())
    total_patches = sum(patch_class_counts.values())
    n_classes = len(le.classes_)
    inv_freqs = {cls: 1.0 / patch_class_counts[cls] for cls in patch_class_counts}
    sum_inv = sum(inv_freqs[c] * patch_class_counts[c] for c in patch_class_counts)
    patch_probs = {pid: inv_freqs[cname] / sum_inv for pid, cname in patch_crop_map.items()}
    patch_weight = {pid: (1.0 * total_patches) / (n_classes * patch_class_counts[cname]) for pid, cname in
                    patch_crop_map.items()}

    # Build Data Generators.
    # The generator yields images of shape: (T, 128, 128, 6)
    train_gen = patch_data_generator_time(
        df_train, train_ids, band_mapping, le,
        batch_size=BATCH_SIZE, infinite=True, target_size=TARGET_SIZE
    )
    val_gen = patch_data_generator_time(
        df_test, test_ids, band_mapping, le,
        batch_size=BATCH_SIZE, infinite=True, target_size=TARGET_SIZE
    )

    # Printing batch shape for debugging and verifying validity. It should be (BATCH_SIZE, T, 128, 128, 6)
    X_temp, y_temp = next(train_gen)
    print("Example training batch shape:", X_temp.shape) 
    print("Example training labels shape:", y_temp.shape)
    
    train_gen = patch_data_generator_time(
        df_train, train_ids, band_mapping, le,
        batch_size=BATCH_SIZE, infinite=True, target_size=TARGET_SIZE
    )

    steps_per_epoch = math.ceil(len(train_ids) / BATCH_SIZE)
    validation_steps = math.ceil(len(test_ids) / BATCH_SIZE)
    print(f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

    input_shape = (T, TARGET_SIZE[0], TARGET_SIZE[1], len(BAND_PREFIXES))  # (T, 128, 128, 6)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling3D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    if len(train_ids) == 0:
        print("No training patches found. Exiting.")
        return

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps
    )

    # Evaluate the Model.
    test_gen = patch_data_generator_time(
        df_test, test_ids, band_mapping, le,
        batch_size=BATCH_SIZE, infinite=False, target_size=TARGET_SIZE
    )
    y_true, y_pred = [], []
    for X_batch, y_batch in test_gen:
        preds = model.predict(X_batch)
        preds_label = np.argmax(preds, axis=1)
        y_true.extend(y_batch.tolist())
        y_pred.extend(preds_label.tolist())

    if len(y_true) == 0:
        print("No valid test patches for evaluation. Exiting.")
        return

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- TEST EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save the model.
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5"))
    print("Model saved to", os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5"))
    joblib.dump(le, os.path.join(MODEL_DIR, "3d_cnn_label_encoder.joblib"))
    print("Label encoder saved to", os.path.join(MODEL_DIR, "3d_cnn_label_encoder.joblib"))

    # ===================== REPORT =====================
    from report import ModelReport

    report = ModelReport("3D CNN Patch-Level")
    report.set_hyperparameters({
        "target_size": list(TARGET_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam(lr=1e-4)",
        "loss": "sparse_categorical_crossentropy",
        "conv3d_filters": [32, 64, 128],
        "bands": BAND_PREFIXES,
        "time_steps": T,
    })
    report.set_split_info(train=len(train_ids), test=len(test_ids), seed=42, split_method="field-based (patch-level)")
    report.set_metrics(y_true, y_pred, le.classes_)
    report.set_training_history(history.history)
    report.generate()


def patch_data_generator_time(df, patch_ids, band_mapping, label_encoder,
                              batch_size=8, infinite=True, target_size=(128, 128)):
    """
    This generator reconstructs each patch using the grouped band columns.
    For each band prefix, it extracts the corresponding columns, takes only the first T
    (T = min number of available months among bands), then stacks them so that the final
    shape is (T, H, W, num_bands). This is our temporal input.
    """
    T = min(len(v) for v in band_mapping.values())
    num_bands = len(band_mapping)
    band_prefixes = sorted(band_mapping.keys())

    df_sub = df[df["patch_id"].isin(patch_ids)]
    unique_patches = df_sub["patch_id"].unique()
    rng = np.random.default_rng()
    while True:
        shuffled = rng.permutation(unique_patches)
        X_batch, y_batch = [], []
        for pid in shuffled:
            df_patch = df_sub[df_sub["patch_id"] == pid]
            crops = df_patch["crop_name"].unique()
            if len(crops) == 0:
                continue
            crop_str = crops[0]
            if crop_str not in label_encoder.classes_:
                continue
            y_val = label_encoder.transform([crop_str])[0]
            band_images = []
            for prefix in band_prefixes:
                cols = band_mapping[prefix][:T]  
                band_img = patch_pixels_to_image(df_patch, cols) 
                
                if band_img.shape[-1] > T:
                    band_img = band_img[..., :T]
                elif band_img.shape[-1] < T:
                    pad_width = T - band_img.shape[-1]
                    band_img = np.pad(band_img, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                band_images.append(band_img)
            
            stacked = np.stack(band_images, axis=-1)  
            patch_img_final = np.transpose(stacked, (2, 0, 1, 3))
          
            T_current = patch_img_final.shape[0]
            resized_slices = []
            for t in range(T_current):
                slice_img = patch_img_final[t]  
                resized_slice = resize_image(slice_img, target_size=target_size)
                resized_slices.append(resized_slice)
            final_patch = np.stack(resized_slices, axis=0)
            X_batch.append(final_patch)
            y_batch.append(y_val)
            if len(X_batch) == batch_size:
                yield (np.array(X_batch), np.array(y_batch))
                X_batch, y_batch = [], []
        if len(X_batch) > 0:
            yield (np.array(X_batch), np.array(y_batch))
            X_batch, y_batch = [], []
        if not infinite:
            break


if __name__ == "__main__":
    main()
