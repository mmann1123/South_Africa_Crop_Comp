"""
Re-train the meta-model (logistic regression) on the base-model outputs
to ensure correct feature ordering. Then generate patch-level & field-level
predictions on the test set, printing accuracy and Cohen's kappa for both.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MODEL_DIR

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras import models
from joblib import dump, load
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Set paths
TRAIN_DF_PATH = os.path.join(MODEL_DIR, "train_df.pkl")
TEST_DF_PATH = os.path.join(MODEL_DIR, "test_df.pkl")
BASE_MODEL_SAVE_DIR = os.path.join(MODEL_DIR, "BaseModels")
NEW_ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model_retrained.joblib")

# The six band prefixes used in training
BAND_PREFIXES = ["SA_B11", "SA_B12", "SA_B2", "SA_B6", "SA_EVI", "SA_hue"]

# Spatial dimensions used for each patch
TARGET_SIZE = (128, 128)

def group_band_columns(channel_cols, band_prefixes):

    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        matching_sorted = sorted(matching, key=lambda x: int(x.split("_")[-1]))
        band_mapping[prefix] = matching_sorted
    return band_mapping

def patch_pixels_to_image(df_patch, cols):

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
    tensor = tf.expand_dims(tensor, axis=0)  # [1, H, W, C]
    resized = tf.image.resize(tensor, target_size)
    return tf.squeeze(resized, axis=0).numpy()

def reconstruct_3d_patch(df_subset, patch_id, band_mapping, target_size):

    df_patch = df_subset[df_subset["patch_id"] == patch_id]
    band_prefixes = sorted(band_mapping.keys())
    T = min(len(band_mapping[p]) for p in band_prefixes)

    band_images = []
    for prefix in band_prefixes:
        cols = band_mapping[prefix][:T]
        band_img = patch_pixels_to_image(df_patch, cols)
        if band_img.shape[-1] > T:
            band_img = band_img[..., :T]
        elif band_img.shape[-1] < T:
            pad_width = T - band_img.shape[-1]
            band_img = np.pad(band_img, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
        band_images.append(band_img)

    # Stack to (H, W, T, num_bands)
    stacked = np.stack(band_images, axis=-1)
    # Move time dimension to front => (T, H, W, num_bands)
    patch_3d = np.transpose(stacked, (2, 0, 1, 3))

    # Resize each time slice
    T_current = patch_3d.shape[0]
    resized_slices = []
    for t_idx in range(T_current):
        slice_img = patch_3d[t_idx]
        resized_slice = resize_image(slice_img, target_size)
        resized_slices.append(resized_slice)

    # Final shape => (T_current, target_size[0], target_size[1], num_bands)
    final_tensor = np.stack(resized_slices, axis=0)
    return final_tensor


def get_patch_probs(df_subset, patch_id, base_models, sorted_classes, band_mapping):
    """
    Reconstruct the patch, then get a probability from each base model (one for each class).
    Return them in the same order as sorted_classes.
    """
    patch_tensor = reconstruct_3d_patch(df_subset, patch_id, band_mapping, TARGET_SIZE)
    X_input = np.expand_dims(patch_tensor, axis=0)  

    # For each class in sorted_classes, run the corresponding base model
    probs = []
    for class_label in sorted_classes:
        model = base_models[class_label]
        prob = model.predict(X_input)[0][0]  
        probs.append(prob)

    return np.array(probs)


def main():
    # 1) Load train & test DataFrames
    with open(TRAIN_DF_PATH, "rb") as f:
        df_train = pickle.load(f)
    with open(TEST_DF_PATH, "rb") as f:
        df_test = pickle.load(f)

    print("Train shape:", df_train.shape, "Test shape:", df_test.shape)

    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    channel_cols = [c for c in df_train.columns if c not in ignore_cols]
    band_mapping = group_band_columns(channel_cols, BAND_PREFIXES)

    # 2) Identify available base models (one per class)
    base_model_files = [
        f for f in os.listdir(BASE_MODEL_SAVE_DIR)
        if f.startswith("conv3d_model_class_") and f.endswith(".h5")
    ]
    raw_labels = [
        f.replace("conv3d_model_class_", "").replace(".h5", "")
        for f in base_model_files
    ]
    sorted_classes = sorted(raw_labels)

    print("Classes (base models) found, in sorted order:", sorted_classes)

    base_models = {}
    for class_label in sorted_classes:
        model_path = os.path.join(BASE_MODEL_SAVE_DIR, f"conv3d_model_class_{class_label}.h5")
        print(f"Loading base model for '{class_label}' from {model_path} ...")
        base_models[class_label] = models.load_model(model_path)

    # 3) Build ensemble features for the training set
    train_patches = df_train["patch_id"].unique()
    X_train_list = []
    y_train_list = []

    for pid in train_patches:
        probs = get_patch_probs(df_train, pid, base_models, sorted_classes, band_mapping)

        true_label = df_train.loc[df_train["patch_id"] == pid, "crop_name"].iloc[0]

        X_train_list.append(probs)
        y_train_list.append(true_label)

    X_train = np.stack(X_train_list, axis=0) 
    y_train = np.array(y_train_list)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    from sklearn.linear_model import LogisticRegression
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)

    test_patches = df_test["patch_id"].unique()
    patch_level_results = []

    for pid in test_patches:
        probs = get_patch_probs(df_test, pid, base_models, sorted_classes, band_mapping)
        probs_2d = probs.reshape(1, -1)

        predicted_label = meta_model.predict(probs_2d)[0]
        true_label = df_test.loc[df_test["patch_id"] == pid, "crop_name"].iloc[0]
        field_id = df_test.loc[df_test["patch_id"] == pid, "field_id"].iloc[0]

        patch_level_results.append({
            "patch_id": pid,
            "field_id": field_id,
            "true_class": true_label,
            "predicted_class": predicted_label
        })

    patch_df = pd.DataFrame(patch_level_results)
    print("\n--- Patch-level Predictions (sample) ---")
    print(patch_df.head())

    # Patch-level metrics
    patch_acc = accuracy_score(patch_df["true_class"], patch_df["predicted_class"])
    patch_kappa = cohen_kappa_score(patch_df["true_class"], patch_df["predicted_class"])

    print("\nPatch-level Accuracy: {:.4f}".format(patch_acc))
    print("Patch-level Cohen's Kappa: {:.4f}".format(patch_kappa))

    # 6) Field-level predictions by majority vote
    field_level_results = []
    for field_id, group in patch_df.groupby("field_id"):
        class_counts = Counter(group["predicted_class"])
        most_common_pred = class_counts.most_common(1)[0][0]

        unique_true = group["true_class"].unique()
        if len(unique_true) == 1:
            true_field_class = unique_true[0]
        else:
            true_field_class = unique_true[0]

        field_level_results.append({
            "field_id": field_id,
            "true_class": true_field_class,
            "predicted_class": most_common_pred
        })

    field_df = pd.DataFrame(field_level_results)
    print("\n--- Field-level Predictions (sample) ---")
    print(field_df.head())

    # Field-level metrics
    field_acc = accuracy_score(field_df["true_class"], field_df["predicted_class"])
    field_kappa = cohen_kappa_score(field_df["true_class"], field_df["predicted_class"])

    print("\nField-level Accuracy: {:.4f}".format(field_acc))
    print("Field-level Cohen's Kappa: {:.4f}".format(field_kappa))

    # 7) Saving to paths
    dump(meta_model, NEW_ENSEMBLE_MODEL_PATH)
    print(f"\nNewly retrained meta-model saved to '{NEW_ENSEMBLE_MODEL_PATH}'.")

    patch_df.to_csv("patch_level_predictions.csv", index=False)
    field_df.to_csv("field_level_predictions.csv", index=False)
    print("Patch-level predictions saved to 'patch_level_predictions.csv'.")
    print("Field-level predictions saved to 'field_level_predictions.csv'.")
    print("\nDone!")


if __name__ == "__main__":
    main()
