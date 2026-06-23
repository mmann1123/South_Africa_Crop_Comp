import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_DATA_PATH, MODEL_DIR, REPORTS_DIR

import math
import time as _time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from collections import Counter
from report import ModelReport

PATCH_PARQUET = PATCH_DATA_PATH
TARGET_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 20

# Reproducibility (split already uses random_state=42)
np.random.seed(42)
tf.random.set_seed(42)


def build_patch_store(df, patch_ids, channel_cols, label_encoder):
    """Pre-extract per-patch pixel arrays once (keeps raw pixels in RAM, ~1-2 GB
    total — NOT the resized 128x128 tensors, which would be ~100 GB). Patches are
    reconstructed and resized lazily, per batch, in PatchSequence."""
    grouped = df.groupby("patch_id")
    store = {}
    labels = {}
    valid_ids = []
    for pid in patch_ids:
        grp = grouped.get_group(pid)
        crops = grp["crop_name"].unique()
        if len(crops) == 0:
            continue
        crop_str = crops[0]
        if crop_str not in label_encoder.classes_:
            continue
        rows = grp["row"].values
        cols = grp["col"].values
        rr = (rows - rows.min()).astype(np.int16)
        cc = (cols - cols.min()).astype(np.int16)
        vals = grp[channel_cols].values.astype(np.float32)
        store[pid] = (rr, cc, vals)
        labels[pid] = int(label_encoder.transform([crop_str])[0])
        valid_ids.append(pid)
        if len(valid_ids) % 5000 == 0:
            print(f"  Indexed {len(valid_ids)} patches...")
    print(f"  Total: {len(valid_ids)} patches")
    return store, labels, valid_ids


class PatchSequence(tf.keras.utils.Sequence):
    """Streams patches to the model one batch at a time. Each patch is placed on
    its native HxW grid and resized to TARGET_SIZE on the fly (matching the
    per-patch resize used at inference time), so memory stays bounded."""

    def __init__(self, store, labels, ids, n_channels, target_size, batch_size, shuffle):
        self.store = store
        self.labels = labels
        self.ids = list(ids)
        self.n_channels = n_channels
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

    def on_epoch_end(self):
        self.order = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.order)

    def __getitem__(self, idx):
        sel = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(sel), self.target_size[0], self.target_size[1], self.n_channels), dtype=np.float32)
        y = np.zeros(len(sel), dtype=np.int64)
        for j, i in enumerate(sel):
            pid = self.ids[i]
            rr, cc, vals = self.store[pid]
            H = int(rr.max()) + 1
            W = int(cc.max()) + 1
            img = np.zeros((H, W, self.n_channels), dtype=np.float32)
            img[rr, cc, :] = vals
            X[j] = tf.image.resize(img[np.newaxis], self.target_size)[0].numpy()
            y[j] = self.labels[pid]
        return X, y


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

    # Index patches (lazy reconstruction happens in the Sequence)
    print("\nIndexing training patches...")
    store_tr, labels_tr, ids_tr = build_patch_store(df_train, train_patch_ids, channel_cols, le)
    print("\nIndexing test patches...")
    store_te, labels_te, ids_te = build_patch_store(df_test, test_patch_ids, channel_cols, le)

    n_channels = len(channel_cols)
    n_classes = len(le.classes_)

    # val_seq has shuffle=False, so its row order is fixed (== ids_te order) and
    # identical across seeds — required to average probabilities positionally.
    val_seq = PatchSequence(store_te, labels_te, ids_te, n_channels, TARGET_SIZE, BATCH_SIZE, shuffle=False)

    # 5-seed ensemble: train one model per seed, average softmax probabilities.
    SEEDS = [42, 101, 202, 303, 404]
    t_train_start = _time.time()
    prob_sum = None          # accumulates per-seed patch-level probabilities
    first_history = None     # training curve from the first seed (for the report)
    per_seed_seconds = {}

    for si, seed in enumerate(SEEDS):
        print(f"\n===== Seed {seed} ({si + 1}/{len(SEEDS)}) =====")
        seed_start = _time.time()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Rebuild train_seq after seeding so the initial shuffle is seed-dependent.
        train_seq = PatchSequence(store_tr, labels_tr, ids_tr, n_channels, TARGET_SIZE, BATCH_SIZE, shuffle=True)

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
        if si == 0:
            model.summary()

        history = model.fit(train_seq, epochs=EPOCHS, validation_data=val_seq)
        if first_history is None:
            first_history = history.history

        probs = model.predict(val_seq)
        prob_sum = probs if prob_sum is None else prob_sum + probs

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(os.path.join(MODEL_DIR, f"patch_level_cnn_seed{seed}.h5"))

        seed_elapsed = _time.time() - seed_start
        per_seed_seconds[str(seed)] = round(seed_elapsed, 1)
        print(f"Seed {seed} done in {seed_elapsed / 60:.1f} min")

    # ---- Ensemble patch-level predictions (mean softmax over seeds) ----
    ensemble_probs = prob_sum / len(SEEDS)
    y_pred_patch = np.argmax(ensemble_probs, axis=1)
    y_true_patch = np.array([labels_te[pid] for pid in ids_te], dtype=np.int64)
    print("Per-seed training time (s):", per_seed_seconds)

    acc_patch = accuracy_score(y_true_patch, y_pred_patch)
    print(f"\n--- PATCH-LEVEL TEST ---  acc={acc_patch:.4f}  "
          f"kappa={cohen_kappa_score(y_true_patch, y_pred_patch):.4f}")

    # ---- Aggregate patch predictions to field level (majority vote) ----
    patch_to_field = df_test.groupby("patch_id")["field_id"].first()
    patch_df = pd.DataFrame({
        "patch_id": ids_te,
        "pred": y_pred_patch,
    })
    patch_df["field_id"] = patch_df["patch_id"].map(patch_to_field)
    field_pred = patch_df.groupby("field_id")["pred"].agg(
        lambda x: Counter(x).most_common(1)[0][0])
    # True field label: encode the (single) crop per field
    field_true_str = df_test.groupby("field_id")["crop_name"].first()
    common_fields = field_pred.index.intersection(field_true_str.index)
    y_field_pred = field_pred.loc[common_fields].values
    y_field_true = le.transform(field_true_str.loc[common_fields].values)

    f1m = f1_score(y_field_true, y_field_pred, average="macro")
    print(f"--- FIELD-LEVEL TEST ---  fields={len(common_fields)}  "
          f"F1_macro={f1m:.4f}  kappa={cohen_kappa_score(y_field_true, y_field_pred):.4f}")

    import joblib
    joblib.dump(le, os.path.join(MODEL_DIR, "multi_channel_cnn_label_encoder.joblib"))

    # Report field-level metrics (matches the field-based convention used for the
    # 3D CNN patch model and the spatial-transfer table).
    report = ModelReport("Multi-Channel CNN Patch-Level (5-seed ensemble)", os.path.abspath(__file__))
    report.set_hyperparameters({
        "target_size": list(TARGET_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "channels": n_channels,
        "n_models": len(SEEDS),
        "seeds": SEEDS,
        "aggregation": "mean softmax over 5 seeds (patch-level), then majority vote by FID",
        "per_seed_seconds": per_seed_seconds,
    })
    report.set_split_info(train=len(ids_tr), test=len(common_fields),
                          seed=42, split_method="field-based (patch-level)")
    report.set_metrics(y_field_true, y_field_pred, le.classes_)
    report.set_training_history(first_history)
    report.set_training_time(_time.time() - t_train_start)
    report.add_notes(f"Per-seed training time (s): {per_seed_seconds}")
    report.generate()


if __name__ == "__main__":
    main()
