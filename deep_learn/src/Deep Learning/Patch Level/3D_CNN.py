"""
Multi-temporal, multi-spectral crop classification using a 3D CNN.
The input patches are restructured to incorporate time.
Instead of using 60 channels (6 bands x ~10 months) as flat channels,
we group them into 6 spectral bands and T time steps.
The final input to the network is of shape (T, 128, 128, 6).
Train/test split is based on field_id.

Data pipeline (optimized): each patch's pixels are pre-extracted once into a
numpy store; patches are reconstructed with vectorized indexing, resized in a
single tf.image.resize call (T treated as the batch dim), and streamed through
tf.data with parallel map + prefetch so the GPU is not starved. Validation runs
as a single batched predict per epoch. This replaces the previous per-pixel
df.iterrows() generator that left the GPU ~20% utilized.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_DATA_PATH, MODEL_DIR

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from collections import Counter
import joblib

PATCH_PARQUET = PATCH_DATA_PATH
TARGET_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 30

BAND_PREFIXES = ['SA_B11', 'SA_B12', 'SA_B2', 'SA_B6', 'SA_EVI', 'SA_hue']


class WeightedSparseFocalLoss(tf.keras.losses.Loss):
    """Weighted Focal Loss for sparse categorical labels (TF/Keras)."""

    def __init__(self, alpha, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # reshape (not squeeze) to a known rank-1 vector; under TF 2.18/Keras 3
        # tf.squeeze on a generator-fed tensor yields an unknown-rank tensor and
        # sparse_categorical_crossentropy then fails ("unknown rank").
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred)
        pt = tf.reduce_sum(y_pred * tf.one_hot(y_true_int, num_classes), axis=-1)
        alpha_t = tf.gather(self.alpha, y_true_int)
        return alpha_t * tf.pow(1.0 - pt, self.gamma) * ce


class F1MacroCheckpoint(tf.keras.callbacks.Callback):
    """Each epoch: one batched validation predict -> F1 macro, save best weights,
    and early-stop when val F1 hasn't improved for `patience` epochs."""

    def __init__(self, val_ds, y_true_val, model_path, patience=6):
        super().__init__()
        self.val_ds = val_ds            # deterministic (unshuffled) tf.data val set
        self.y_true_val = np.asarray(y_true_val)
        self.model_path = model_path
        self.best_f1 = 0.0
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        f1 = f1_score(self.y_true_val, y_pred, average='macro')
        print(f"  val_f1_macro={f1:.4f}", end="")
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            self.model.save_weights(self.model_path)
            print(" [saved]", end="")
        else:
            self.wait += 1
            print(f" [no improve {self.wait}/{self.patience}]", end="")
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(" [early stop]", end="")
        print()


def group_band_columns(channel_cols, band_prefixes):
    """Map each band prefix to its columns sorted by trailing month number."""
    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        if len(matching) == 0:
            raise ValueError(f"No columns found for band prefix: {prefix}")

        def extract_month(col):
            parts = col.split('_')
            try:
                return int(parts[-1])
            except Exception:
                return 0

        band_mapping[prefix] = sorted(matching, key=extract_month)
    return band_mapping


def build_patch_store(df, patch_ids, col_order, label_encoder):
    """Pre-extract each patch's pixels ONCE (vectorized, no df.iterrows()).

    For every patch we keep:
      rr, cc : int16 row/col offsets within the patch
      vals   : float32 [n_pixels, num_bands*T] in `col_order` (band-major, time-minor)
    Patches are reconstructed/resized lazily per batch in the tf.data pipeline,
    so only the raw pixels live in RAM (not the 128x128 tensors).
    """
    grouped = df.groupby("patch_id")
    store, labels, valid_ids = {}, {}, []
    for pid in patch_ids:
        try:
            grp = grouped.get_group(pid)
        except KeyError:
            continue
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
        vals = grp[col_order].values.astype(np.float32)
        store[pid] = (rr, cc, vals)
        labels[pid] = int(label_encoder.transform([crop_str])[0])
        valid_ids.append(pid)
        if len(valid_ids) % 5000 == 0:
            print(f"  Indexed {len(valid_ids)} patches...")
    print(f"  Total: {len(valid_ids)} patches")
    return store, labels, valid_ids


def reconstruct_patch(pid, store, num_bands, T):
    """Vectorized reconstruction -> raw [T, H, W, num_bands] float32 (native H,W).

    Band axis follows sorted(band prefixes); time axis follows month order,
    matching the original generator's (T, H, W, num_bands) layout.
    """
    rr, cc, vals = store[pid]
    H = int(rr.max()) + 1
    W = int(cc.max()) + 1
    n_pix = vals.shape[0]
    img = np.zeros((H, W, num_bands, T), dtype=np.float32)
    img[rr, cc, :, :] = vals.reshape(n_pix, num_bands, T)
    return np.transpose(img, (3, 0, 1, 2))  # (T, H, W, num_bands)


def make_dataset(ids, store, labels, num_bands, T, target_size, batch_size,
                 shuffle, seed=None):
    """tf.data pipeline: parallel vectorized reconstruct (py_function) + in-graph
    resize (T as batch dim) + prefetch. Deterministic order when not shuffling so
    predictions align across seeds for ensemble averaging."""
    ids = list(ids)
    th, tw = target_size

    def _py(i):
        pid = ids[int(i)]
        img = reconstruct_patch(pid, store, num_bands, T)
        return img, np.int64(labels[pid])

    def _load(i):
        img, lab = tf.py_function(_py, [i], [tf.float32, tf.int64])
        # py_function output has unknown rank; declare it (rank 4, dynamic H/W)
        # so tf.image.resize — which treats the leading T dim as the batch — works.
        img.set_shape([T, None, None, num_bands])
        img = tf.image.resize(img, target_size)            # (T, th, tw, num_bands)
        img.set_shape([T, th, tw, num_bands])
        lab.set_shape([])
        return img, lab

    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(ids)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(ids), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    import time as _time
    t_train_start = _time.time()

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

    # Field-based split (avoids leakage), identical seed to all other models.
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

    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])
    print("Channel columns used:", channel_cols, "=> count:", len(channel_cols))

    band_mapping = group_band_columns(channel_cols, BAND_PREFIXES)
    num_bands = len(band_mapping)
    T = min(len(v) for v in band_mapping.values())
    print(f"Using {T} time steps for each band.")

    # Column order for the store: band-major, time-minor (matches reconstruct_patch).
    sorted_prefixes = sorted(band_mapping.keys())
    col_order = [band_mapping[p][t] for p in sorted_prefixes for t in range(T)]

    # Class weights for focal loss (per-patch class frequencies on the train split).
    patch_crop_map = {}
    for pid, grp_crop in df_train.groupby("patch_id")["crop_name"]:
        crops = grp_crop.unique()
        if len(crops) == 1:
            patch_crop_map[pid] = crops[0]
    patch_class_counts = Counter(patch_crop_map.values())
    class_counts_arr = np.array([patch_class_counts.get(cls, 1) for cls in le.classes_], dtype=np.float64)
    class_counts_arr = np.maximum(class_counts_arr, 1.0)
    alpha = 1.0 / class_counts_arr
    alpha = alpha / alpha.sum() * num_classes
    print(f"Class weights (alpha): {alpha.tolist()}")

    # Pre-extract patches once (the expensive df work happens here, not per epoch).
    print("\nIndexing training patches...")
    store_tr, labels_tr, ids_tr = build_patch_store(df_train, train_ids, col_order, le)
    print("Indexing test patches...")
    store_te, labels_te, ids_te = build_patch_store(df_test, test_ids, col_order, le)

    if len(ids_tr) == 0:
        print("No training patches found. Exiting.")
        return

    # Deterministic validation set + its label vector (fixed across seeds).
    val_ds = make_dataset(ids_te, store_te, labels_te, num_bands, T, TARGET_SIZE,
                          BATCH_SIZE, shuffle=False)
    y_true_val = np.array([labels_te[pid] for pid in ids_te], dtype=np.int64)

    input_shape = (T, TARGET_SIZE[0], TARGET_SIZE[1], num_bands)  # (T, 128, 128, 6)

    # 5-seed ensemble: train one model per seed, average softmax probabilities.
    SEEDS = [42, 101, 202, 303, 404]
    prob_sum = None
    first_history = None
    per_seed_seconds = {}

    for si, seed in enumerate(SEEDS):
        print(f"\n===== Seed {seed} ({si + 1}/{len(SEEDS)}) =====")
        seed_start = _time.time()
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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
                      loss=WeightedSparseFocalLoss(alpha, gamma=2.0),
                      metrics=['accuracy'])
        if si == 0:
            model.summary()

        train_ds = make_dataset(ids_tr, store_tr, labels_tr, num_bands, T, TARGET_SIZE,
                                BATCH_SIZE, shuffle=True, seed=seed)

        best_weights_path = os.path.join(MODEL_DIR, f"conv3d_best_weights_seed{seed}.weights.h5")
        f1_callback = F1MacroCheckpoint(val_ds, y_true_val, best_weights_path, patience=6)

        # No validation_data: the callback runs the single batched val pass per epoch
        # and handles best-weight saving + early stopping.
        history = model.fit(train_ds, epochs=EPOCHS, callbacks=[f1_callback])
        if first_history is None:
            first_history = history.history

        if os.path.exists(best_weights_path):
            model.load_weights(best_weights_path)
            print(f"Loaded best weights (F1 macro={f1_callback.best_f1:.4f})")

        # Deterministic test pass -> probabilities aligned with ids_te across seeds.
        seed_probs = model.predict(val_ds, verbose=0)
        prob_sum = seed_probs if prob_sum is None else prob_sum + seed_probs

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(os.path.join(MODEL_DIR, f"conv3d_time_patch_level_seed{seed}.h5"))

        seed_elapsed = _time.time() - seed_start
        per_seed_seconds[str(seed)] = round(seed_elapsed, 1)
        print(f"Seed {seed} done in {seed_elapsed / 60:.1f} min")

    # Ensemble: average softmax probabilities across seeds -> per-patch prediction.
    ensemble_probs = prob_sum / len(SEEDS)
    y_pred_patch = np.argmax(ensemble_probs, axis=1)
    y_true_patch = y_true_val

    # Patch-level (secondary, for reference only).
    acc_p = accuracy_score(y_true_patch, y_pred_patch)
    kappa_p = cohen_kappa_score(y_true_patch, y_pred_patch)
    print("\n--- PATCH-LEVEL TEST (5-seed ensemble) ---")
    print(f"Accuracy: {acc_p:.4f}  Cohen's Kappa: {kappa_p:.4f}")

    # Aggregate patch predictions to FIELD level (majority vote by field_id).
    # This is the project's primary reporting unit and matches the Multi-Channel
    # CNN and the field-level temporal models, so metrics are directly comparable.
    patch_to_field = df_test.groupby("patch_id")["field_id"].first()
    patch_df = pd.DataFrame({"patch_id": ids_te, "pred": y_pred_patch})
    patch_df["field_id"] = patch_df["patch_id"].map(patch_to_field)
    field_pred = patch_df.groupby("field_id")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])
    field_true_str = df_test.groupby("field_id")["crop_name"].first()
    common_fields = field_pred.index.intersection(field_true_str.index)
    y_field_pred = field_pred.loc[common_fields].values
    y_field_true = le.transform(field_true_str.loc[common_fields].values)

    f1m = f1_score(y_field_true, y_field_pred, average="macro")
    kappa_f = cohen_kappa_score(y_field_true, y_field_pred)
    cm = confusion_matrix(y_field_true, y_field_pred)
    print(f"\n--- FIELD-LEVEL TEST (5-seed ensemble) ---  fields={len(common_fields)}")
    print(f"F1_macro: {f1m:.4f}  Cohen's Kappa: {kappa_f:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Per-seed training time (s):", per_seed_seconds)

    joblib.dump(le, os.path.join(MODEL_DIR, "3d_cnn_label_encoder.joblib"))
    print("Label encoder saved to", os.path.join(MODEL_DIR, "3d_cnn_label_encoder.joblib"))

    # ===================== REPORT =====================
    from report import ModelReport

    report = ModelReport("3D CNN Patch-Level (5-seed ensemble)")
    report.set_hyperparameters({
        "target_size": list(TARGET_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam(lr=1e-4)",
        "loss": "WeightedSparseFocalLoss(gamma=2.0, alpha=1/class_counts)",
        "model_selection": "val F1 macro (maximize)",
        "early_stopping_patience": 6,
        "epochs_note": "max EPOCHS with early stopping on val F1; single batched val pass per epoch",
        "data_pipeline": "numpy patch store + vectorized reconstruct + tf.data parallel map/prefetch",
        "conv3d_filters": [32, 64, 128],
        "bands": BAND_PREFIXES,
        "time_steps": T,
        "n_models": len(SEEDS),
        "seeds": SEEDS,
        "reporting_level": "field-level (majority vote by FID)",
        "aggregation": "mean softmax over 5 seeds (patch-level) -> argmax -> majority vote by FID",
        "patch_level_accuracy": round(float(acc_p), 4),
        "patch_level_kappa": round(float(kappa_p), 4),
        "per_seed_seconds": per_seed_seconds,
    })
    # Field-level split info (test = number of fields) to match the other models.
    report.set_split_info(train=len(ids_tr), test=len(common_fields), seed=42,
                          split_method="field-based (patch-level); metrics aggregated to FID")
    report.set_metrics(y_field_true, y_field_pred, le.classes_)
    report.set_training_history(first_history)
    report.set_training_time(_time.time() - t_train_start)
    report.add_notes(f"Field-level metrics (majority vote by FID); patch-level acc={acc_p:.4f}, "
                     f"kappa={kappa_p:.4f}. Per-seed training time (s): {per_seed_seconds}")
    report.generate()


if __name__ == "__main__":
    main()
