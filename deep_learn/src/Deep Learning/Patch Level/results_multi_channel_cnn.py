#!/usr/bin/env python3
"""
inference script matching the original 2D-CNN training pipeline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_DATA_PATH, MODEL_DIR

import math, numpy as np, pandas as pd, tensorflow as tf
import seaborn as sns, matplotlib.pyplot as plt

from tensorflow.keras import models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    f1_score, log_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split

PARQUET_PATH    = PATCH_DATA_PATH
MODEL_PATH      = os.path.join(MODEL_DIR, "patch_level_cnn.h5")

OUTPUT_CSV      = "patch_level_predictions.csv"
TARGET_SIZE     = (128,128)
TEST_SIZE       = 0.20
SEED            = 42
CHUNK_SIZE      = 128  # for memory sigkill issues

def patch_pixels_to_image(df_patch, channel_cols):
    r0,r1 = df_patch.row.min(), df_patch.row.max()
    c0,c1 = df_patch.col.min(), df_patch.col.max()
    H,W = (r1-r0)+1, (c1-c0)+1
    img = np.zeros((H, W, len(channel_cols)), dtype=np.float32)
    for _, px in df_patch.iterrows():
        img[int(px.row-r0), int(px.col-c0), :] = [px[c] for c in channel_cols]
    return img

def resize_image(img, target=TARGET_SIZE):
    t = tf.expand_dims(tf.convert_to_tensor(img, tf.float32), 0)
    r = tf.image.resize(t, target)
    return tf.squeeze(r, 0).numpy()

def main():
    # 1) Load full parquet
    df = pd.read_parquet(PARQUET_PATH)
    # 2) Drop NaN‐containing band cols (like training)
    ignore_cols = {"patch_id","field_id","crop_name","row","col"}
    df_ignore = df[list(ignore_cols)]
    chans = [c for c in df.columns if c not in ignore_cols]
    df_chan = df[chans].dropna(axis=1)
    df = pd.concat([df_ignore, df_chan], axis=1)
    channel_cols = sorted(df_chan.columns)
    print(f"Using {len(channel_cols)} channels (dropped {len(chans)-len(channel_cols)})")

    # 3) Filter and split on field_id
    df = df.dropna(subset=["crop_name"])
    df = df[df.crop_name.str.lower()!="none"]
    fields = df.field_id.dropna().unique()
    tr_f, te_f = train_test_split(fields, test_size=TEST_SIZE, random_state=SEED)
    df_test = df[df.field_id.isin(te_f)].copy()
    patch_ids = df_test.patch_id.unique()
    print(f"{len(patch_ids)} test patches across {len(te_f)} fields")

    # 4) Build LabelEncoder from train split
    df_train = df[df.field_id.isin(tr_f)]
    le = LabelEncoder().fit(df_train.crop_name.unique())

    # 5) Load model
    model = models.load_model(MODEL_PATH)

    # 6) Chunked inference
    all_preds, all_truth, all_fids, all_pids = [], [], [], []
    for i in range(0, len(patch_ids), CHUNK_SIZE):
        batch = patch_ids[i:i+CHUNK_SIZE]
        imgs, truths, fids, pids = [], [], [], []
        for pid in batch:
            sub = df_test[df_test.patch_id==pid]
            img = patch_pixels_to_image(sub, channel_cols)
            img = resize_image(img)
            imgs.append(img)
            truths.append(sub.crop_name.iat[0])
            fids.append(sub.field_id.iat[0])
            pids.append(pid)
        X = np.stack(imgs,0)
        probs = model.predict(X, batch_size=len(imgs))
        idxs = np.argmax(probs, axis=1)
        preds = le.inverse_transform(idxs)
        print(f"Processed chunk {i}–{i+len(batch)}")
        all_preds.extend(preds)
        all_truth.extend(truths)
        all_fids.extend(fids)
        all_pids.extend(pids)
        del X, imgs, probs, idxs, preds

    # 7) Save predictions CSV
    out = pd.DataFrame({
        "patch_id":        all_pids,
        "field_id":        all_fids,
        "true_label":      all_truth,
        "predicted_label": all_preds
    })
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions → '{OUTPUT_CSV}'")

    # 8) Compute metrics
    y_true = np.array(all_truth)
    y_pred = np.array(all_preds)
    le_all = LabelEncoder().fit(np.concatenate([y_true, y_pred]))
    yt = le_all.transform(y_true)
    yp = le_all.transform(y_pred)
    oh = np.zeros((len(yp), len(le_all.classes_)), int)
    oh[np.arange(len(yp)), yp] = 1

    print("\n=== Test Metrics ===")
    print(f"Accuracy    : {accuracy_score(yt,yp):.4f}")
    print(f"Cohen κ     : {cohen_kappa_score(yt,yp):.4f}")
    print(f"F1 Micro    : {f1_score(yt,yp,average='micro'):.4f}")
    print(f"F1 Macro    : {f1_score(yt,yp,average='macro'):.4f}")
    print(f"F1 Weighted : {f1_score(yt,yp,average='weighted'):.4f}")
    print(f"Entropy     : {log_loss(yt, oh, labels=np.arange(len(le_all.classes_))):.4f}")

    # 9) Confusion matrix plot
    cm = confusion_matrix(yt, yp, labels=np.arange(len(le_all.classes_)))
    cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
    cm_pct = np.nan_to_num(cm_pct)
    annot = [[f"{v:.1f}%" for v in row] for row in cm_pct]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
                xticklabels=le_all.classes_, yticklabels=le_all.classes_)
    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
