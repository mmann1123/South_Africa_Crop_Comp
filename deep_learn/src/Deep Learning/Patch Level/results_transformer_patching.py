#!/usr/bin/env python3
"""
inference script for the Transformer‐based model
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
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

PATCH_PARQUET = PATCH_DATA_PATH
WEIGHTS_FILE  = os.path.join(MODEL_DIR, "transformer_time_model.h5")
PATCH_OUT_CSV = "transformer_patch_predictions.csv"
FIELD_OUT_CSV = "field_level_predictions.csv"
TARGET_SIZE   = (128, 128)
TEST_SIZE     = 0.20
SEED          = 42
BAND_PREFIXES = ['SA_B11','SA_B12','SA_B2','SA_B6','SA_EVI','SA_hue']
CHUNK_SIZE    = 32   # number of patches to process in one go

def group_band_columns(cols, prefixes):
    mapping = {}
    for p in prefixes:
        lst = [c for c in cols if c.startswith(p)]
        if not lst:
            raise ValueError(f"No columns for band prefix {p}")
        mapping[p] = sorted(lst, key=lambda x: int(x.rsplit('_',1)[-1]))
    return mapping

def patch_pixels_to_image(df_patch, cols):
    r0, r1 = df_patch.row.min(), df_patch.row.max()
    c0, c1 = df_patch.col.min(), df_patch.col.max()
    H, W = int(r1 - r0 + 1), int(c1 - c0 + 1)
    img = np.zeros((H, W, len(cols)), np.float32)
    for _, px in df_patch.iterrows():
        rr, cc = int(px.row - r0), int(px.col - c0)
        img[rr, cc, :] = px[cols].values
    return img

def reconstruct_patch_time(df_patch, band_map):
    T = min(len(v) for v in band_map.values())
    imgs = []
    for b in sorted(band_map):
        arr = patch_pixels_to_image(df_patch, band_map[b][:T])
        if arr.shape[-1] < T:
            pad = T - arr.shape[-1]
            arr = np.pad(arr, ((0,0),(0,0),(0,pad)), mode='constant')
        else:
            arr = arr[..., :T]
        imgs.append(arr)
    cube = np.transpose(np.stack(imgs, axis=-1), (2,0,1,3))  # (T,H,W,B)
    return tf.image.resize(cube, TARGET_SIZE).numpy()

def build_transformer_model(time_steps, num_bands, num_classes):
    inp = layers.Input((time_steps, *TARGET_SIZE, num_bands))
    x = layers.TimeDistributed(layers.Conv2D(32,(1,1),activation='relu'))(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    pos = tf.range(time_steps)
    pe  = layers.Embedding(time_steps,32)(pos)
    x  += pe
    def block(x):
        a  = layers.MultiHeadAttention(4,32)(x,x)
        x1 = layers.Add()([x, layers.Dropout(0.1)(a)])
        f  = layers.Dense(64,'relu')(x1)
        f  = layers.Dense(32)(f)
        return layers.Add()([x1, layers.Dropout(0.1)(f)])
    x = block(x); x = block(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inp,out)

def main():
    # 1) load & clean parquet
    df = pd.read_parquet(PATCH_PARQUET)
    ignore = ['patch_id','field_id','crop_name','row','col']
    df = pd.concat([ df[ignore], df.drop(columns=ignore).dropna(axis=1) ], axis=1)
    df = df.dropna(subset=['crop_name'])
    df = df[df.crop_name.str.lower()!='none']

    # 2) split fields → test set
    fields     = df.field_id.dropna().unique()
    tr_f, te_f = train_test_split(fields, test_size=TEST_SIZE, random_state=SEED)
    df_test    = df[df.field_id.isin(te_f)].copy()
    test_pids  = df_test.patch_id.unique().tolist()
    print(f"Test: {len(test_pids)} patches across {len(te_f)} fields")

    # 3) group bands, determine dims
    channels = [c for c in df.columns if c not in ignore]
    band_map = group_band_columns(channels,BAND_PREFIXES)
    T        = min(len(v) for v in band_map.values())
    B        = len(BAND_PREFIXES)

    # 4) build label encoder from train split
    df_train    = df[df.field_id.isin(tr_f)]
    le          = LabelEncoder().fit(df_train.crop_name.unique())
    num_classes = len(le.classes_)

    # 5) build model & load weights
    model = build_transformer_model(T,B,num_classes)
    model.load_weights(WEIGHTS_FILE)
    print("Model weights loaded.")

    # 6) chunked inference with status prints
    num_chunks = math.ceil(len(test_pids) / CHUNK_SIZE)
    patch_recs = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        chunk = test_pids[start:start+CHUNK_SIZE]
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} ({len(chunk)} patches)...")

        Xb, info = [], []
        for pid in chunk:
            sub = df_test[df_test.patch_id==pid]
            Xb.append(reconstruct_patch_time(sub,band_map))
            info.append((pid, sub.field_id.iat[0], sub.crop_name.iat[0]))

        Xb   = np.stack(Xb,axis=0)        # (batch,T,128,128,B)
        probs = model.predict(Xb, verbose=0)
        idxs  = np.argmax(probs, axis=1)

        for (pid,fid,true_lbl), idx in zip(info, idxs):
            patch_recs.append({
                'patch_id': pid,
                'field_id': fid,
                'true_label': true_lbl,
                'predicted_label': le.inverse_transform([idx])[0]
            })

        # free memory
        del Xb, probs

    # 7) save patch‐level outputs & metrics
    dfp = pd.DataFrame(patch_recs)
    dfp.to_csv(PATCH_OUT_CSV, index=False)
    print(f"Saved patch‐level predictions → {PATCH_OUT_CSV}")

    acc_p   = accuracy_score(dfp.true_label, dfp.predicted_label)
    kappa_p = cohen_kappa_score(dfp.true_label, dfp.predicted_label)
    print(f"\nPatch‐level Accuracy: {acc_p:.4f}")
    print(f"Patch‐level Cohen’s Kappa: {kappa_p:.4f}")

    # 8) majority‐vote → field & save
    field_recs = []
    for fid, grp in dfp.groupby('field_id'):
        field_recs.append({
            'field_id': fid,
            'true_label': grp.true_label.mode().iat[0],
            'predicted_label': grp.predicted_label.mode().iat[0]
        })
    dff = pd.DataFrame(field_recs)
    dff.to_csv(FIELD_OUT_CSV, index=False)
    print(f"\nSaved field‐level predictions → {FIELD_OUT_CSV}")

    # 9) field‐level metrics
    acc_f   = accuracy_score(dff.true_label, dff.predicted_label)
    kappa_f = cohen_kappa_score(dff.true_label, dff.predicted_label)
    print(f"\nField‐level Accuracy: {acc_f:.4f}")
    print(f"Field‐level Cohen’s Kappa: {kappa_f:.4f}")

if __name__=="__main__":
    main()
