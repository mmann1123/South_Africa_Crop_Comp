"""
Inference script for the 3D‐CNN time‐aware patch classifier, with integrated
patch‐level and field‐level evaluation.
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
from collections import Counter

# Setting paths
PARQUET_PATH   = PATCH_DATA_PATH
MODEL_PATH     = os.path.join(MODEL_DIR, "conv3d_time_patch_level2.h5")
PATCH_CSV      = "patch_time_predictions.csv"
FIELD_CSV      = "field_time_predictions.csv"
TARGET_SIZE    = (128,128)
TEST_SIZE      = 0.20
SEED           = 42
CHUNK_SIZE     = 64
BAND_PREFIXES  = ['SA_B11','SA_B12','SA_B2','SA_B6','SA_EVI','SA_hue']

def group_band_columns(cols, prefixes):
    mapping = {}
    for p in prefixes:
        grp = sorted([c for c in cols if c.startswith(p)],
                     key=lambda x: int(x.split('_')[-1]))
        if not grp:
            raise ValueError(f"No columns for prefix {p}")
        mapping[p] = grp
    return mapping

def patch_pixels_to_image(df_patch, cols):
    r0,r1 = df_patch.row.min(), df_patch.row.max()
    c0,c1 = df_patch.col.min(), df_patch.col.max()
    H,W = (r1-r0)+1, (c1-c0)+1
    img = np.zeros((H, W, len(cols)), np.float32)
    for _, px in df_patch.iterrows():
        img[int(px.row-r0), int(px.col-c0), :] = [px[c] for c in cols]
    return img

def reconstruct_patch_time(df_patch, band_map):
    T = min(len(v) for v in band_map.values())
    bands = sorted(band_map)
    slices = []
    for b in bands:
        arr = patch_pixels_to_image(df_patch, band_map[b][:T])
        if arr.shape[-1] < T:
            arr = np.pad(arr, ((0,0),(0,0),(0, T-arr.shape[-1])), mode='constant')
        else:
            arr = arr[..., :T]
        slices.append(arr)
    # stack→ (H,W,T,B), transpose→ (T,H,W,B)
    cube = np.transpose(np.stack(slices, -1), (2,0,1,3))
    # resize all T at once → (T,128,128,B)
    return tf.image.resize(cube, TARGET_SIZE).numpy()

def compute_and_plot_cm(y_true, y_pred, classes, title, cmap):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_pct = np.nan_to_num(cm_pct)
    annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{title} Confusion Matrix (%)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    # 1) Load & drop NaN bands
    df = pd.read_parquet(PARQUET_PATH)
    ignore = {"patch_id","field_id","crop_name","row","col"}
    df = pd.concat([df[list(ignore)], df.drop(columns=ignore).dropna(axis=1)], axis=1)
    df = df.dropna(subset=["crop_name"])
    df = df[df.crop_name.str.lower()!='none']

    # 2) Split fields
    fields = df.field_id.dropna().unique()
    tr_f, te_f = train_test_split(fields, test_size=TEST_SIZE, random_state=SEED)
    df_test = df[df.field_id.isin(te_f)].copy()
    patch_ids = df_test.patch_id.unique().tolist()
    print(f"{len(patch_ids)} test patches across {len(te_f)} fields")

    # 3) Build band_map
    chans = [c for c in df.columns if c not in ignore]
    band_map = group_band_columns(chans, BAND_PREFIXES)

    # 4) Build LabelEncoder from train
    df_train = df[df.field_id.isin(tr_f)]
    le = LabelEncoder().fit(df_train.crop_name.unique())

    # 5) Load model
    model = models.load_model(MODEL_PATH)

    # 6) Chunked patch-level inference
    recs = []
    for i in range(0, len(patch_ids), CHUNK_SIZE):
        batch = patch_ids[i:i+CHUNK_SIZE]
        Xb, tb, fb, pb = [], [], [], []
        for pid in batch:
            sub = df_test[df_test.patch_id==pid]
            Xb.append(reconstruct_patch_time(sub, band_map))
            tb.append(sub.crop_name.iat[0])
            fb.append(sub.field_id.iat[0])
            pb.append(pid)
        Xb = np.stack(Xb,0)
        probs = model.predict(Xb, batch_size=len(Xb))
        idxs  = np.argmax(probs, axis=1)
        labs  = le.inverse_transform(idxs)
        for pid,fid,t,l in zip(pb,fb,tb,labs):
            recs.append((pid, fid, t, l))
        print(f"Processed patches {i}-{i+len(batch)}")
        del Xb, probs

    # 7) Patch-level DataFrame & CSV
    dfp = pd.DataFrame(recs, columns=["patch_id","field_id","true_label","predicted_label"])
    dfp.to_csv(PATCH_CSV, index=False)
    print(f"Saved patch-level → '{PATCH_CSV}'")
    print(dfp.head())

    # 8) Patch-level metrics
    y_true = dfp.true_label.values
    y_pred = dfp.predicted_label.values
    le_all = LabelEncoder().fit(np.concatenate([y_true, y_pred]))
    yt = le_all.transform(y_true)
    yp = le_all.transform(y_pred)
    oh = np.zeros((len(yp), len(le_all.classes_)), int)
    oh[np.arange(len(yp)), yp] = 1

    print("\n--- Patch-level Metrics ---")
    print(f"Accuracy    : {accuracy_score(yt,yp):.4f}")
    print(f"Cohen’s κ   : {cohen_kappa_score(yt,yp):.4f}")
    print(f"F1 Micro    : {f1_score(yt,yp,average='micro'):.4f}")
    print(f"F1 Macro    : {f1_score(yt,yp,average='macro'):.4f}")
    print(f"F1 Weighted : {f1_score(yt,yp,average='weighted'):.4f}")
    print(f"Entropy     : {log_loss(yt,oh,labels=np.arange(len(le_all.classes_))):.4f}")
    compute_and_plot_cm(yt, yp, le_all.classes_, "Patch-level", "Blues")

    # 9) Field-level aggregation & CSV
    fld = []
    for fid, grp in dfp.groupby("field_id"):
        true_mode = grp.true_label.mode().iat[0]
        pred_mode = grp.predicted_label.mode().iat[0]
        fld.append((fid, true_mode, pred_mode))
    dff = pd.DataFrame(fld, columns=["field_id","true_label","predicted_label"])
    dff.to_csv(FIELD_CSV, index=False)
    print(f"\nSaved field-level → '{FIELD_CSV}'")
    print(dff.head())

    # 10) Field-level metrics
    y_true_f = dff.true_label.values
    y_pred_f = dff.predicted_label.values
    yt_f = le_all.transform(y_true_f)
    yp_f = le_all.transform(y_pred_f)
    ohf = np.zeros((len(yp_f), len(le_all.classes_)), int)
    ohf[np.arange(len(yp_f)), yp_f] = 1

    print("\n--- Field-level Metrics ---")
    print(f"Accuracy    : {accuracy_score(yt_f,yp_f):.4f}")
    print(f"Cohen’s κ   : {cohen_kappa_score(yt_f,yp_f):.4f}")
    print(f"F1 Micro    : {f1_score(yt_f,yp_f,average='micro'):.4f}")
    print(f"F1 Macro    : {f1_score(yt_f,yp_f,average='macro'):.4f}")
    print(f"F1 Weighted : {f1_score(yt_f,yp_f,average='weighted'):.4f}")
    print(f"Entropy     : {log_loss(yt_f,ohf,labels=np.arange(len(le_all.classes_))):.4f}")
    compute_and_plot_cm(yt_f, yp_f, le_all.classes_, "Field-level", "Greens")

if __name__=="__main__":
    main()
