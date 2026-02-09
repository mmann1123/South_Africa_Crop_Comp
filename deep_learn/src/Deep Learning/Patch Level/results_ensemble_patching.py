"""
Inference script for the labeled test set (using the pickled DataFrame from training).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MODEL_DIR

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import models
from joblib           import load
from collections      import Counter
from sklearn.metrics  import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

# Setting paths
TEST_DF_PATH       = os.path.join(MODEL_DIR, "test_df.pkl")
BASE_MODEL_DIR     = MODEL_DIR
META_MODEL_PATH    = os.path.join(MODEL_DIR, "meta_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
OUTPUT_CSV         = "results_ensemble_patching.csv"

BAND_PREFIXES = ["SA_B11","SA_B12","SA_B2","SA_B6","SA_EVI","SA_hue"]
TARGET_SIZE   = (128,128)

def group_band_columns(cols, prefixes):
    d = {}
    for p in prefixes:
        lst = [c for c in cols if c.startswith(p)]
        d[p] = sorted(lst, key=lambda x:int(x.split("_")[-1]))
    return d

def patch_pixels_to_image(df_patch, cols):
    r0,r1 = df_patch.row.min(), df_patch.row.max()
    c0,c1 = df_patch.col.min(), df_patch.col.max()
    H,W = (r1-r0)+1, (c1-c0)+1
    img = np.zeros((H,W,len(cols)), dtype=np.float32)
    for _,px in df_patch.iterrows():
        img[int(px.row-r0), int(px.col-c0), :] = [px[c] for c in cols]
    return img

def resize(img, target=TARGET_SIZE):
    return tf.squeeze(
        tf.image.resize(tf.expand_dims(img,0), target),
        0
    ).numpy()

def reconstruct_patch(df, pid, band_map):
    sub = df[df.patch_id==pid]
    T = min(len(v) for v in band_map.values())
    stacks = []
    for p in sorted(band_map):
        img = patch_pixels_to_image(sub, band_map[p][:T])
        if img.shape[-1] < T:
            pad = T - img.shape[-1]
            img = np.pad(img, ((0,0),(0,0),(0,pad)))
        stacks.append(img)
    arr = np.stack(stacks, -1)            # H,W,T,B
    cube = np.transpose(arr, (2,0,1,3))   # T,H,W,B
    # vectorized resize all timesteps
    return tf.image.resize(cube, TARGET_SIZE).numpy()  # (T,128,128,B)

def main():
    # 1) Load the test DataFrame (with labels) from pickle
    with open(TEST_DF_PATH, "rb") as f:
        df_test = pickle.load(f)
    print("Loaded test_df:", df_test.shape)

    # 2) Build band_map
    ignore = {"patch_id","field_id","crop_name","row","col"}
    chan_cols = [c for c in df_test.columns if c not in ignore]
    band_map = group_band_columns(chan_cols, BAND_PREFIXES)

    # 3) Load base models
    model_files = sorted(
        f for f in os.listdir(BASE_MODEL_DIR)
        if f.startswith("conv3d_model_class_") and f.endswith(".h5")
    )
    classes = sorted(fn.replace("conv3d_model_class_","").replace(".h5","")
                     for fn in model_files)
    base_models = {
        cls: models.load_model(os.path.join(BASE_MODEL_DIR, fn))
        for cls, fn in zip(classes, model_files)
    }
    print("Loaded base models:", classes)

    # 4) Load meta-model and encoder
    meta = load(META_MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    print("Loaded meta-model and label encoder.")

    # 5) Predict patch-level
    records = []
    for pid in df_test.patch_id.unique():
        # reconstruct + batch dim
        X = np.expand_dims(reconstruct_patch(df_test, pid, band_map), 0)
        # collect probs from each base model
        probs = np.array([base_models[c].predict(X)[0][0] for c in classes]).reshape(1,-1)
        raw = meta.predict(probs)[0]
        code = int(raw)
        pred_lbl = le.inverse_transform([code])[0]
        sub = df_test[df_test.patch_id==pid]
        true_lbl = sub.crop_name.iloc[0]
        fid = sub.field_id.iloc[0]
        records.append({
            "patch_id": pid,
            "field_id": fid,
            "true_label": true_lbl,
            "predicted_label": pred_lbl
        })
    patch_df = pd.DataFrame(records)

    # 6) Immediately save field-level predictions CSV
    field_recs = []
    for fid, grp in patch_df.groupby("field_id"):
        maj = Counter(grp.predicted_label).most_common(1)[0][0]
        field_recs.append({"field_id": fid, "predicted": maj})
    field_df = pd.DataFrame(field_recs)
    field_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved field-level predictions → '{OUTPUT_CSV}'")

    # 7) Compute & print patch-level metrics
    p_acc   = accuracy_score(patch_df.true_label,      patch_df.predicted_label)
    p_kappa = cohen_kappa_score(patch_df.true_label,   patch_df.predicted_label)
    print(f"\nPatch-level   Acc: {p_acc:.4f} | Cohen’s κ: {p_kappa:.4f}")
    cm_p = confusion_matrix(patch_df.true_label, patch_df.predicted_label,
                            labels=le.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_p, annot=True, fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap="Blues")
    plt.title("Patch-level Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # 8) Compute & print field-level metrics
    truth_map = {fid: grp.true_label.mode().iat[0]
                 for fid, grp in patch_df.groupby("field_id")}
    field_df["true"] = field_df.field_id.map(truth_map)
    f_acc   = accuracy_score(field_df.true, field_df.predicted)
    f_kappa = cohen_kappa_score(field_df.true, field_df.predicted)
    print(f"\nField-level   Acc: {f_acc:.4f} | Cohen’s κ: {f_kappa:.4f}")
    cm_f = confusion_matrix(field_df.true, field_df.predicted,
                            labels=le.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_f, annot=True, fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap="Greens")
    plt.title("Field-level Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
