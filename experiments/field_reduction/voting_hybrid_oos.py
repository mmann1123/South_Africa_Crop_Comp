"""
Hybrid pixel-level voting (RF+XGB+LGBM soft vote + confidence-margin / mode
fallback) evaluated on BOTH the in-region test split and the spatially disjoint
holdout tile (34S_20E_259N).

This reproduces the ensemble documented in the manuscript (eqs for the
"Pixel-Level Ensemble with Hybrid Voting") and replaces the previously reported
field-level 4-learner Voting row.

Holdout pixel features are assembled from the aligned X_testing_*_34S_20E_259N
xr_fresh parquets (all 3,409,489 rows; the B12 grid mismatch only affected the
older testing_* set).

Outputs:
  - out_of_sample/predictions_voting.csv  (field-level hybrid predictions on holdout)
  - prints in-region and spatial-transfer metrics (F1m, kappa, wF1, Xent)
"""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deep_learn", "src"))
from config import FINAL_DATA_PATH, REPO_ROOT

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import TEST_LABELS_GEOJSON

FEAT = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/features"
BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
KEYS = ["id", "point", "fid"]
OUT_CSV = os.path.join(REPO_ROOT, "out_of_sample", "predictions_voting.csv")
CONF_THRESH = 0.10


def hybrid_field_preds(probs, fids):
    """Field-level hybrid: soft vote, fall back to pixel-label mode if top-2 gap < thresh."""
    C = probs.shape[1]
    cc = [f"c{i}" for i in range(C)]
    df = pd.DataFrame(probs, columns=cc)
    df["fid"] = fids
    df["pix"] = probs.argmax(1)
    fp = df.groupby("fid")[cc].mean()
    soft = fp.values.argmax(1)
    sorted_p = np.sort(fp.values, axis=1)
    gap = sorted_p[:, -1] - sorted_p[:, -2]
    mode = df.groupby("fid")["pix"].agg(lambda x: x.mode()[0]).reindex(fp.index).values
    hybrid = np.where(gap < CONF_THRESH, mode, soft)
    return fp.index.values, hybrid


def ovr_hard_xent(y_true_lab, y_pred_lab):
    labels = sorted(pd.unique(y_true_lab))
    yt = label_binarize(y_true_lab, classes=labels)
    yp = np.clip(label_binarize(y_pred_lab, classes=labels).astype(float), 1e-7, 1 - 1e-7)
    return log_loss(yt, yp)


def metrics(y_true_lab, y_pred_lab, tag):
    f1m = f1_score(y_true_lab, y_pred_lab, average="macro")
    wf1 = f1_score(y_true_lab, y_pred_lab, average="weighted")
    kap = cohen_kappa_score(y_true_lab, y_pred_lab)
    acc = accuracy_score(y_true_lab, y_pred_lab)
    xent = ovr_hard_xent(y_true_lab, y_pred_lab)
    print(f"[{tag}] F1m={f1m:.4f} wF1={wf1:.4f} kappa={kap:.4f} acc={acc:.4f} Xent={xent:.4f} n={len(y_true_lab)}")
    return dict(f1m=f1m, wf1=wf1, kappa=kap, acc=acc, xent=xent)


def main():
    # ---------- training data ----------
    print("Loading training data...", flush=True)
    data = pd.read_parquet(FINAL_DATA_PATH)
    le = LabelEncoder()
    data["y"] = le.fit_transform(data["crop_name"])
    print(f"  classes: {list(le.classes_)}", flush=True)

    tr_fids, te_fids = train_test_split(data["fid"].unique(), test_size=0.2, random_state=42)
    tr_fids, _ = train_test_split(tr_fids, test_size=0.2, random_state=42)
    train_data = data[data["fid"].isin(tr_fids)]
    test_data = data[data["fid"].isin(te_fids)].copy()
    del data; gc.collect()

    drop = ["y", "crop_name", "crop_id", "SHAPE_AREA", "SHAPE_LEN", "fid", "id", "point"]
    Xtr = train_data.drop(columns=drop, errors="ignore").dropna(axis=1, how="all")
    feat_cols = list(Xtr.columns)
    print(f"  feature cols: {len(feat_cols)}", flush=True)
    ytr = train_data["y"].astype(np.int32).values
    Xtr = Xtr.astype(np.float32)
    Xte = test_data[feat_cols].astype(np.float32)

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xtr_np = sc.fit_transform(imp.fit_transform(Xtr)).astype(np.float32)
    Xte_np = sc.transform(imp.transform(Xte)).astype(np.float32)
    del Xtr, train_data; gc.collect()

    # ---------- train 3 models (pixel_voting hyperparameters) ----------
    print("Training RF/XGB/LGBM...", flush=True)
    rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr_np, ytr)
    rf_te = rf.predict_proba(Xte_np)
    print("  RF done", flush=True)
    xgb = XGBClassifier(n_estimators=30, max_depth=6, learning_rate=0.1, tree_method="hist",
                        eval_metric="mlogloss", random_state=42, n_jobs=-1).fit(Xtr_np, ytr)
    xgb_te = xgb.predict_proba(Xte_np)
    print("  XGB done", flush=True)
    lgbm = LGBMClassifier(n_estimators=30, max_depth=6, learning_rate=0.1, random_state=42,
                          n_jobs=-1, verbose=-1).fit(Xtr_np, ytr)
    lgbm_te = lgbm.predict_proba(Xte_np)
    print("  LGBM done", flush=True)
    del Xtr_np; gc.collect()

    # ---------- in-region hybrid voting ----------
    te_probs = (rf_te + xgb_te + lgbm_te) / 3.0
    fids_i, hyb_i = hybrid_field_preds(te_probs, test_data["fid"].values)
    truth_i = test_data.groupby("fid")["y"].agg(lambda x: x.mode()[0]).reindex(fids_i).values
    inreg = metrics(le.inverse_transform(truth_i), le.inverse_transform(hyb_i), "IN-REGION")
    del Xte_np, test_data; gc.collect()

    # ---------- holdout pixel features (aligned X_testing_*) ----------
    print("Assembling holdout pixel features...", flush=True)
    parts = []
    for i, b in enumerate(BANDS):
        d = pd.read_parquet(f"{FEAT}/X_testing_{b}_34S_20E_259N.parquet")
        bandcols = [c for c in d.columns if c in feat_cols]
        if i == 0:
            parts.append(d.set_index(KEYS)[bandcols])
        else:
            parts.append(d.set_index(KEYS)[bandcols])
        del d; gc.collect()
    hold = pd.concat(parts, axis=1)
    del parts; gc.collect()
    hold = hold.reset_index()
    print(f"  holdout shape: {hold.shape}, fields: {hold['fid'].nunique()}", flush=True)

    Xh = hold[feat_cols].astype(np.float32)
    Xh_np = sc.transform(imp.transform(Xh)).astype(np.float32)
    del Xh; gc.collect()

    h_probs = (rf.predict_proba(Xh_np) + xgb.predict_proba(Xh_np) + lgbm.predict_proba(Xh_np)) / 3.0
    fids_h, hyb_h = hybrid_field_preds(h_probs, hold["fid"].values)
    labels_h = le.inverse_transform(hyb_h)
    pd.DataFrame({"fid": fids_h, "crop_name": labels_h}).to_csv(OUT_CSV, index=False)
    print(f"  saved {len(fids_h)} field predictions -> {OUT_CSV}", flush=True)

    # ---------- score holdout vs ground truth ----------
    gt = gpd.read_file(TEST_LABELS_GEOJSON)[["fid", "crop_name"]].copy()
    gt["fid"] = gt["fid"].astype(int)
    pred = pd.DataFrame({"fid": fids_h.astype(int), "crop_name": labels_h})
    m = gt.rename(columns={"crop_name": "t"}).merge(pred, on="fid", how="inner")
    st = metrics(m["t"].values, m["crop_name"].values, "SPATIAL-TRANSFER")

    print("\n==== SUMMARY (Voting hybrid) ====")
    print(f"In-region  F1m={inreg['f1m']:.4f} kappa={inreg['kappa']:.4f} wF1={inreg['wf1']:.4f} Xent={inreg['xent']:.4f}")
    print(f"Spatial    F1m={st['f1m']:.4f} kappa={st['kappa']:.4f} wF1={st['wf1']:.4f} Xent={st['xent']:.4f}")
    print(f"Delta (ST - in-region) F1m = {st['f1m'] - inreg['f1m']:+.4f}")


if __name__ == "__main__":
    main()
