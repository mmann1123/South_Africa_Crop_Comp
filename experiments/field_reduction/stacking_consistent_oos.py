"""
Stacked ensemble consistent with the manuscript's documented architecture
(eq. for the Stacked Ensemble Classifier): Random Forest + XGBoost base learners,
Logistic Regression meta-learner with passthrough, at the FIELD level --- WITHOUT
SMOTE (the SMOTE variant is reported separately as "SMOTE Stacked").

Mirrors SMOTE_meta.py exactly except the SMOTETomek resampling step is removed,
so "Stacking" and "SMOTE Stacked" share the documented architecture and differ
only in resampling.

Evaluated on the in-region field test split and the spatially disjoint holdout
(combined_test_features.parquet). Replaces the previously reported 4-learner
Stacking row.

Outputs:
  - out_of_sample/predictions_stacking.csv  (field-level holdout predictions)
  - prints in-region and spatial-transfer metrics (F1m, kappa, wF1, Xent)
"""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deep_learn", "src"))
from config import FINAL_DATA_PATH, COMBINED_TEST_FEATURES_PATH, XGB_TUNER_DIR, REPO_ROOT

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import TEST_LABELS_GEOJSON

OUT_CSV = os.path.join(REPO_ROOT, "out_of_sample", "predictions_stacking.csv")


def ovr_hard_xent(y_true_lab, y_pred_lab):
    labels = sorted(pd.unique(y_true_lab))
    yt = label_binarize(y_true_lab, classes=labels)
    yp = np.clip(label_binarize(y_pred_lab, classes=labels).astype(float), 1e-7, 1 - 1e-7)
    return log_loss(yt, yp)


def metrics(y_true, y_pred, tag):
    f1m = f1_score(y_true, y_pred, average="macro")
    wf1 = f1_score(y_true, y_pred, average="weighted")
    kap = cohen_kappa_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    xent = ovr_hard_xent(y_true, y_pred)
    print(f"[{tag}] F1m={f1m:.4f} wF1={wf1:.4f} kappa={kap:.4f} acc={acc:.4f} Xent={xent:.4f} n={len(y_true)}")
    return dict(f1m=f1m, wf1=wf1, kappa=kap, acc=acc, xent=xent)


def aggregate_field(df):
    y = df.groupby("fid")["crop_name_encoded"].agg(lambda x: x.mode()[0])
    d = df.drop(columns=["crop_name_encoded", "crop_name"], errors="ignore")
    X = d.groupby("fid").mean(numeric_only=True).drop(columns=["crop_id", "SHAPE_AREA", "SHAPE_LEN"], errors="ignore")
    return X, y


def main():
    print("Loading training data...", flush=True)
    data = pd.read_parquet(FINAL_DATA_PATH)
    le = LabelEncoder()
    data["crop_name_encoded"] = le.fit_transform(data["crop_name"])
    print(f"  classes: {list(le.classes_)}", flush=True)

    fids = data["fid"].unique()
    tr_fids, te_fids = train_test_split(fids, test_size=0.2, random_state=42)
    tr_fids, _ = train_test_split(tr_fids, test_size=0.2, random_state=42)
    train_data = data[data["fid"].isin(tr_fids)].copy()
    test_data = data[data["fid"].isin(te_fids)].copy()
    del data; gc.collect()

    X_train, y_train = aggregate_field(train_data)
    X_test, y_test = aggregate_field(test_data)
    X_train = X_train.dropna(axis=1, how="all")
    feat_cols = list(X_train.columns)
    X_test = X_test[feat_cols]
    print(f"  field-level feature cols: {len(feat_cols)}; train fields={len(X_train)}, test fields={len(X_test)}", flush=True)

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    X_train_s = pd.DataFrame(sc.fit_transform(imp.fit_transform(X_train)), columns=feat_cols, index=X_train.index)
    X_test_s = pd.DataFrame(sc.transform(imp.transform(X_test)), columns=feat_cols, index=X_test.index)

    # base learners (match SMOTE_meta.py, minus SMOTE)
    rf = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=42)
    tuned_path = os.path.join(XGB_TUNER_DIR, "best_xgb_params.joblib")
    if os.path.exists(tuned_path):
        tp = joblib.load(tuned_path)
        xp = {k: v for k, v in tp.items() if k != "n_estimators"}
        xgb_model = xgb.XGBClassifier(n_estimators=tp.get("n_estimators", 1000), eval_metric="mlogloss",
                                      random_state=42, tree_method="hist", **xp)
        print("  using Optuna-tuned XGB params", flush=True)
    else:
        xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1, subsample=0.8,
                                      colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, tree_method="hist")
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    stacked = StackingClassifier(estimators=[("rf", rf), ("xgb", xgb_model)],
                                 final_estimator=meta, passthrough=True, n_jobs=-1)
    print("Training stacked ensemble (no SMOTE)...", flush=True)
    stacked.fit(X_train_s, y_train)

    # in-region
    yp_i = stacked.predict(X_test_s)
    inreg = metrics(le.inverse_transform(y_test.values), le.inverse_transform(yp_i), "IN-REGION")

    # holdout
    print("Loading holdout field features...", flush=True)
    hold = pd.read_parquet(COMBINED_TEST_FEATURES_PATH)
    hold = hold.drop(columns=["May"], errors="ignore")
    fids_h = hold["fid"].values
    Xh = hold.reindex(columns=feat_cols)  # align to training features; missing -> NaN
    Xh_s = pd.DataFrame(sc.transform(imp.transform(Xh)), columns=feat_cols)
    yp_h = stacked.predict(Xh_s)
    labels_h = le.inverse_transform(yp_h)
    pd.DataFrame({"fid": fids_h, "crop_name": labels_h}).to_csv(OUT_CSV, index=False)
    print(f"  saved {len(fids_h)} field predictions -> {OUT_CSV}", flush=True)

    gt = gpd.read_file(TEST_LABELS_GEOJSON)[["fid", "crop_name"]].copy()
    gt["fid"] = gt["fid"].astype(int)
    pred = pd.DataFrame({"fid": np.asarray(fids_h).astype(int), "crop_name": labels_h})
    m = gt.rename(columns={"crop_name": "t"}).merge(pred, on="fid", how="inner")
    st = metrics(m["t"].values, m["crop_name"].values, "SPATIAL-TRANSFER")

    print("\n==== SUMMARY (Stacking, RF+XGB+LR-meta, no SMOTE) ====")
    print(f"In-region  F1m={inreg['f1m']:.4f} kappa={inreg['kappa']:.4f} wF1={inreg['wf1']:.4f} Xent={inreg['xent']:.4f}")
    print(f"Spatial    F1m={st['f1m']:.4f} kappa={st['kappa']:.4f} wF1={st['wf1']:.4f} Xent={st['xent']:.4f}")
    print(f"Delta (ST - in-region) F1m = {st['f1m'] - inreg['f1m']:+.4f}")


if __name__ == "__main__":
    main()
