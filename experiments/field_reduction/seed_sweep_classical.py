"""Classical arm of the subsample-seed variance sweep: XGBoost (field) + Logistic Regression (pixel).

These are the robust controls. They are deterministic given the data (no training-seed ensemble),
so varying the subsample seed isolates draw-to-draw variance directly. Pipelines replicate
train_xgboost_field.py / train_base_ml.py and their OOS inference in predict_oos.py exactly; OOS is
scored inline against the holdout GeoJSON. Rows append to the same results CSV as the other arms.
Run in the ml_field env (XGBoost GPU + sklearn).

Usage:
    python seed_sweep_classical.py --models xgboost_field base_lr_pixel \
        --fractions 0.50 0.75 0.25 --subsample-seeds 42 101 202 303 404 --out results/seed_sweep_var.csv
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from experiment_config import (FINAL_DATA_PATH, COMBINED_TEST_FEATURES_PATH, XGB_TUNER_DIR,
                               TEST_LABELS_GEOJSON)
from subsample import get_fid_split_dl, get_fid_split_base_ml, subsample_train_fids

sys.stdout.reconfigure(line_buffering=True)


def load_gt():
    return {f["properties"]["fid"]: f["properties"]["crop_name"]
            for f in json.load(open(TEST_LABELS_GEOJSON))["features"]}


def score(field_fids, field_labels, gt):
    fl = pd.DataFrame({"pred": field_labels}, index=field_fids)
    fl["true"] = [gt.get(i) for i in fl.index]
    fl = fl.dropna()
    return (f1_score(fl["true"], fl["pred"], average="macro"),
            cohen_kappa_score(fl["true"], fl["pred"]))


def agg_field(df):
    """Mean features + mode label per fid (mirrors train_xgboost_field.aggregate_field)."""
    y = df.groupby("fid")["crop_name_encoded"].agg(lambda x: x.mode()[0])
    feat = df.drop(columns=["crop_name_encoded", "crop_name"], errors="ignore")
    X = feat.groupby("fid").mean(numeric_only=True).drop(
        columns=["crop_id", "SHAPE_AREA", "SHAPE_LEN"], errors="ignore")
    return X, y


# ----------------------- XGBoost (field) -----------------------
def run_xgboost(data, frac, sseed, le, best_params, test_df, gt):
    import xgboost as xgb
    train_fids, val_fids, test_fids = get_fid_split_dl(data)
    sub = subsample_train_fids(data, train_fids, frac, seed=sseed)
    Xtr, ytr = agg_field(data[data["fid"].isin(sub)].copy())
    Xv, yv = agg_field(data[data["fid"].isin(val_fids)].copy())
    Xtr = Xtr.dropna(axis=1, how="all"); Xv = Xv[Xtr.columns]
    imp = SimpleImputer(strategy="mean")
    Xtr_i = pd.DataFrame(imp.fit_transform(Xtr), columns=Xtr.columns)
    Xv_i = pd.DataFrame(imp.transform(Xv), columns=Xtr.columns)
    sc = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr_i), columns=Xtr.columns)
    Xv_s = pd.DataFrame(sc.transform(Xv_i), columns=Xtr.columns)
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(ytr)),
                              device="cuda", tree_method="hist", eval_metric="mlogloss",
                              early_stopping_rounds=50, random_state=42, **best_params)
    model.fit(Xtr_s, ytr, sample_weight=compute_sample_weight("balanced", ytr),
              eval_set=[(Xv_s, yv)], verbose=False)
    # OOS (mirrors predict_oos.predict_xgboost_field): COMBINED_TEST is already field-level
    X = test_df.drop(columns=["fid", "crop_name"], errors="ignore").select_dtypes(include=[np.number])
    for c in set(Xtr.columns) - set(X.columns):
        X[c] = 0.0
    X = X[list(Xtr.columns)]
    codes = model.predict(sc.transform(imp.transform(X)))
    labels = le.inverse_transform(codes)
    return score(test_df["fid"].values, labels, gt)


# ----------------------- Logistic Regression (pixel) -----------------------
def run_lr(data, frac, sseed, test_df, gt):
    # crop_name_encoded is added in main() for the XGBoost path; must exclude it here or the
    # encoded label leaks in as a feature (absent from the holdout -> zero-filled -> garbage).
    exclude = ["id", "point", "fid", "crop_id", "crop_name_encoded", "SHAPE_AREA", "SHAPE_LEN"]
    le = LabelEncoder(); le.fit(data["crop_name"])
    full_cols = data.select_dtypes(include=[np.number]).columns
    feat_cols = [c for c in full_cols if c not in exclude and not data[c].isna().any()]
    train_fids, val_fids, test_fids = get_fid_split_base_ml(data)
    sub = subsample_train_fids(data, train_fids, frac, seed=sseed)
    tr = data[data["fid"].isin(sub)]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[feat_cols]).astype(np.float32)
    ytr = le.transform(tr["crop_name"])
    model = LogisticRegression(max_iter=500, n_jobs=4, class_weight="balanced")
    model.fit(Xtr, ytr)
    # OOS (mirrors predict_oos.predict_base_ml): COMBINED_TEST is field-level
    df = test_df.drop(columns=["May"], errors="ignore").copy()
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = sc.transform(df[feat_cols].fillna(0)).astype(np.float32)
    labels = le.inverse_transform(model.predict(X))
    return score(df["fid"].values, labels, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgboost_field", "base_lr_pixel"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.50, 0.75, 0.25])
    ap.add_argument("--subsample-seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404])
    ap.add_argument("--out", default="results/seed_sweep_var.csv")
    args = ap.parse_args()

    print("Loading FINAL_DATA + holdout...")
    data = pd.read_parquet(FINAL_DATA_PATH, engine="pyarrow")
    data["crop_name_encoded"] = LabelEncoder().fit_transform(data["crop_name"])
    le_xgb = LabelEncoder(); le_xgb.fit(data["crop_name"])
    test_df = pd.read_parquet(COMBINED_TEST_FEATURES_PATH)
    gt = load_gt()
    best_params = joblib.load(os.path.join(XGB_TUNER_DIR, "best_xgb_params.joblib"))
    best_params["n_estimators"] = 2000

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write("model,fraction,subsample_seed,oos_f1_macro,oos_kappa,n_seeds,mean_epoch_s,epochs_run\n")

    for model_key in args.models:
        for frac in args.fractions:
            for sseed in args.subsample_seeds:
                t0 = time.time()
                if model_key == "xgboost_field":
                    f1m, kap = run_xgboost(data, frac, sseed, le_xgb, best_params, test_df, gt)
                elif model_key == "base_lr_pixel":
                    f1m, kap = run_lr(data, frac, sseed, test_df, gt)
                else:
                    print(f"[SKIP] unknown {model_key}"); continue
                dt = time.time() - t0
                print(f"{model_key} frac={frac} sseed={sseed} -> F1m={f1m:.4f} kappa={kap:.4f} ({dt:.0f}s)")
                with open(args.out, "a") as f:
                    f.write(f"{model_key},{frac},{sseed},{f1m:.4f},{kap:.4f},1,0,0\n")


if __name__ == "__main__":
    main()
