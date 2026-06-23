"""
Controlled comparison: does the xr_fresh representation help the trees, or is its
value purely operational (speed / compactness)?

Addresses reviewer point 4.2 -- the manuscript previously conceded that "our
experiments do not isolate the marginal value of the xr_fresh representation
against raw reflectance for the trees." This script isolates it.

ONE learner (XGBoost, the paper's pixel-level configuration), held fixed, is
trained twice -- once on raw monthly reflectance and once on xr_fresh time-series
features -- with identical FID-wise splits, identical preprocessing, and identical
field-level majority-vote aggregation. Only the input representation changes.

Both representations are scored on:
  - IN-REGION   : held-out FIDs from the two training tiles (34S_19E_258N/259N)
  - SPATIAL-XFER: the disjoint holdout tile 34S_20E_259N

Inputs
  raw      train  data/merged_dl_train.parquet        (pixel, monthly band values)
  raw      ST     data/merged_dl_test.parquet         (pixel, holdout tile)
  xr_fresh train  data/final_data.parquet             (pixel, xr_fresh stats)
  xr_fresh ST     <FEAT>/X_testing_{band}_34S_20E_259N.parquet  (pixel, holdout tile)
  ST truth        TEST_LABELS_GEOJSON (labels.geojson, merged by fid)

Output
  experiments/results/xrfresh_vs_raw_xgb.csv   (one row per representation x split)
  prints a formatted summary table

Run with the ml_field env (GPU XGBoost):
  /home/mmann1123/miniconda3/envs/ml_field/bin/python3 experiments/xrfresh_vs_raw_xgb.py
"""
import sys, os, gc, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import (
    FINAL_DATA_PATH, MERGED_DL_PATH, MERGED_DL_TEST_PATH,
    TEST_LABELS_DIR, TEST_REGION, REPO_ROOT,
)

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

FEAT = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/features"
BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]
KEYS = ["id", "point", "fid"]
META = {"id", "point", "fid", "crop_id", "crop_name", "SHAPE_AREA", "SHAPE_LEN"}
SEED = 42
TEST_LABELS_GEOJSON = os.path.join(
    TEST_LABELS_DIR,
    f"ref_fusion_competition_south_africa_test_labels_{TEST_REGION}",
    "labels.geojson",
)
OUT_CSV = os.path.join(SCRIPT_DIR, "results", "xrfresh_vs_raw_xgb.csv")


def new_xgb(n_classes):
    """The manuscript's pixel-level XGBoost configuration."""
    return XGBClassifier(
        n_estimators=1000, eval_metric="mlogloss", device="cuda",
        tree_method="hist", early_stopping_rounds=50, random_state=SEED,
    )


def field_aggregate(fids, proba, classes):
    """Aggregate pixel probabilities to field level (mean pooling -> argmax)."""
    df = pd.DataFrame(proba, columns=range(len(classes)))
    df["fid"] = fids
    fmean = df.groupby("fid").mean()
    fid_order = fmean.index.values
    fproba = fmean.values
    fpred = classes[fproba.argmax(axis=1)]
    return fid_order, fpred, fproba


def metrics(y_true_lab, y_pred_lab, fproba, classes, tag):
    labels = list(classes)
    xent = log_loss(y_true_lab, np.clip(fproba, 1e-7, 1 - 1e-7), labels=labels)
    f1m = f1_score(y_true_lab, y_pred_lab, average="macro")
    wf1 = f1_score(y_true_lab, y_pred_lab, average="weighted")
    kap = cohen_kappa_score(y_true_lab, y_pred_lab)
    acc = accuracy_score(y_true_lab, y_pred_lab)
    print(f"  [{tag}] F1m={f1m:.4f} wF1={wf1:.4f} kappa={kap:.4f} "
          f"acc={acc:.4f} Xent={xent:.4f} n={len(y_true_lab)}", flush=True)
    return dict(f1m=f1m, wf1=wf1, kappa=kap, acc=acc, xent=xent, n=len(y_true_lab))


def load_raw():
    """Raw monthly reflectance. Drop *_May to match the project's month regime
    (months 05/06 excluded; June is already absent), keeping raw and xr_fresh on
    the same underlying months."""
    tr = pd.read_parquet(MERGED_DL_PATH)
    feat = sorted(c for c in tr.columns if c not in META and not c.endswith("_May"))
    st = pd.read_parquet(MERGED_DL_TEST_PATH)
    return tr, st, feat


def load_xrfresh():
    """xr_fresh per-pixel time-series statistics. ST assembled from the aligned
    X_testing_* band parquets (all 3,409,489 rows)."""
    tr = pd.read_parquet(FINAL_DATA_PATH)
    # feature cols = numeric, non-meta, no all-NaN (matches base_ml_models.py)
    num = tr.select_dtypes(include=[np.number]).columns
    feat = sorted(c for c in num if c not in META and not tr[c].isna().any())
    parts = []
    for i, b in enumerate(BANDS):
        d = pd.read_parquet(f"{FEAT}/X_testing_{b}_34S_20E_259N.parquet")
        cols = [c for c in d.columns if c in feat]
        parts.append(d.set_index(KEYS)[cols])
        del d; gc.collect()
    st = pd.concat(parts, axis=1).reset_index()
    del parts; gc.collect()
    feat = [c for c in feat if c in st.columns]  # only cols present in ST too
    return tr, st, feat


def run(name, loader):
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    train_all, st, feat = loader()
    print(f"  train pixels={len(train_all):,}  features={len(feat)}  "
          f"ST pixels={len(st):,}", flush=True)

    le = LabelEncoder().fit(train_all["crop_name"])
    classes = le.classes_

    # FID-wise split (matches base_ml_models.py: 0.2 test, then 0.15 val)
    fids = train_all["fid"].unique()
    tr_fids, te_fids = train_test_split(fids, test_size=0.2, random_state=SEED)
    tr_fids, va_fids = train_test_split(tr_fids, test_size=0.15, random_state=SEED)
    tr_df = train_all[train_all["fid"].isin(tr_fids)]
    va_df = train_all[train_all["fid"].isin(va_fids)]
    te_df = train_all[train_all["fid"].isin(te_fids)].copy()

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(tr_df[feat])).astype(np.float32)
    ytr = le.transform(tr_df["crop_name"])
    Xva = sc.transform(imp.transform(va_df[feat])).astype(np.float32)
    yva = le.transform(va_df["crop_name"])
    sw = compute_sample_weight("balanced", ytr)
    del train_all, tr_df, va_df; gc.collect()

    model = new_xgb(len(classes))
    model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    print(f"  trained (best_iter={model.best_iteration}) in {time.time()-t0:.0f}s",
          flush=True)
    del Xtr, Xva; gc.collect()

    rows = []

    # ---- in-region ----
    Xte = sc.transform(imp.transform(te_df[feat])).astype(np.float32)
    fid_i, pred_i, prob_i = field_aggregate(
        te_df["fid"].values, model.predict_proba(Xte), classes)
    truth_i = (te_df.groupby("fid")["crop_name"]
               .agg(lambda x: x.mode()[0]).reindex(fid_i).values)
    m = metrics(truth_i, pred_i, prob_i, classes, "IN-REGION")
    rows.append(dict(representation=name, split="in_region", **m))
    del Xte, te_df; gc.collect()

    # ---- spatial transfer ----
    Xst = sc.transform(imp.transform(st[feat])).astype(np.float32)
    fid_s, pred_s, prob_s = field_aggregate(
        st["fid"].values, model.predict_proba(Xst), classes)
    gt = gpd.read_file(TEST_LABELS_GEOJSON)[["fid", "crop_name"]].copy()
    gt["fid"] = gt["fid"].astype(int)
    pred_df = pd.DataFrame({"fid": fid_s.astype(int), "_pi": np.arange(len(fid_s))})
    mrg = gt.merge(pred_df, on="fid", how="inner")
    idx = mrg["_pi"].values
    m = metrics(mrg["crop_name"].values, pred_s[idx], prob_s[idx], classes,
                "SPATIAL-XFER")
    rows.append(dict(representation=name, split="spatial_transfer", **m))
    del Xst, st; gc.collect()

    return rows


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_rows = []
    all_rows += run("raw_reflectance", load_raw)
    all_rows += run("xr_fresh", load_xrfresh)

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 64)
    print("CONTROLLED COMPARISON  (XGBoost, field-level majority pooling)")
    print("=" * 64)
    print(f"{'representation':<16}{'split':<18}{'F1m':>7}{'wF1':>7}"
          f"{'kappa':>7}{'Xent':>8}")
    for _, r in res.iterrows():
        print(f"{r.representation:<16}{r.split:<18}{r.f1m:>7.3f}{r.wf1:>7.3f}"
              f"{r.kappa:>7.3f}{r.xent:>8.3f}")

    piv = res.pivot(index="split", columns="representation", values="f1m")
    print("\nF1m delta (xr_fresh - raw):")
    for split in piv.index:
        d = piv.loc[split, "xr_fresh"] - piv.loc[split, "raw_reflectance"]
        print(f"  {split:<18}{d:+.3f}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
