"""
Ensemble Voting and Tacking classifier
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import FINAL_DATA_PATH, MODEL_DIR

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.multiclass      import OneVsRestClassifier
from sklearn.metrics         import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

os.makedirs(MODEL_DIR, exist_ok=True)

# Set paths
PARQUET_PATH = FINAL_DATA_PATH
EXCLUDE_COLS = ['id','point','fid','crop_id','SHAPE_AREA','SHAPE_LEN']
TEST_SIZE    = 0.20
SEED         = 42
raw = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
print("Raw rows:", raw.shape)
feature_cols = [c for c in raw.columns if c not in EXCLUDE_COLS + ['crop_name']]

mapping = {c: "mean" for c in feature_cols}
mapping["crop_name"] = lambda x: x.mode().iat[0] if not x.mode().empty else None

field_df = raw.groupby("fid").agg(mapping).reset_index()
print("Aggregated (fields):", field_df.shape)

train_fids, test_fids = train_test_split(
    field_df.fid.unique(),
    test_size=TEST_SIZE,
    random_state=SEED,
)
train_df = field_df[field_df.fid.isin(train_fids)].reset_index(drop=True)
test_df  = field_df[field_df.fid.isin(test_fids )].reset_index(drop=True)
print(f"Train fields: {train_df.shape[0]} | Test fields: {test_df.shape[0]}")
X_train_raw, X_test_raw = train_df[feature_cols], test_df[feature_cols]

train_df.to_parquet(os.path.join(MODEL_DIR, "train_data.parquet"), index=False)
test_df.to_parquet(os.path.join(MODEL_DIR, "test_data.parquet"), index=False)
print("Saved train_data.parquet and test_data.parquet")

le = LabelEncoder()
y_train = le.fit_transform(train_df.crop_name)
y_test  = le.transform(test_df.crop_name)

num_pipe   = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])
preproc = ColumnTransformer([("num", num_pipe, feature_cols)])

# Define One‑vs‑Rest base learners
base_lr  = OneVsRestClassifier(LogisticRegression(max_iter=1000))
base_rf  = OneVsRestClassifier(RandomForestClassifier(
    n_estimators=400, n_jobs=-1, random_state=SEED
))
base_hgb = OneVsRestClassifier(HistGradientBoostingClassifier(
    random_state=SEED
))
estimators = [('lr', base_lr), ('rf', base_rf), ('hgb', base_hgb)]

# Added try catch for XGBoost due to load issues. Can omit
try:
    from xgboost import XGBClassifier
    base_xgb = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED
        )
    )
    estimators.append(('xgb', base_xgb))
    print("xgboost: added.")
except ImportError:
    print("xgboost: not installed, skipping.")

voting_pipe = Pipeline([
    ("prep", preproc),
    ("vote", VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1))
])
stack_pipe = Pipeline([
    ("prep", preproc),
    ("stack", StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True,
        n_jobs=-1
    ))
])

def eval_plot(name, pipe):
    pipe.fit(X_train_raw, y_train)
    preds = pipe.predict(X_test_raw)
    acc   = accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f} | Cohen κ: {kappa:.4f}")

    cm    = confusion_matrix(y_test, preds)
    pct   = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True) * 100)
    labels = np.array([f"{v:.1f}%" for v in pct.flatten()]).reshape(pct.shape)

    plt.figure(figsize=(9,7))
    sns.heatmap(pct, annot=labels, fmt='', cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix (%)")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.close()
    return preds

preds_v = eval_plot("Voting",   voting_pipe)
preds_s = eval_plot("Stacking", stack_pipe)

# Save required files for standalone inference
joblib.dump(voting_pipe, os.path.join(MODEL_DIR, "ensemble_voting.pkl"))
joblib.dump(stack_pipe,  os.path.join(MODEL_DIR, "ensemble_stacking.pkl"))
joblib.dump(le,          os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("Saved: ensemble_voting.pkl, ensemble_stacking.pkl, label_encoder.pkl")

# ===================== REPORT =====================
from report import ModelReport

for name, preds in [("Voting Ensemble", preds_v), ("Stacking Ensemble", preds_s)]:
    report = ModelReport(name)
    report.set_hyperparameters({
        "estimators": ["LR (OVR)", "RF (OVR, n=400)", "HGB (OVR)", "XGB (OVR, n=400, depth=6, lr=0.1)"],
        "voting": "soft" if "Voting" in name else "N/A",
        "final_estimator": "LogisticRegression(max_iter=1000)" if "Stacking" in name else "N/A",
    })
    report.set_split_info(train=len(train_df), test=len(test_df), seed=SEED)
    report.set_metrics(y_test, preds, le.classes_)
    report.set_predictions(test_df["fid"].values, y_test, preds, le.classes_)
    report.generate()

### ---- End of Train script rest is test inference ---###



import numpy as np
import pandas as pd
from joblib import load

TEST_PARQUET       = FINAL_DATA_PATH
VOTE_PIPE_FILE     = os.path.join(MODEL_DIR, "ensemble_voting.pkl")
STACK_PIPE_FILE    = os.path.join(MODEL_DIR, "ensemble_stacking.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
OUT_VOTING         = os.path.join(MODEL_DIR, "results_ensemble_field_voting.csv")
OUT_STACK          = os.path.join(MODEL_DIR, "results_ensemble_field_stacking.csv")


def main():
    df = pd.read_parquet(TEST_PARQUET, engine="pyarrow")
    df_num = df.select_dtypes(include=[np.number])
    if 'fid' not in df_num.columns:
        raise ValueError("'fid' must be numeric")
    df_feat = df_num.groupby('fid', as_index=False).mean()

    fids = df_feat['fid'].to_numpy()
    X    = df_feat.drop(columns=['fid'])

    vote_pipe  = load(VOTE_PIPE_FILE)
    stack_pipe = load(STACK_PIPE_FILE)
    le          = load(LABEL_ENCODER_FILE)

    codes_v   = vote_pipe.predict(X)
    codes_s   = stack_pipe.predict(X)
    labels_v  = le.inverse_transform(codes_v)
    labels_s  = le.inverse_transform(codes_s)

    pd.DataFrame({"fid":fids, "predicted":labels_v}).to_csv(OUT_VOTING, index=False)
    print(f"Saved → {OUT_VOTING}")
    pd.DataFrame({"fid":fids, "predicted":labels_s}).to_csv(OUT_STACK, index=False)
    print(f"Saved → {OUT_STACK}")

if __name__=="__main__":
    main()
