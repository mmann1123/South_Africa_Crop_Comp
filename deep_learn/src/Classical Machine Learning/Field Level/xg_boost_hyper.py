# ===================== IMPORTS =====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import FINAL_DATA_PATH, XGB_TUNER_DIR

import pandas as pd, numpy as np, gc, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import xgboost as xgb
import optuna

os.makedirs(XGB_TUNER_DIR, exist_ok=True)

# ===================== LOAD & SPLIT =====================
print("Loading dataset...")
data = pd.read_parquet(FINAL_DATA_PATH, engine="pyarrow")

# Label Encode
le = LabelEncoder()
data['crop_name_encoded'] = le.fit_transform(data['crop_name'])
joblib.dump(le, os.path.join(XGB_TUNER_DIR, 'label_encoder.joblib'))

# ===================== SPLIT BY FIELD =====================
print("Splitting FIDs first...")
fids = data['fid'].unique()
train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)

train_data = data[data['fid'].isin(train_fids)].copy()
val_data = data[data['fid'].isin(val_fids)].copy()
test_data = data[data['fid'].isin(test_fids)].copy()
joblib.dump(test_fids, os.path.join(XGB_TUNER_DIR, "test_fids.joblib"))

# ===================== AGGREGATE PER FIELD =====================
print("Aggregating per field split...")


def aggregate_field(df):
    y = df.groupby('fid')['crop_name_encoded'].agg(lambda x: x.mode()[0])
    df = df.drop(columns=['crop_name_encoded', 'crop_name'], errors='ignore')
    X = df.groupby('fid').mean(numeric_only=True).drop(columns=['crop_id', 'SHAPE_AREA', 'SHAPE_LEN'], errors='ignore')
    return X, y


X_train, y_train = aggregate_field(train_data)
X_val, y_val = aggregate_field(val_data)
X_test, y_test = aggregate_field(test_data)

# ===================== IMPUTE + SCALE =====================
print("Preprocessing features...")

X_train = X_train.dropna(axis=1, how='all')
X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

joblib.dump(imputer, os.path.join(XGB_TUNER_DIR, 'imputer.joblib'))
joblib.dump(scaler, os.path.join(XGB_TUNER_DIR, 'scaler.joblib'))

# ===================== HYPERPARAMETER TUNING FOR XGBOOST =====================
print("Starting Hyperparameter Tuning...")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        device="cuda",
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        **params
    )

    # 5-fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_train_scaled, y_train):
        X_tr, X_val_fold = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val_fold)
        score = f1_score(y_val_fold, preds, average='weighted')
        scores.append(score)
    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=75)

print("\nBest hyperparameters found:")
print(study.best_params)

joblib.dump(study.best_params, os.path.join(XGB_TUNER_DIR, 'best_xgb_params.joblib'))

# ===================== TRAIN FINAL MODEL =====================
print("Training final XGBoost model...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train)),
    device="cuda",
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    **study.best_params
)

final_model.fit(X_train_scaled, y_train)

# ===================== EVALUATE =====================
print("Evaluating final model on test set...")
y_pred = final_model.predict(X_test_scaled)

print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# ===================== SAVE MODEL =====================
print("Saving final model...")
joblib.dump(final_model, os.path.join(XGB_TUNER_DIR, "final_xgb_model.joblib"), compress=3)

# ===================== REPORT =====================
from report import ModelReport

report = ModelReport("XGBoost Field-Level")
report.set_hyperparameters(study.best_params)
report.set_split_info(train=len(X_train), val=len(X_val), test=len(X_test), seed=42)
report.set_metrics(y_test, y_pred, le.classes_)
report.set_predictions(X_test.index, y_test, y_pred, le.classes_)
report.set_feature_importance(final_model.feature_importances_, X_train.columns)
report.add_notes(f"Optuna tuning: {study.best_trial.number + 1} trials, best F1={study.best_value:.4f}")
report.generate()
