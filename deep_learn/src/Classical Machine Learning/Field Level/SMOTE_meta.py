# ===================== IMPORTS =====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import FINAL_DATA_PATH, MODEL_DIR, XGB_TUNER_DIR

import pandas as pd, numpy as np, gc, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
import xgboost as xgb

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(XGB_TUNER_DIR, exist_ok=True)

# ===================== LOAD & SPLIT =====================
print("Loading dataset...")
data = pd.read_parquet(FINAL_DATA_PATH, engine="pyarrow")
le = LabelEncoder()
data['crop_name_encoded'] = le.fit_transform(data['crop_name'])
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

# ===================== SPLIT BY FIELD FIRST (NO LEAKAGE) =====================
print("Splitting FIDs first...")
fids = data['fid'].unique()
train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)

train_data = data[data['fid'].isin(train_fids)].copy()
val_data = data[data['fid'].isin(val_fids)].copy()
test_data = data[data['fid'].isin(test_fids)].copy()
joblib.dump(test_fids, os.path.join(MODEL_DIR, "test_fids.joblib"))

# ===================== AGGREGATE PER FIELD =====================
print("Aggregating per field split...")
def aggregate_field(df):
    y = df.groupby('fid')['crop_name_encoded'].agg(lambda x: x.mode()[0])
    df = df.drop(columns=['crop_name_encoded', 'crop_name'], errors='ignore')  # Drop target columns before feature averaging
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

joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.joblib'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

# ===================== SMOTE-TOMEK =====================
print("Balancing classes with SMOTETomek...")
resampler = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)

# ===================== MODEL SETUP =====================
print("Training model...")

rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    device="cuda",
    tree_method="hist",
)
meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')

stacked = StackingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1
)

stacked.fit(X_train_resampled, y_train_resampled)

# ===================== EVALUATE =====================
print("Evaluating on test set...")
y_pred = stacked.predict(X_test_scaled)

print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# ===================== SAVE MODEL =====================
print("Saving model...")
joblib.dump(stacked, os.path.join(MODEL_DIR, "stacked_model_v1.joblib"), compress=3)

print("Saving FID, True Label, and Predicted Label to CSV...")

# Predict (already done above, but let's make sure)
y_pred = stacked.predict(X_test_scaled)

# Decode numeric labels to original crop names
true_labels = le.inverse_transform(y_test)
predicted_labels = le.inverse_transform(y_pred)

# Create DataFrame
results_df = pd.DataFrame({
    'fid': X_test.index,
    'true_label': true_labels,
    'predicted_label': predicted_labels
})

# Save to CSV
results_df.to_csv(os.path.join(XGB_TUNER_DIR, "fid_true_SMOTE_predicted_labels_stacked.csv"), index=False)

print("Saved as fid_true_predicted_labels_stacked.csv")

# ===================== REPORT =====================
from report import ModelReport

report = ModelReport("SMOTE Stacked Ensemble")
report.set_hyperparameters({
    "base_rf": {"n_estimators": 300, "max_depth": 20, "class_weight": "balanced"},
    "base_xgb": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    "meta": "LogisticRegression(max_iter=1000, class_weight=balanced)",
    "resampler": "SMOTETomek(random_state=42)",
})
report.set_split_info(train=len(X_train), val=len(X_val), test=len(X_test), seed=42)
report.set_metrics(y_test, y_pred, le.classes_)
report.set_predictions(X_test.index, y_test, y_pred, le.classes_)
report.generate()
