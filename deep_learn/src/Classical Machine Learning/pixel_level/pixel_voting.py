# ===================== IMPORTS =====================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import FINAL_DATA_PATH, MODEL_DIR

import pandas as pd, numpy as np, gc, joblib, psutil
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_memory(stage=""):
    mem = psutil.virtual_memory().used / 1024 ** 3
    print(f"{stage} RAM Usage: {mem:.2f} GB")

# ===================== LOAD DATA =====================
print("Loading dataset...")
data = pd.read_parquet(FINAL_DATA_PATH)
le = LabelEncoder()
data['crop_name_encoded'] = le.fit_transform(data['crop_name'])
joblib.dump(data['crop_name_encoded'].unique(), os.path.join(MODEL_DIR, 'label_encoder.joblib'))
print_memory("After data load")

# ===================== FIELD-WISE SPLIT =====================
print("Splitting by FIDs...")
train_fids, test_fids = train_test_split(data['fid'].unique(), test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)

train_data = data[data['fid'].isin(train_fids)]
val_data = data[data['fid'].isin(val_fids)]
test_data = data[data['fid'].isin(test_fids)]

del data
gc.collect()
print_memory("After field split")

# ===================== FEATURE PROCESSING =====================
def prepare_features(df):
    y = df['crop_name_encoded'].astype(np.int32).values
    X = df.drop(columns=['crop_name_encoded', 'crop_name', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN', 'fid'], errors='ignore')
    return X, y

X_train, y_train = prepare_features(train_data)
X_val, y_val = prepare_features(val_data)
X_test, y_test = prepare_features(test_data)

del train_data, val_data  # keep only test_data for final field-level eval
gc.collect()

X_train = X_train.dropna(axis=1, how='all')
X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

# Downcast all feature columns to float32
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

# Imputation + Scaling
imputer = SimpleImputer(strategy='mean')
X_train_np = imputer.fit_transform(X_train)
X_val_np = imputer.transform(X_val)
X_test_np = imputer.transform(X_test)

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np).astype(np.float32)
X_val_np = scaler.transform(X_val_np).astype(np.float32)
X_test_np = scaler.transform(X_test_np).astype(np.float32)

joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.joblib'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

del X_train, X_val, X_test
gc.collect()
print_memory("After preprocessing")

# ===================== TRAIN MODELS ONE-BY-ONE =====================
print("Training models one-by-one...")

# ---- RF
rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=2)
rf.fit(X_train_np, y_train)
rf_probs = rf.predict_proba(X_test_np)
print("RF done.")
# Save models
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"), compress=3)

del rf; gc.collect(); print_memory("Post RF")

# ---- XGB
xgb = XGBClassifier(n_estimators=30, max_depth=6, learning_rate=0.1,
                    tree_method="hist", eval_metric='mlogloss', use_label_encoder=False,
                    random_state=42, n_jobs=2)
xgb.fit(X_train_np, y_train)
xgb_probs = xgb.predict_proba(X_test_np)
print("XGB done.")
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.joblib"), compress=3)

del xgb; gc.collect(); print_memory("Post XGB")

# ---- LGBM
lgbm = LGBMClassifier(n_estimators=30, max_depth=6, learning_rate=0.1,
                      random_state=42, n_jobs=2)
lgbm.fit(X_train_np, y_train)
lgbm_probs = lgbm.predict_proba(X_test_np)
print("LGBM done.")
joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm_model.joblib"), compress=3)
del lgbm; gc.collect(); print_memory("Post LGBM")

# ===================== SOFT VOTING + EVALUATION =====================
print("Soft voting...")
final_probs = (rf_probs + xgb_probs + lgbm_probs) / 3
y_pred = np.argmax(final_probs, axis=1)

print("\n=== Pixel-Level Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Cohen’s Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")

# ===================== IMPROVED FIELD-LEVEL SOFT VOTING =====================
print("\n=== Hybrid Field-Level Voting + Export + Confusion Matrix ===")

# Prepare probabilities
probs_df = pd.DataFrame(final_probs, columns=[f'class_{i}' for i in range(final_probs.shape[1])])
probs_df['fid'] = test_data['fid'].values
probs_df['true'] = test_data['crop_name_encoded'].values
probs_df['pred'] = y_pred

class_cols = [col for col in probs_df.columns if col.startswith("class_")]
field_probs = probs_df.groupby('fid')[class_cols].mean()

# SOFT voting
fid_preds_soft = field_probs.idxmax(axis=1).apply(lambda x: int(x.replace("class_", "")))
field_confidence = field_probs.max(axis=1)
field_top2gap = field_probs.apply(lambda row: np.sort(row.values)[-1] - np.sort(row.values)[-2], axis=1)

# MODE voting
test_data['pred'] = y_pred
fid_preds_mode = test_data.groupby('fid')['pred'].agg(lambda x: x.mode()[0])

# TRUE labels
fid_truth = test_data.groupby('fid')['crop_name_encoded'].agg(lambda x: x.mode()[0])

# HYBRID voting: fallback to mode if top2 class gap is too low
confidence_threshold = 0.10
fid_preds_hybrid = fid_preds_soft.copy()
for fid in fid_preds_hybrid.index:
    if field_top2gap[fid] < confidence_threshold:
        fid_preds_hybrid[fid] = fid_preds_mode[fid]

# EVALUATE hybrid
hybrid_acc = accuracy_score(fid_truth, fid_preds_hybrid)
hybrid_kappa = cohen_kappa_score(fid_truth, fid_preds_hybrid)
print(f"Hybrid Field Accuracy: {hybrid_acc:.4f}")
print(f"Hybrid Field Kappa:    {hybrid_kappa:.4f}")

# CSV EXPORT
field_pred_df = pd.DataFrame({
    'fid': fid_truth.index,
    'true_label': fid_truth.values,
    'soft_vote_pred': fid_preds_soft.values,
    'mode_pred': fid_preds_mode.values,
    'hybrid_pred': fid_preds_hybrid.values,
    'soft_vote_confidence': field_confidence.values
})
field_pred_df.to_csv(os.path.join(MODEL_DIR, "pixel_predictions.csv"), index=False)
print("Saved field predictions to field_predictions.csv")

# CONFUSION MATRIX (field-level, hybrid)
label_names = le.inverse_transform(sorted(fid_truth.unique()))
cm = confusion_matrix(fid_truth, fid_preds_hybrid, labels=sorted(fid_truth.unique()))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names)
plt.xlabel("Predicted Crop")
plt.ylabel("True Crop")
plt.title("Hybrid Voting - Field-Level Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "field_confusion_matrix.png"))
plt.show()
