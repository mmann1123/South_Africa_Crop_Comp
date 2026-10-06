"""Try alternative pixel->field aggregation rules for the existing L-TAE pixel
checkpoints and test which (if any) reproduces the published confusion matrix
(out_of_sample/scoring_results/confusion_matrix_L_TAE_(pixel).csv).
Read-only on models and out_of_sample; writes to experiments/paired_bootstrap/rerun/."""
import os, sys
from collections import Counter
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
OOS = os.path.join(HERE, "..", "..", "out_of_sample")
sys.path.insert(0, OOS)
import inference_ltae as mod
from joblib import load
from compare_predictions import load_ground_truth

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fc = load(os.path.join(mod.MODEL_DIR, "ltae_feature_columns.joblib"))
sc = load(os.path.join(mod.MODEL_DIR, "ltae_scaler.joblib"))
le = load(os.path.join(mod.MODEL_DIR, "ltae_label_encoder.joblib"))
df = pd.read_parquet(mod.TEST_PARQUET)
for c in fc:
    if c not in df.columns: df[c] = 0
df[fc] = df[fc].fillna(0)
X = np.nan_to_num(sc.transform(df[fc].values).astype(np.float32))
fids = df["fid"].values
dl = DataLoader(mod.TemporalDataset(X, fids), batch_size=2048, shuffle=False)
L = []
for s in mod.SEEDS:
    m = mod.LTAE(in_channels=mod.N_BANDS, num_classes=len(le.classes_)).to(dev)
    m.load_state_dict(torch.load(os.path.join(mod.MODEL_DIR, f"ltae_seed_{s}.pt"), map_location=dev)); m.eval()
    out = []
    with torch.no_grad():
        for xb, _ in dl: out.append(m(xb.to(dev)).cpu())
    L.append(torch.cat(out).float().numpy())
L = np.stack(L)                                   # (seeds, pixels, K)
P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)

def vote(pred):
    return pd.Series(pred).groupby(fids).agg(lambda x: Counter(x).most_common(1)[0][0])
def pool(arr):
    return pd.DataFrame(arr).groupby(fids).mean().values.argmax(1), np.unique(fids)

variants = {}
variants["avg_logits->pixel argmax->field vote (current)"] = vote(L.mean(0).argmax(1))
variants["avg_softmax->pixel argmax->field vote"] = vote(P.mean(0).argmax(1))
a, u = pool(L.mean(0)); variants["avg_logits->field mean pool"] = pd.Series(a, index=u)
a, u = pool(P.mean(0)); variants["avg_softmax->field mean pool"] = pd.Series(a, index=u)
seed_field = np.stack([vote(L[i].argmax(1)).sort_index().values for i in range(len(mod.SEEDS))])
idx = vote(L[0].argmax(1)).sort_index().index
variants["per-seed field vote->seed vote"] = pd.Series([Counter(c).most_common(1)[0][0] for c in seed_field.T], index=idx)

gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
canon = pd.read_csv(os.path.join(OOS, "scoring_results", "confusion_matrix_L_TAE_(pixel).csv"), index_col=0)
from sklearn.metrics import f1_score
os.makedirs(os.path.join(HERE, "rerun"), exist_ok=True)
for name, s in variants.items():
    lab = pd.Series(le.inverse_transform(s.values.astype(int)), index=s.index, name="crop_name")
    m = pd.concat([gt, lab], axis=1, join="inner")
    cm = pd.crosstab(m.iloc[:, 0], m.iloc[:, 1]).reindex(index=canon.index, columns=canon.columns, fill_value=0)
    match = bool((cm.values == canon.values).all())
    print(f"{name:52s} F1m={f1_score(m.iloc[:,0], m.iloc[:,1], average='macro'):.4f}  matches_published={match}")
    if match:
        lab.rename_axis("fid").reset_index().to_csv(os.path.join(HERE, "rerun", "predictions_ltae_published_match.csv"), index=False)
