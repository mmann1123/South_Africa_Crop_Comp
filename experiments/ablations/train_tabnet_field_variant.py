"""Train a TabNet field-level (xr_fresh) 5-seed ensemble for the loss/sampler
ablation (reviewer comment 3.4).

Lean variant of deep_learn/src/Classical Machine Learning/Field Level/tabnet_field.py:
same data prep, fid split, aggregation, imputation and scaling, but with toggleable
imbalance handling and no ModelReport. Writes the same artifact filenames as
tabnet_field.py (tabnet_field_seed_*.zip + tabnet_field_{imputer,scaler,
label_encoder,feature_columns}.joblib) to --output-dir, so
predict_oos.predict_tabnet_field can score it unchanged.

  --loss focal       WeightedFocalLoss(gamma=2, alpha=1/freq)   [baseline]
  --loss weighted_ce class-weighted CE (gamma=0)
  --loss plain_ce    unweighted CE
  --sampler off|on   on => pytorch-tabnet fit(weights=1) inverse-frequency oversampling

Run with deep_field python.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from pytorch_tabnet import TabNetClassifier
from pytorch_tabnet.metrics import Metric

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "deep_learn", "src"))
from config import FINAL_DATA_PATH  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

SEEDS = [42, 101, 202, 303, 404]


# NB: pytorch-tabnet 4.6.0 calls loss_fn(y_pred, y_true, weight=...). Our loss
# callables accept and ignore that kwarg so the controlled imbalance setting is
# exactly --loss (our own alpha), independent of tabnet's internal class weighting.
# The --sampler lever is realized separately via fit(weights=...).
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, weight=None):
        alpha = self.alpha.to(input.device)
        ce = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce)
        loss = alpha[target] * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class WeightedCE(nn.Module):
    """Class-weighted cross-entropy (gamma=0): focal without the focusing term."""
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target, weight=None):
        return F.cross_entropy(input, target, weight=self.alpha.to(input.device))


class PlainCE(nn.Module):
    """Unweighted cross-entropy: no imbalance handling."""
    def forward(self, input, target, weight=None):
        return F.cross_entropy(input, target)


class F1MacroMetric(Metric):
    def __init__(self):
        self._name = "f1_macro"
        self._maximize = True

    def __call__(self, y_true, y_score, *args, **kwargs):
        if hasattr(y_score, "detach"):
            y_score = y_score.detach().cpu().numpy()
        if hasattr(y_true, "detach"):
            y_true = y_true.detach().cpu().numpy()
        return f1_score(np.asarray(y_true), np.argmax(np.asarray(y_score), axis=1), average='macro')


def aggregate_field(df):
    y = df.groupby('fid')['crop_name_encoded'].agg(lambda x: x.mode()[0])
    feats = df.drop(columns=['crop_name_encoded', 'crop_name'], errors='ignore')
    X = feats.groupby('fid').mean(numeric_only=True).drop(
        columns=['crop_id', 'SHAPE_AREA', 'SHAPE_LEN'], errors='ignore')
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', choices=['focal', 'weighted_ce', 'plain_ce'], default='focal')
    parser.add_argument('--sampler', choices=['off', 'on'], default='off')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== TabNet Field variant === loss={args.loss}, sampler={args.sampler}, out={args.output_dir}")

    data = pd.read_parquet(FINAL_DATA_PATH, engine="pyarrow")
    le = LabelEncoder()
    data['crop_name_encoded'] = le.fit_transform(data['crop_name'])

    fids = data['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_fids, val_fids = train_test_split(train_fids, test_size=0.2, random_state=42)
    train_data = data[data['fid'].isin(train_fids)].copy()
    val_data = data[data['fid'].isin(val_fids)].copy()
    test_data = data[data['fid'].isin(test_fids)].copy()

    X_train, y_train = aggregate_field(train_data)
    X_val, y_val = aggregate_field(val_data)
    X_test, y_test = aggregate_field(test_data)

    X_train = X_train.dropna(axis=1, how='all')
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]
    feature_columns = list(X_train.columns)

    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train_imp).astype(np.float32))
    X_val_scaled = np.nan_to_num(scaler.transform(X_val_imp).astype(np.float32))

    y_train_arr = y_train.values
    y_val_arr = y_val.values

    counts = np.maximum(np.bincount(y_train_arr, minlength=len(le.classes_)).astype(np.float64), 1.0)
    alpha = 1.0 / counts
    alpha = alpha / alpha.sum() * len(le.classes_)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(dev)

    if args.loss == 'focal':
        loss_fn = WeightedFocalLoss(alpha_tensor, gamma=2.0)
    elif args.loss == 'weighted_ce':
        loss_fn = WeightedCE(alpha_tensor)
    else:
        loss_fn = PlainCE()

    # Weighted-sampling lever: pytorch-tabnet 4.6.0 delivers fit(weights=) only via
    # the loss-fn weight kwarg (which our custom losses ignore), so it is a no-op for
    # our controlled loss settings. We therefore realize resampling explicitly as a
    # class-balanced oversampling of the training set (the one-shot analogue of a
    # WeightedRandomSampler with inverse-frequency weights), keeping alpha fixed to the
    # original class distribution so the only thing that changes is the sampling.
    if args.sampler == 'on':
        rng = np.random.RandomState(42)
        max_n = int(counts.max())
        idx = np.concatenate([
            rng.choice(np.where(y_train_arr == c)[0], size=max_n, replace=True)
            for c in range(len(le.classes_)) if (y_train_arr == c).any()
        ])
        rng.shuffle(idx)
        X_train_scaled = X_train_scaled[idx]
        y_train_arr = y_train_arr[idx]
        print(f"  resampled (balanced oversample) train size: {len(y_train_arr)}")

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-3),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, seed=seed, verbose=0,
        )
        model.fit(
            X_train=X_train_scaled, y_train=y_train_arr,
            eval_set=[(X_val_scaled, y_val_arr)], eval_metric=[F1MacroMetric],
            loss_fn=loss_fn, weights=0,
            max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128,
            num_workers=0, drop_last=False,
        )
        model.save_model(os.path.join(args.output_dir, f"tabnet_field_seed_{seed}"))

    joblib.dump(le, os.path.join(args.output_dir, "tabnet_field_label_encoder.joblib"))
    joblib.dump(imputer, os.path.join(args.output_dir, "tabnet_field_imputer.joblib"))
    joblib.dump(scaler, os.path.join(args.output_dir, "tabnet_field_scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(args.output_dir, "tabnet_field_feature_columns.joblib"))
    print(f"Saved 5-seed TabNet field ensemble + artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
