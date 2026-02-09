"""
CNN-BiLSTM ensemble inference on holdout test data (34S_20E_259N).

Uses saved PyTorch models (5-seed ensemble) to generate predictions.

Input: merged_dl_test_259N.parquet (pixel-level)
Output: predictions_cnn_bilstm.csv (field-level)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Import config for shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import MERGED_DL_TEST_PATH, MODEL_DIR

# Input
TEST_PARQUET = MERGED_DL_TEST_PATH

# Output
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "predictions_cnn_bilstm.csv")

# Class names (alphabetical order as LabelEncoder encodes them)
CLASS_NAMES = ["Barley", "Canola", "Lucerne/Medics", "Small grain grazing", "Wheat"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CropDataset(Dataset):
    """Dataset for pixel-level features with FID tracking."""

    def __init__(self, X, fids):
        self.X = X.astype(np.float32)
        self.fids = fids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), self.fids[idx]


class CropCNNBiLSTM(nn.Module):
    """CNN + BiLSTM model architecture (must match training)."""

    def __init__(self, input_size, num_classes, conv_filters=64, lstm_hidden=64, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(6, conv_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 6, -1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


def load_test_data():
    """Load and prepare test data."""
    print(f"Loading test data: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape: {df.shape}")
    print(f"Unique fields: {df['fid'].nunique()}")

    # Feature columns (same pattern as training)
    feature_cols = [
        col for col in df.columns
        if any(b in col for b in ["B2_", "B6_", "B11_", "B12_", "hue_", "EVI_"])
    ]
    # Drop columns that are entirely NaN
    feature_cols = [col for col in feature_cols if not df[col].isna().all()]
    # Fill remaining NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"Feature columns: {len(feature_cols)}")

    X = df[feature_cols].values
    fids = df["fid"].values

    return X, fids, len(feature_cols)


def load_models(n_features, n_classes):
    """Load all 5 ensemble models."""
    models = []
    for seed in range(5):
        model_path = os.path.join(MODEL_DIR, f"new_model_seed_{seed}_25epochs.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = CropCNNBiLSTM(n_features, n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded: {os.path.basename(model_path)}")

    return models


def run_inference(models, dataloader):
    """Run ensemble inference."""
    all_preds = []
    all_fids = []

    with torch.no_grad():
        for X_batch, fid_batch in dataloader:
            X_batch = X_batch.to(device)

            # Average logits across ensemble
            logits_list = [model(X_batch) for model in models]
            logits_ensemble = torch.stack(logits_list, dim=0)
            mean_logits = logits_ensemble.mean(dim=0)

            preds = torch.argmax(mean_logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_fids.extend(fid_batch.tolist())

    return all_preds, all_fids


def aggregate_to_field_level(preds, fids):
    """Aggregate pixel predictions to field level by majority vote."""
    df = pd.DataFrame({"fid": fids, "pred_label": preds})

    # Majority vote per field
    field_preds = df.groupby("fid")["pred_label"].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    )

    return field_preds


def main():
    print("=== CNN-BiLSTM Ensemble Inference ===")
    print(f"Device: {device}")

    # Load data
    X, fids, n_features = load_test_data()

    # Create dataset and dataloader
    dataset = CropDataset(X, fids)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Load models
    print("\nLoading models...")
    models = load_models(n_features, len(CLASS_NAMES))
    print(f"Loaded {len(models)} models")

    # Run inference
    print("\nRunning inference...")
    preds, fids_out = run_inference(models, dataloader)
    print(f"Total pixel predictions: {len(preds)}")

    # Aggregate to field level
    print("\nAggregating to field level...")
    field_preds = aggregate_to_field_level(preds, fids_out)
    print(f"Total fields: {len(field_preds)}")

    # Convert label indices to class names
    field_labels = [CLASS_NAMES[p] for p in field_preds.values]

    # Save predictions
    df_out = pd.DataFrame({
        "fid": field_preds.index,
        "crop_name": field_labels,
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Summary
    print("\n=== Prediction Distribution ===")
    print(df_out["crop_name"].value_counts())


if __name__ == "__main__":
    main()
