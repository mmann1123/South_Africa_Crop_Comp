"""
Merge pre-computed xr_fresh feature parquets into a single parquet for classical ML models.

Creates:
  - Data/final_data.parquet (training data only: regions 258N + 259N)

Merges on ['id', 'point', 'fid'] keys (not positional) because xr_fresh parquets
may have different row counts from raw parquets due to dropped rows during extraction.
"""

import os
import pandas as pd
from config import (
    FEATURES_DIR, FINAL_DATA_PATH,
    BANDS, TRAIN_REGIONS,
)

META_COLS = ["id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name"]
MERGE_KEYS = ["id", "point", "fid"]


def merge_region_features(region, bands):
    """Merge all xr_fresh band features for a single region."""
    base_df = None

    for band in bands:
        path = os.path.join(FEATURES_DIR, f"{band}_{region}.parquet")
        print(f"  Reading {os.path.basename(path)}...")
        df = pd.read_parquet(path)

        if base_df is None:
            base_df = df
        else:
            # Only take feature columns not already in base_df (avoids _x/_y suffixes)
            feature_cols = [c for c in df.columns if c not in META_COLS and c not in base_df.columns]
            if feature_cols:
                base_df = base_df.merge(
                    df[MERGE_KEYS + feature_cols],
                    on=MERGE_KEYS,
                    how="inner",
                )

    return base_df


def main():
    print("=== Merging xr_fresh features ===")
    region_dfs = []

    for region in TRAIN_REGIONS:
        print(f"\nRegion: {region}")
        df = merge_region_features(region, BANDS)
        region_dfs.append(df)
        print(f"  Shape: {df.shape}")

    merged = pd.concat(region_dfs, ignore_index=True)
    print(f"\nMerged shape: {merged.shape}")
    print(f"Crop classes: {merged['crop_name'].unique()}")

    feature_cols = [c for c in merged.columns if c not in META_COLS]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:6]}...")

    os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
    merged.to_parquet(FINAL_DATA_PATH, index=False)
    print(f"Saved to {FINAL_DATA_PATH}")


if __name__ == "__main__":
    main()
