"""
Combine test parquets for out-of-sample inference.

Creates:
  - combined_test_features.parquet (field-level aggregated features for Classical ML)

Input: 6 per-band parquets from /mnt/bigdrive/.../features/testing_*.parquet
"""

import os
import sys
import pandas as pd

# Import config for shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import TEST_FEATURES_DIR, TEST_REGION, BANDS, COMBINED_TEST_FEATURES_PATH, DATA_OUTPUT_DIR

OUTPUT_PARQUET = COMBINED_TEST_FEATURES_PATH

META_COLS = ["id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name"]
MERGE_KEYS = ["id", "point", "fid"]
EXCLUDE_COLS = ["id", "point", "fid", "crop_id", "SHAPE_AREA", "SHAPE_LEN"]


def merge_test_features():
    """Merge all xr_fresh band features for the test region."""
    base_df = None

    for band in BANDS:
        path = os.path.join(TEST_FEATURES_DIR, f"testing_{band}_{TEST_REGION}.parquet")
        print(f"Reading {os.path.basename(path)}...")
        df = pd.read_parquet(path)
        print(f"  Shape: {df.shape}")

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


def aggregate_to_field_level(df):
    """Aggregate pixel-level data to field level (group by fid, mean)."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + ["crop_name"]]

    # Create aggregation mapping
    mapping = {c: "mean" for c in feature_cols}
    # Keep crop_name by mode (most common)
    if "crop_name" in df.columns:
        mapping["crop_name"] = lambda x: x.mode().iat[0] if not x.mode().empty else None

    field_df = df.groupby("fid").agg(mapping).reset_index()
    return field_df


def main():
    print("=== Combining Test Parquets ===")
    print(f"Test region: {TEST_REGION}")

    # Step 1: Merge all bands
    merged = merge_test_features()
    print(f"\nMerged shape: {merged.shape}")
    print(f"Unique fields: {merged['fid'].nunique()}")

    # Step 2: Aggregate to field level
    print("\nAggregating to field level...")
    field_df = aggregate_to_field_level(merged)
    print(f"Field-level shape: {field_df.shape}")

    # Step 3: Save
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    field_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved to {OUTPUT_PARQUET}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total fields: {len(field_df)}")
    if "crop_name" in field_df.columns:
        print(f"Crop distribution:\n{field_df['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()
