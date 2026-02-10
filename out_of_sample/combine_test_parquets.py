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
EXCLUDE_COLS = ["id", "point", "fid", "crop_id", "SHAPE_AREA", "SHAPE_LEN"]


def aggregate_band_to_field(df):
    """Aggregate a single band's pixel data to field level (mean per fid)."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + ["crop_name"]]
    mapping = {c: "mean" for c in feature_cols}
    if "crop_name" in df.columns:
        mapping["crop_name"] = lambda x: x.mode().iat[0] if not x.mode().empty else None
    return df.groupby("fid").agg(mapping).reset_index()


def merge_test_features():
    """Aggregate each band to field level, then merge on fid.

    Bands may have different pixel grids (e.g. B12 has extra pixels),
    so we aggregate first to avoid inner-join data loss.
    """
    base_df = None

    for band in BANDS:
        path = os.path.join(TEST_FEATURES_DIR, f"testing_{band}_{TEST_REGION}.parquet")
        print(f"Reading {os.path.basename(path)}...")
        df = pd.read_parquet(path)
        print(f"  Pixels: {df.shape}, Fields: {df['fid'].nunique()}")

        # Aggregate to field level first
        field_df = aggregate_band_to_field(df)
        print(f"  Field-level: {field_df.shape}")
        del df

        if base_df is None:
            base_df = field_df
        else:
            # Only take feature columns not already in base_df (avoids _x/_y suffixes)
            feature_cols = [c for c in field_df.columns if c not in META_COLS and c not in base_df.columns]
            if feature_cols:
                base_df = base_df.merge(
                    field_df[["fid"] + feature_cols],
                    on="fid",
                    how="inner",
                )
                print(f"  After merge: {base_df.shape}")
            else:
                print(f"  Skipped (no new features)")

    return base_df


def main():
    print("=== Combining Test Parquets ===")
    print(f"Test region: {TEST_REGION}")

    # Aggregate each band to field level, then merge
    field_df = merge_test_features()
    print(f"\nFinal shape: {field_df.shape}")
    print(f"Unique fields: {field_df['fid'].nunique()}")

    # Save
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
