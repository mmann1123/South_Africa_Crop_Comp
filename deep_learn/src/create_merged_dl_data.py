"""
Merge per-band per-region parquet files into a single parquet for deep learning models.

Creates:
  - merged_dl_258_259.parquet  (training data: regions 258N + 259N)
  - merged_dl_test_259N.parquet (test data: region 259N at 20E, inference only)

Column renaming:
  Training: "B2/SA_B2_1C_2017_01" -> "B2_January"
  Test:     "SA_B2_1C_2017_01"    -> "B2_January"
"""

import os
import pandas as pd
from config import (
    DATA_DIR, MERGED_DL_PATH, MERGED_DL_TEST_PATH,
    BANDS, TRAIN_REGIONS, TEST_REGION, MONTH_MAP,
)

META_COLS_TRAIN = ["id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name"]
META_COLS_TEST = ["id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN"]


def rename_band_cols(columns, band, is_test=False):
    """Build rename mapping from raw parquet column names to {BAND}_{MonthName}."""
    rename_map = {}
    meta = META_COLS_TEST if is_test else META_COLS_TRAIN
    for col in columns:
        if col in meta:
            continue
        # Extract month number from end of column name
        # Training format: "B2/SA_B2_1C_2017_01"
        # Test format:     "SA_B2_1C_2017_01"
        month_num = col.rsplit("_", 1)[-1]
        if month_num in MONTH_MAP:
            rename_map[col] = f"{band}_{MONTH_MAP[month_num]}"
        else:
            print(f"  Warning: unrecognized column '{col}', skipping")
    return rename_map


def merge_region(region, bands, is_test=False):
    """Merge all bands for a single region into one DataFrame."""
    meta_cols = META_COLS_TEST if is_test else META_COLS_TRAIN

    base_df = None
    expected_rows = None

    for band in bands:
        if is_test:
            path = os.path.join(DATA_DIR, f"X_testing_{band}_raw_{region}.parquet")
        else:
            path = os.path.join(DATA_DIR, f"{band}_raw_{region}.parquet")

        print(f"  Reading {os.path.basename(path)}...")
        df = pd.read_parquet(path)

        # Verify row count consistency across bands
        if expected_rows is None:
            expected_rows = len(df)
        else:
            assert len(df) == expected_rows, (
                f"Row count mismatch for {band} in {region}: "
                f"expected {expected_rows}, got {len(df)}"
            )

        rename_map = rename_band_cols(df.columns, band, is_test=is_test)
        df = df.rename(columns=rename_map)
        new_band_cols = list(rename_map.values())

        if base_df is None:
            base_df = df
        else:
            # Join only the new band columns (metadata is identical)
            base_df = pd.concat([base_df, df[new_band_cols]], axis=1)

    return base_df


def main():
    # --- Training data ---
    print("=== Merging training data ===")
    region_dfs = []
    for region in TRAIN_REGIONS:
        print(f"\nRegion: {region}")
        df = merge_region(region, BANDS, is_test=False)
        region_dfs.append(df)
        print(f"  Shape: {df.shape}")

    merged_train = pd.concat(region_dfs, ignore_index=True)
    print(f"\nMerged training shape: {merged_train.shape}")
    print(f"Crop classes: {merged_train['crop_name'].unique()}")

    # Verify feature columns match expected pattern
    feature_cols = [c for c in merged_train.columns if any(
        c.startswith(f"{b}_") for b in BANDS
    )]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:6]}... ")

    os.makedirs(os.path.dirname(MERGED_DL_PATH) or ".", exist_ok=True)
    merged_train.to_parquet(MERGED_DL_PATH, index=False)
    print(f"Saved to {MERGED_DL_PATH}")

    # --- Test data ---
    print("\n=== Merging test data ===")
    print(f"\nRegion: {TEST_REGION}")
    merged_test = merge_region(TEST_REGION, BANDS, is_test=True)
    print(f"Merged test shape: {merged_test.shape}")

    feature_cols_test = [c for c in merged_test.columns if any(
        c.startswith(f"{b}_") for b in BANDS
    )]
    print(f"Feature columns ({len(feature_cols_test)}): {feature_cols_test[:6]}... ")

    merged_test.to_parquet(MERGED_DL_TEST_PATH, index=False)
    print(f"Saved to {MERGED_DL_TEST_PATH}")


if __name__ == "__main__":
    main()
