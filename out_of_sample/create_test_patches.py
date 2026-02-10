"""
Create patch data for the test region.

Uses the shared patch_utils module to ensure identical processing
between training and test data.

Input: data/test_fields.geojson (test field boundaries)
Output: data/test_patches.geojson, data/test_patch_data.parquet
"""

import os
import sys

# Import config and shared patch utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import (
    TEST_FIELDS_PATH, DATA_DIR,
    TEST_PATCHES_GEOJSON_PATH, TEST_PATCH_DATA_PATH,
)
from patch_utils import (
    find_raster_files, preload_rasters,
    deduplicate_fields, create_patches, collate_patch_data,
)

import geopandas as gpd


def main():
    print("=== Create Test Patches ===")

    # Check test fields exist
    if not os.path.exists(TEST_FIELDS_PATH):
        print(f"\nError: Test fields not found: {TEST_FIELDS_PATH}")
        return

    # Load test field boundaries
    print(f"\nLoading test fields: {TEST_FIELDS_PATH}")
    gdf = gpd.read_file(TEST_FIELDS_PATH)
    print(f"Fields: {len(gdf)}")
    print(f"CRS: {gdf.crs}")

    # Deduplicate (handles NaN fids and duplicates)
    gdf = deduplicate_fields(gdf)

    # Create patches using shared utility
    print("\nCreating patches...")
    patches_gdf = create_patches(gdf, patch_size=100.0)

    # Save patch geometries
    patches_gdf.to_file(TEST_PATCHES_GEOJSON_PATH, driver="GeoJSON")
    print(f"\nSaved: {TEST_PATCHES_GEOJSON_PATH}")

    # Find and preload rasters
    raster_files = find_raster_files(DATA_DIR)
    if not raster_files:
        print(f"\nWarning: No GeoTIFFs found in {DATA_DIR}")
        print("Patch geometries saved, but pixel extraction skipped.")
        return

    print(f"\nFound {len(raster_files)} raster files")
    print("Preloading rasters into memory...")
    preloaded = preload_rasters(raster_files)

    # Extract pixel values using shared utility
    print("\nExtracting pixel values...")
    collate_patch_data(patches_gdf, preloaded, TEST_PATCH_DATA_PATH)

    # Summary
    print("\n=== Summary ===")
    print(f"Total patches: {len(patches_gdf)}")
    print(f"Unique fields: {patches_gdf['field_id'].nunique()}")
    if "crop_name" in patches_gdf.columns:
        print(f"Crop distribution:\n{patches_gdf['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()
