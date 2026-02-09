"""
Create patch data for the test region (34S_20E_259N).

This script creates 100x100m patches from test field geometries and extracts
pixel values from GeoTIFF imagery. Required for 3D CNN inference.

Requirements:
  - Test labels GeoJSON exists
  - Test region GeoTIFFs exist (band imagery by month)

Input: labels.geojson (test region field boundaries)
Output: test_patches.geojson, test_patch_data.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping

# Check for rasterio
try:
    import rasterio
    from rasterio.mask import mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Import config for shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deep_learn", "src"))
from config import (
    TEST_LABELS_DIR as CONFIG_TEST_LABELS_DIR, TEST_REGION,
    TEST_PATCHES_GEOJSON_PATH, TEST_PATCH_DATA_PATH, DATA_OUTPUT_DIR,
)

# Test region labels
TEST_LABELS_DIR = os.path.join(
    CONFIG_TEST_LABELS_DIR,
    f"ref_fusion_competition_south_africa_test_labels_{TEST_REGION}",
)
TEST_LABELS_GEOJSON = os.path.join(TEST_LABELS_DIR, "labels.geojson")

# Test region imagery (NOTE: may need to be extracted/processed first)
TEST_DATA_DIR = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/S1c_data_test"

# Output (centralized in data/)
OUTPUT_PATCHES_GEOJSON = TEST_PATCHES_GEOJSON_PATH
OUTPUT_PATCH_DATA = TEST_PATCH_DATA_PATH

# Patch parameters
PATCH_SIZE = 100.0  # meters
CHUNK_SIZE = 100000


def find_raster_files(root_dir):
    """Find all band .tif files, excluding non-band files."""
    if not os.path.exists(root_dir):
        return {}

    raster_dict = {}
    exclude_files = {"rgb.tif", "rgb"}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".tif"):
                key = os.path.splitext(f)[0]
                if key.lower() in exclude_files or f.lower() in exclude_files:
                    continue
                full_path = os.path.join(dirpath, f)
                raster_dict[key] = full_path

    return raster_dict


def create_patches(gdf, patch_size=PATCH_SIZE):
    """Create 100x100m patches from field geometries."""
    patches_list = []
    indices_with_patches = set()

    for idx, field in gdf.iterrows():
        field_id = field["fid"]
        crop_name = field.get("crop_name", None)
        field_geom = field.geometry

        minx, miny, maxx, maxy = field_geom.bounds
        width = maxx - minx
        height = maxy - miny
        area = width * height

        if area < patch_size * patch_size:
            # Small field: use entire field as one patch
            patches_list.append({
                "field_id": field_id,
                "crop_name": crop_name,
                "geometry": field_geom,
            })
            indices_with_patches.add(idx)
        else:
            # Large field: create grid of patches
            x_coords = np.arange(minx, maxx, patch_size)
            y_coords = np.arange(miny, maxy, patch_size)
            added_patch = False

            for x in x_coords:
                for y in y_coords:
                    patch_poly = box(x, y, x + patch_size, y + patch_size)
                    # Only add patches fully within the field
                    if patch_poly.within(field_geom):
                        patches_list.append({
                            "field_id": field_id,
                            "crop_name": crop_name,
                            "geometry": patch_poly,
                        })
                        added_patch = True

            if added_patch:
                indices_with_patches.add(idx)

    # Add full field for fields with no patches (irregular shapes)
    all_indices = set(gdf.index)
    missing_indices = all_indices - indices_with_patches

    for idx in missing_indices:
        field_row = gdf.loc[idx]
        patches_list.append({
            "field_id": field_row["fid"],
            "crop_name": field_row.get("crop_name", None),
            "geometry": field_row.geometry,
        })

    patches_gdf = gpd.GeoDataFrame(patches_list, crs=gdf.crs)
    return patches_gdf


def collate_patch_data(patches_gdf, raster_files, output_parquet, chunk_size=CHUNK_SIZE):
    """Extract pixel values from rasters for each patch."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for patch data extraction")

    if not raster_files:
        raise ValueError("No raster files found")

    # Reproject patches to match raster CRS
    ref_raster = next(iter(raster_files.values()))
    with rasterio.open(ref_raster) as src_ref:
        raster_crs = src_ref.crs

    if patches_gdf.crs != raster_crs:
        print(f"Reprojecting patches from {patches_gdf.crs} to {raster_crs}...")
        patches_gdf = patches_gdf.to_crs(raster_crs)

    df_chunks = []
    rows_list = []
    patch_count = 0

    print(f"\nProcessing {len(patches_gdf)} patches...")

    for idx, patch_row in patches_gdf.iterrows():
        patch_count += 1
        patch_id = idx + 1
        field_id = patch_row.get("field_id", None)
        crop_name = patch_row.get("crop_name", None)
        geom = [mapping(patch_row.geometry)]

        band_arrays = {}
        for band_key, tif_path in raster_files.items():
            with rasterio.open(tif_path) as src:
                out_image, _ = mask(src, geom, crop=True)
                band_arrays[band_key] = out_image[0]

        sample_shape = next(iter(band_arrays.values())).shape
        H, W = sample_shape
        flattened_bands = {bk: band_arrays[bk].flatten() for bk in band_arrays}

        for r in range(H):
            for c in range(W):
                flat_idx = r * W + c
                row_data = {
                    "patch_id": patch_id,
                    "field_id": field_id,
                    "crop_name": crop_name,
                    "row": r,
                    "col": c,
                }
                for bk in flattened_bands:
                    row_data[bk] = flattened_bands[bk][flat_idx]
                rows_list.append(row_data)

                if len(rows_list) >= chunk_size:
                    df_chunk = pd.DataFrame(rows_list)
                    df_chunks.append(df_chunk)
                    rows_list = []

        if patch_count % 100 == 0:
            print(f"Processed {patch_count}/{len(patches_gdf)} patches...")

    if rows_list:
        df_chunks.append(pd.DataFrame(rows_list))

    final_df = pd.concat(df_chunks, ignore_index=True)
    print(f"\nFinal shape: {final_df.shape}")

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    final_df.to_parquet(output_parquet, index=False)
    print(f"Saved: {output_parquet}")

    return final_df


def main():
    print("=== Create Test Patches ===")

    # Check test labels exist
    if not os.path.exists(TEST_LABELS_GEOJSON):
        print(f"\nError: Test labels not found: {TEST_LABELS_GEOJSON}")
        return

    # Check test imagery
    raster_files = find_raster_files(TEST_DATA_DIR)
    if not raster_files:
        print(f"\nWarning: No test region GeoTIFFs found in {TEST_DATA_DIR}")
        print("\nTo use 3D CNN inference, you need to:")
        print("1. Extract ref_fusion_competition_south_africa_test_source_sentinel_2.tar.gz")
        print("2. Process the imagery into monthly band GeoTIFFs (similar to training data)")
        print("3. Place them in a directory and update TEST_DATA_DIR in this script")
        print("\nContinuing to create patch geometries only...")

    # Load test labels
    print(f"\nLoading test labels: {TEST_LABELS_GEOJSON}")
    gdf = gpd.read_file(TEST_LABELS_GEOJSON)
    print(f"Fields: {len(gdf)}")
    print(f"CRS: {gdf.crs}")

    # Filter NaN fid values
    nan_count = gdf["fid"].isna().sum()
    if nan_count > 0:
        print(f"[WARNING] Dropping {nan_count} rows with NaN fid values")
        gdf = gdf[gdf["fid"].notna()].reset_index(drop=True)

    # Create patches
    print("\nCreating patches...")
    patches_gdf = create_patches(gdf)
    print(f"Created {len(patches_gdf)} patches from {gdf['fid'].nunique()} fields")

    # Save patch geometries
    patches_gdf.to_file(OUTPUT_PATCHES_GEOJSON, driver="GeoJSON")
    print(f"\nSaved: {OUTPUT_PATCHES_GEOJSON}")

    # Extract patch data if rasters exist
    if raster_files:
        print(f"\nFound {len(raster_files)} raster files")
        print("Extracting pixel values...")
        collate_patch_data(patches_gdf, raster_files, OUTPUT_PATCH_DATA)
    else:
        print("\nSkipping pixel extraction (no raster files)")
        print(f"Patch geometries saved to: {OUTPUT_PATCHES_GEOJSON}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total patches: {len(patches_gdf)}")
    print(f"Unique fields: {patches_gdf['field_id'].nunique()}")
    if "crop_name" in patches_gdf.columns:
        print(f"Crop distribution:\n{patches_gdf['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()
