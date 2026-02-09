## This file creates the master parquet for Deep Learning with patches. It needs patch geometries and the .tif files from the sentinel 2 imagery for the entire area

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_GEOJSON_PATH, PATCH_DATA_PATH, DATA_DIR

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping


# Set paths - you would need a folder containing all the .tif files
BOX_DIR = DATA_DIR
PATCH_GEOJSON = PATCH_GEOJSON_PATH
OUTPUT_PARQUET = PATCH_DATA_PATH

# Chunking as we ran into RAM issues
CHUNK_SIZE = 100000

def find_raster_files(root_dir):
    """Find all band .tif files, excluding non-band files like rgb.tif."""
    raster_dict = {}
    # Exclude non-band files
    exclude_files = {'rgb.tif', 'rgb'}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".tif"):
                key = os.path.splitext(f)[0]
                # Skip non-band files
                if key.lower() in exclude_files or f.lower() in exclude_files:
                    print(f"Skipping non-band file: {f}")
                    continue
                full_path = os.path.join(dirpath, f)
                if key in raster_dict:
                    print(f"Warning: Duplicate key '{key}' found. Overwriting previous entry.")
                raster_dict[key] = full_path
    return raster_dict


def collate_patch_data(patch_geojson, raster_files, output_parquet, chunk_size=CHUNK_SIZE):
    """
    Loads each patch polygon from patch_geojson, masks each raster,
    flattens pixel arrays, and saves a DataFrame to output_parquet.
    The resulting DataFrame includes columns:
      patch_id, field_id, crop_name, row, col, plus one column per band key.

    Uses a chunking approach to avoid memory issues.
    """
    print("Loading patch-level GeoJSON...")
    gdf = gpd.read_file(patch_geojson)
    print("GeoDataFrame shape:", gdf.shape)
    print("Columns:", gdf.columns.tolist())
    print("CRS:", gdf.crs)
    print("Bounds:", gdf.total_bounds)

    if not raster_files:
        print("No .tif files were found in the BOX_DIR. Exiting...")
        return pd.DataFrame()

    # Reproject fields to match the CRS of the first raster.
    ref_raster = next(iter(raster_files.values()))
    with rasterio.open(ref_raster) as src_ref:
        raster_crs = src_ref.crs
    if gdf.crs != raster_crs:
        print(f"Reprojecting patches from {gdf.crs} to {raster_crs}...")
        gdf = gdf.to_crs(raster_crs)

    df_chunks = []
    rows_list = []
    patch_count = 0

    print("\nMasking each patch. This may take a while if many patches...")
    for idx, patch_row in gdf.iterrows():
        patch_count += 1
        patch_id = idx + 1
        field_id = patch_row.get("field_id", None)
        crop_name = patch_row.get("crop_name", None)
        geom = [mapping(patch_row.geometry)]
        band_arrays = {}

        for band_key, tif_path in raster_files.items():
            with rasterio.open(tif_path) as src:
                out_image, out_transform = mask(src, geom, crop=True)
                band_arrays[band_key] = out_image[0]  # (height, width)

        sample_shape = next(iter(band_arrays.values())).shape  # (H, W)
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
                    "col": c
                }
                for bk in flattened_bands:
                    row_data[bk] = flattened_bands[bk][flat_idx]
                rows_list.append(row_data)

                if len(rows_list) >= chunk_size:
                    df_chunk = pd.DataFrame(rows_list)
                    df_chunks.append(df_chunk)
                    print(
                        f"Chunk saved with {len(df_chunk)} rows; total rows so far: {sum(len(chunk) for chunk in df_chunks)}")
                    rows_list = []

        # This can be ommited, just printing progress but slows down execution
        if patch_count % 100 == 0:
            print(f"Processed {patch_count} patches...")

    if rows_list:
        df_chunk = pd.DataFrame(rows_list)
        df_chunks.append(df_chunk)
        print(f"Final chunk saved with {len(df_chunk)} rows.")

    # Concatenate all chunks.
    final_df = pd.concat(df_chunks, ignore_index=True)
    print(f"\nFinal patch-level DataFrame shape: {final_df.shape}")
    print("Columns in DataFrame:", final_df.columns.tolist())
    print("\nHead:\n", final_df.head(5))

    print("\nValue counts of patch_id (top 10):")
    print(final_df["patch_id"].value_counts().head(10))
    print("\nValue counts of field_id (top 10):")
    print(final_df["field_id"].value_counts(dropna=False).head(10))
    print("\nValue counts of crop_name (top 10):")
    print(final_df["crop_name"].value_counts(dropna=False).head(10))

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    final_df.to_parquet(output_parquet, index=False)
    print(f"\nSaved patch-level data to: {output_parquet}")

    return final_df


def main():
    raster_files = find_raster_files(BOX_DIR)
    print(f"Found {len(raster_files)} .tif files in '{BOX_DIR}'.")
    collate_patch_data(
        patch_geojson=PATCH_GEOJSON,
        raster_files=raster_files,
        output_parquet=OUTPUT_PARQUET
    )


if __name__ == "__main__":
    main()
