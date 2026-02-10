## This file creates the master parquet for Deep Learning with patches.
## It needs patch geometries and the .tif files from the sentinel 2 imagery.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import PATCH_GEOJSON_PATH, PATCH_DATA_PATH, DATA_DIR
from patch_utils import find_raster_files, preload_rasters, collate_patch_data

import geopandas as gpd


def main():
    print("=== Create Master Patch Data ===")

    # Find raster files
    raster_files = find_raster_files(DATA_DIR)
    print(f"Found {len(raster_files)} .tif files in '{DATA_DIR}'")

    if not raster_files:
        print("No .tif files found. Exiting...")
        return

    # Preload all rasters into memory
    print("\nPreloading rasters into memory...")
    preloaded = preload_rasters(raster_files)

    # Load patch geometries
    print(f"\nLoading patches: {PATCH_GEOJSON_PATH}")
    patches_gdf = gpd.read_file(PATCH_GEOJSON_PATH)
    print(f"Patches: {len(patches_gdf)}")

    # Extract pixel values
    collate_patch_data(patches_gdf, preloaded, PATCH_DATA_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
