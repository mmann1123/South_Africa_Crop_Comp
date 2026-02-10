"""
Shared utilities for patch data creation and pixel extraction.

Used by both training (Create_Patches.py, Create Master Data.py) and
test (create_test_patches.py) pipelines to ensure identical processing.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box


def find_raster_files(root_dir, exclude_months=None):
    """Find all band .tif files, excluding non-band files and bad months.

    Args:
        root_dir: directory to walk for .tif files
        exclude_months: set of month strings to skip (e.g. {"05", "06"}).
            If None, imports EXCLUDE_MONTHS from config.

    Returns dict mapping band_key (filename without extension) to full path.
    """
    if exclude_months is None:
        try:
            from config import EXCLUDE_MONTHS
            exclude_months = EXCLUDE_MONTHS
        except ImportError:
            exclude_months = set()

    raster_dict = {}
    exclude_files = {"rgb.tif", "rgb"}
    skipped = []

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".tif"):
                key = os.path.splitext(f)[0]
                if key.lower() in exclude_files or f.lower() in exclude_files:
                    continue
                # Check if filename contains an excluded month
                # Filenames like SA_B2_1C_2017_05.tif — last part before .tif is month
                parts = key.split("_")
                if exclude_months and len(parts) >= 2 and parts[-1] in exclude_months:
                    skipped.append(f)
                    continue
                full_path = os.path.join(dirpath, f)
                if key in raster_dict:
                    print(f"Warning: Duplicate key '{key}' found. Overwriting.")
                raster_dict[key] = full_path

    if skipped:
        print(f"Skipped {len(skipped)} rasters for excluded months {exclude_months}: "
              f"{sorted(skipped)[:6]}{'...' if len(skipped) > 6 else ''}")

    return raster_dict


def preload_rasters(raster_files):
    """Load all rasters into memory as numpy arrays.

    Args:
        raster_files: dict mapping band_key to .tif file path

    Returns:
        dict mapping band_key to (numpy_array, affine_transform, crs)
        where numpy_array is shape (height, width) float32
    """
    preloaded = {}
    crs = None

    for i, (band_key, tif_path) in enumerate(sorted(raster_files.items())):
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            transform = src.transform
            if crs is None:
                crs = src.crs
            preloaded[band_key] = (arr, transform, src.crs)

        if (i + 1) % 10 == 0 or (i + 1) == len(raster_files):
            print(f"  Loaded {i + 1}/{len(raster_files)} rasters...")

    print(f"Preloaded {len(preloaded)} rasters into memory")
    return preloaded


def deduplicate_fields(gdf):
    """Drop rows with NaN fid, then drop duplicate fids (keep first).

    Args:
        gdf: GeoDataFrame with 'fid' column

    Returns:
        Cleaned GeoDataFrame with unique, non-NaN fids
    """
    original_count = len(gdf)

    # Drop NaN fids
    nan_count = gdf["fid"].isna().sum()
    if nan_count > 0:
        print(f"[WARNING] Dropping {nan_count} rows with NaN fid values")
        gdf = gdf[gdf["fid"].notna()].reset_index(drop=True)

    # Drop duplicate fids
    dup_count = gdf["fid"].duplicated().sum()
    if dup_count > 0:
        print(f"[WARNING] Dropping {dup_count} duplicate fid rows (keeping first)")
        gdf = gdf.drop_duplicates(subset="fid", keep="first").reset_index(drop=True)

    print(f"Fields: {original_count} -> {len(gdf)} (unique fids: {gdf['fid'].nunique()})")
    return gdf


def create_patches(gdf, patch_size=100.0):
    """Create patches from field geometries.

    Small fields (bounding box area < patch_size^2): use entire field as one patch.
    Large fields: create grid of patch_size x patch_size patches, only keep those
    fully within the field boundary. Fields with no grid patches get the whole field.

    Args:
        gdf: GeoDataFrame with 'fid', 'crop_name', and geometry columns
        patch_size: patch dimension in CRS units (meters for UTM)

    Returns:
        GeoDataFrame with columns: field_id, crop_name, geometry
    """
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
            patches_list.append({
                "field_id": field_id,
                "crop_name": crop_name,
                "geometry": field_geom,
            })
            indices_with_patches.add(idx)
        else:
            x_coords = np.arange(minx, maxx, patch_size)
            y_coords = np.arange(miny, maxy, patch_size)
            added_patch = False

            for x in x_coords:
                for y in y_coords:
                    patch_poly = box(x, y, x + patch_size, y + patch_size)
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
    missing_indices = set(gdf.index) - indices_with_patches
    if missing_indices:
        print(f"Fields with no grid patches (using full field): {len(missing_indices)}")

    for idx in missing_indices:
        field_row = gdf.loc[idx]
        patches_list.append({
            "field_id": field_row["fid"],
            "crop_name": field_row.get("crop_name", None),
            "geometry": field_row.geometry,
        })

    patches_gdf = gpd.GeoDataFrame(patches_list, crs=gdf.crs)
    print(f"Created {len(patches_gdf)} patches from {gdf['fid'].nunique()} fields")
    return patches_gdf


def collate_patch_data(patches_gdf, preloaded_rasters, output_parquet, chunk_size=100000):
    """Extract pixel values from preloaded rasters using numpy slicing.

    For each patch, converts bounding box to pixel coordinates via affine
    inverse, slices the preloaded numpy arrays directly, and builds the
    DataFrame with vectorized operations.

    Args:
        patches_gdf: GeoDataFrame with field_id, crop_name, geometry
        preloaded_rasters: dict from preload_rasters()
        output_parquet: path to save output parquet
        chunk_size: rows per chunk for memory management

    Returns:
        Final DataFrame
    """
    if not preloaded_rasters:
        raise ValueError("No preloaded rasters provided")

    # Get reference CRS and transform from first raster
    ref_key = next(iter(preloaded_rasters))
    _, ref_transform, ref_crs = preloaded_rasters[ref_key]

    # Reproject patches to match raster CRS if needed
    if patches_gdf.crs != ref_crs:
        print(f"Reprojecting patches from {patches_gdf.crs} to {ref_crs}...")
        patches_gdf = patches_gdf.to_crs(ref_crs)

    # Sort band keys for consistent column ordering
    sorted_bands = sorted(preloaded_rasters.keys())

    # Inverse transform for converting geographic coords to pixel coords
    inv_transform = ~ref_transform

    df_chunks = []
    rows_collected = 0

    print(f"\nProcessing {len(patches_gdf)} patches...")

    for patch_idx, patch_row in patches_gdf.iterrows():
        patch_id = patch_idx + 1
        field_id = patch_row.get("field_id", None)
        crop_name = patch_row.get("crop_name", None)
        geom = patch_row.geometry

        # Get bounding box in pixel coordinates
        minx, miny, maxx, maxy = geom.bounds
        # Convert geographic bounds to pixel coordinates
        col_start_f, row_start_f = inv_transform * (minx, maxy)  # top-left
        col_end_f, row_end_f = inv_transform * (maxx, miny)  # bottom-right

        # Round to integer pixel coords
        r0 = max(0, int(np.floor(row_start_f)))
        r1 = max(0, int(np.ceil(row_end_f)))
        c0 = max(0, int(np.floor(col_start_f)))
        c1 = max(0, int(np.ceil(col_end_f)))

        if r1 <= r0 or c1 <= c0:
            continue

        H = r1 - r0
        W = c1 - c0
        n_pixels = H * W

        # Build row/col indices
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        rr_flat = rr.ravel()
        cc_flat = cc.ravel()

        # Extract all band values via numpy slicing
        band_data = {}
        for band_key in sorted_bands:
            arr, _, _ = preloaded_rasters[band_key]
            # Clip to array bounds
            ar0 = max(0, min(r0, arr.shape[0]))
            ar1 = max(0, min(r1, arr.shape[0]))
            ac0 = max(0, min(c0, arr.shape[1]))
            ac1 = max(0, min(c1, arr.shape[1]))

            if ar1 > ar0 and ac1 > ac0:
                sliced = arr[ar0:ar1, ac0:ac1]
                # Pad if slice is smaller than expected (edge patches)
                if sliced.shape != (H, W):
                    padded = np.zeros((H, W), dtype=np.float32)
                    ph, pw = sliced.shape
                    padded[:ph, :pw] = sliced
                    sliced = padded
                band_data[band_key] = sliced.ravel()
            else:
                band_data[band_key] = np.zeros(n_pixels, dtype=np.float32)

        # Build chunk data dict
        chunk_data = {
            "patch_id": np.full(n_pixels, patch_id, dtype=np.int64),
            "field_id": np.full(n_pixels, field_id if field_id is not None else np.nan),
            "crop_name": [crop_name] * n_pixels,
            "row": rr_flat,
            "col": cc_flat,
        }
        for band_key in sorted_bands:
            chunk_data[band_key] = band_data[band_key]

        df_chunks.append(pd.DataFrame(chunk_data))
        rows_collected += n_pixels

        if (patch_idx + 1) % 500 == 0:
            print(f"  Processed {patch_idx + 1}/{len(patches_gdf)} patches "
                  f"({rows_collected:,} rows)...")

    print(f"  Total: {len(patches_gdf)} patches, {rows_collected:,} rows")

    # Concatenate all chunks
    if not df_chunks:
        print("Warning: No patch data extracted")
        return pd.DataFrame()

    final_df = pd.concat(df_chunks, ignore_index=True)
    print(f"\nFinal shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Unique patches: {final_df['patch_id'].nunique()}")
    print(f"Unique fields: {final_df['field_id'].nunique()}")

    if "crop_name" in final_df.columns:
        print(f"\nCrop distribution:")
        print(final_df["crop_name"].value_counts(dropna=False))

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    final_df.to_parquet(output_parquet, index=False)
    print(f"\nSaved: {output_parquet}")

    return final_df
