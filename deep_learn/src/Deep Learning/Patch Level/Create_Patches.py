import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import COMBINED_FIELDS_PATH, PATCH_GEOJSON_PATH

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

def main():

    # Path to the geojson. This has to be field level
    geojson_path = COMBINED_FIELDS_PATH
    patch_size = 100.0
    patches_geojson_out = PATCH_GEOJSON_PATH

    gdf = gpd.read_file(geojson_path)
    print("Original fields GeoDataFrame:")
    print(gdf.tail())
    print(gdf.columns)
    print("CRS:", gdf.crs)

    # Filter out rows with NaN fid values
    nan_count = gdf['fid'].isna().sum()
    if nan_count > 0:
        print(f"[WARNING] Dropping {nan_count} rows with NaN fid values")
        gdf = gdf[gdf['fid'].notna()].reset_index(drop=True)

    print("Number of fields:", gdf['fid'].nunique())
    print("Number of Crops:", gdf['crop_name'].value_counts().sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title("Original Field Boundaries")
    plt.show()

    patches_list = []
    # Track indices of fields that got at least one patch
    indices_with_patches = set()

    for idx, field in gdf.iterrows():
        field_id = field['fid']
        crop_name = field["crop_name"]
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
                "geometry": field_geom
            })
            indices_with_patches.add(idx)
        else:
            # Large field: create grid of 100x100m patches
            x_coords = np.arange(minx, maxx, patch_size)
            y_coords = np.arange(miny, maxy, patch_size)
            added_patch = False
            for x in x_coords:
                for y in y_coords:
                    patch_poly = box(x, y, x + patch_size, y + patch_size)

                    # Only add patches fully within the field. Avoids any data leakage issues
                    if patch_poly.within(field_geom):
                        patches_list.append({
                            "field_id": field_id,
                            "crop_name": crop_name,
                            "geometry": patch_poly
                        })
                        added_patch = True
            if added_patch:
                indices_with_patches.add(idx)

    # After processing all fields, check for fields with no patches (irregular shapes)
    all_indices = set(gdf.index)
    missing_indices = all_indices - indices_with_patches
    print(f"Fields with no patches created (using full field): {len(missing_indices)}")

    # For each missing field, add the entire field as one patch
    for idx in missing_indices:
        field_row = gdf.loc[idx]
        patches_list.append({
            "field_id": field_row["fid"],
            "crop_name": field_row["crop_name"],
            "geometry": field_row.geometry
        })

    # Create a GeoDataFrame of patches
    patches_gdf = gpd.GeoDataFrame(patches_list, crs=gdf.crs)
    print(f"\nCreated {len(patches_gdf)} patches (including full-field patches for missing fields).")

    fig, ax = plt.subplots(figsize=(10, 8))
    # gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    patches_gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='red')
    ax.set_title("Field Boundaries with Patch Grid")
    plt.show()

    if patches_geojson_out:
        patches_gdf.to_file(patches_geojson_out, driver='GeoJSON')
        print(f"Saved patch GeoDataFrame to {patches_geojson_out}")

    print(patches_gdf['crop_name'].value_counts())
    print("Unique field_id count:", patches_gdf['field_id'].nunique())

if __name__ == "__main__":
    main()
