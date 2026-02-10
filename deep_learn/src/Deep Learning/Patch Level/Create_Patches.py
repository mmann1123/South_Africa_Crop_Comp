import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import COMBINED_FIELDS_PATH, PATCH_GEOJSON_PATH
from patch_utils import deduplicate_fields, create_patches

import geopandas as gpd
import matplotlib.pyplot as plt


def main():
    geojson_path = COMBINED_FIELDS_PATH
    patches_geojson_out = PATCH_GEOJSON_PATH

    print(f"Loading fields: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    print(f"Original fields: {len(gdf)}")
    print(f"CRS: {gdf.crs}")

    # Deduplicate: drop NaN fids and duplicate fids
    gdf = deduplicate_fields(gdf)

    print(f"Unique fields: {gdf['fid'].nunique()}")
    print(f"Crop distribution:\n{gdf['crop_name'].value_counts()}")

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title("Original Field Boundaries")
    plt.show()

    # Create patches using shared utility
    patches_gdf = create_patches(gdf, patch_size=100.0)

    fig, ax = plt.subplots(figsize=(10, 8))
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
