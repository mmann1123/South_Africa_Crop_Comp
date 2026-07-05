#!/usr/bin/env python
"""Deliverable C: quantify the study area for the Data Description section.

Reads the training (two-tile) and holdout (one-tile) field polygons and reports
field counts, total/mean field area, and bounding extent per region. Both layers
are in EPSG:32734 (UTM 34S, metres), so polygon areas are in m^2.

Outputs:
  writeup/study_area_stats.csv      (per-region summary)
  writeup/study_area_class_counts.csv (field counts per crop per region)
Prints a markdown table for pasting into the manuscript.
"""
import os
import geopandas as gpd
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(REPO, "writeup")

REGIONS = {
    "Training (34S_19E_258N + 34S_19E_259N)": os.path.join(DATA, "combined_training_fields.geojson"),
    "Holdout (34S_20E_259N)": os.path.join(DATA, "test_fields.geojson"),
}

summary_rows = []
class_frames = []
for name, path in REGIONS.items():
    g = gpd.read_file(path)
    area_m2 = g.geometry.area
    minx, miny, maxx, maxy = g.total_bounds
    extent_km2 = ((maxx - minx) / 1000.0) * ((maxy - miny) / 1000.0)
    summary_rows.append({
        "Region": name,
        "Fields": len(g),
        "Total field area (km^2)": round(area_m2.sum() / 1e6, 2),
        "Mean field area (ha)": round(area_m2.mean() / 1e4, 2),
        "Median field area (ha)": round(area_m2.median() / 1e4, 2),
        "Bounding extent (km^2)": round(extent_km2, 1),
    })
    vc = g["crop_name"].value_counts().rename(name)
    class_frames.append(vc)

summary = pd.DataFrame(summary_rows)
classes = pd.concat(class_frames, axis=1).fillna(0).astype(int)
classes.index.name = "Crop"

summary.to_csv(os.path.join(OUT, "study_area_stats.csv"), index=False)
classes.to_csv(os.path.join(OUT, "study_area_class_counts.csv"))

print("\n=== Per-region summary ===")
print(summary.to_markdown(index=False))
print("\n=== Field counts per crop ===")
print(classes.to_markdown())
print("\nWrote study_area_stats.csv and study_area_class_counts.csv to writeup/")
