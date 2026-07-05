#!/usr/bin/env python
"""Regenerate the EDA distribution figures (field counts and mean field area per
crop) with the shared readable style and the colorblind-safe crop palette.
Source: data/combined_training_fields.geojson
"""
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from figstyle import apply_style, CROP_COLORS

apply_style()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEO = os.path.join(REPO, "data", "combined_training_fields.geojson")
OUT = os.path.join(REPO, "writeup", "figures")

g = gpd.read_file(GEO)
order = g["crop_name"].value_counts().index.tolist()   # most abundant first
counts = g["crop_name"].value_counts().reindex(order)
area_ha = (g.assign(a=g.geometry.area / 1e4)            # m^2 -> hectares
            .groupby("crop_name")["a"].mean().reindex(order))
colors = [CROP_COLORS[c] for c in order]

# Field counts
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.bar(range(len(order)), counts.values, color=colors, edgecolor="white", linewidth=0.6)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=18, ha="right")
ax.set_ylabel("Number of fields")
ax.set_title("Number of fields per crop type")
ax.grid(axis="y"); ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fields_count.pdf"))
print("wrote fields_count.pdf", dict(zip(order, counts.values.tolist())))

# Mean field area (hectares)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.bar(range(len(order)), area_ha.values, color=colors, edgecolor="white", linewidth=0.6)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=18, ha="right")
ax.set_ylabel("Mean field area (hectares)")
ax.set_title("Average field area by crop type")
ax.grid(axis="y"); ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "avg_field_area.pdf"))
print("wrote avg_field_area.pdf", {k: round(v, 1) for k, v in zip(order, area_ha.values.tolist())})
