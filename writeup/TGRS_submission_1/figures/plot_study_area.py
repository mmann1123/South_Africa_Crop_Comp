#!/usr/bin/env python
"""Study-area map: the two training tiles and the spatially disjoint holdout tile,
with every field colored by crop type (color-blind-safe Okabe & Ito palette).

Reads the combined training fields (tiles 34S_19E_258N + 34S_19E_259N) and the
holdout fields (34S_20E_259N), both in EPSG:32734 (UTM 34S, metres). Plots each
field polygon colored by `crop_name`, outlines the training region and the holdout
tile, splits the training region into its two tiles, and adds a scale bar, north
arrow, and crop legend.

Output: writeup/figures/study_area_map.pdf  (300 dpi, drops into the paper)
Run:    python writeup/figures/plot_study_area.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrow, Rectangle
import geopandas as gpd

import figstyle

figstyle.apply_style()
CROP_COLORS = figstyle.CROP_COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data")

TILE_Y = 6_216_000  # UTM northing dividing the two training tiles (258N south / 259N north)

train = gpd.read_file(os.path.join(DATA, "combined_training_fields.geojson"))
hold = gpd.read_file(os.path.join(DATA, "test_fields.geojson"))

# split the training region into its two source tiles by field centroid northing
tc = train.geometry.centroid
train_s = train[tc.y < TILE_Y]    # 34S_19E_258N
train_n = train[tc.y >= TILE_Y]   # 34S_19E_259N

fig, ax = plt.subplots(figsize=(7.6, 7.4))

# --- fields colored by crop (shared palette; thin edges keep small fields visible) ---
for df in (train, hold):
    for crop, color in CROP_COLORS.items():
        sub = df[df["crop_name"] == crop]
        if len(sub):
            sub.plot(ax=ax, facecolor=color, edgecolor="white", linewidth=0.05)

# --- region outlines: training (solid) vs holdout (dashed, shaded) ---
train_hull = gpd.GeoSeries([train.union_all().convex_hull], crs=train.crs)
hold_hull = gpd.GeoSeries([hold.union_all().convex_hull], crs=hold.crs)
hold_hull.plot(ax=ax, facecolor="0.5", alpha=0.07, edgecolor="none", zorder=0)
train_hull.boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=6)
hold_hull.boundary.plot(ax=ax, color="black", linewidth=1.8, linestyle=(0, (5, 3)), zorder=6)

# --- thicker dashed divider between the two training tiles ---
txmin, _, txmax, _ = train.total_bounds
ax.plot([txmin, txmax], [TILE_Y, TILE_Y], color="black", linewidth=2.4,
        linestyle=(0, (6, 4)), zorder=7)


def label(df, text, **kw):
    minx, miny, maxx, maxy = df.total_bounds
    ax.text((minx + maxx) / 2, (miny + maxy) / 2, text, ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.85), **kw)


label(train_n, "Training\n34S_19E_259N")
label(train_s, "Training\n34S_19E_258N")
label(hold, "Holdout\n34S_20E_259N\n(spatial transfer)")

# --- one framed map-legend box in the empty bottom-right (SE) corner:
#     crop swatches on the left, scale bar + north arrow in the white space
#     to the right of the (shorter) crop labels. ---
bx0, by0, bx1, by1 = 481_000, 6_194_000, 504_000, 6_214_500
ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="white",
                       edgecolor="0.4", linewidth=1.0, zorder=9))
ax.text((bx0 + bx1) / 2, by1 - 1300, "Crop type", ha="center", va="top",
        fontsize=13, fontweight="bold", zorder=10)

rows = [6_210_000, 6_207_200, 6_204_400, 6_201_600, 6_198_800]
sw = 1700  # swatch side (m)
for (crop, color), ry in zip(CROP_COLORS.items(), rows):
    ax.add_patch(Rectangle((bx0 + 1300, ry - sw / 2), sw, sw, facecolor=color,
                           edgecolor="0.4", linewidth=0.4, zorder=10))
    ax.text(bx0 + 1300 + sw + 700, ry, crop, ha="left", va="center",
            fontsize=11, zorder=10)

# north arrow in the right white space (level with the "Barley" row)
nax, nay = 500_000, 6_203_200
ax.add_patch(FancyArrow(nax, nay, 0, 2_900, width=280, head_width=1400,
                        head_length=1400, length_includes_head=True,
                        color="black", zorder=10))
ax.text(nax, nay + 3_400, "N", ha="center", va="bottom", fontsize=12,
        fontweight="bold", zorder=10)

# --- scale bar (5 km) in the lower-left white space, over a light backing ---
sx0, sy, bar_m = 457_000, 6_196_500, 5_000
ax.add_patch(Rectangle((sx0 - 1200, sy - 1600), bar_m + 2400, 4000,
                       facecolor="white", edgecolor="none", alpha=0.85, zorder=7))
ax.plot([sx0, sx0 + bar_m], [sy, sy], color="black", linewidth=2.5,
        solid_capstyle="butt", zorder=8)
ax.plot([sx0, sx0], [sy - 450, sy + 450], color="black", linewidth=1.2, zorder=8)
ax.plot([sx0 + bar_m, sx0 + bar_m], [sy - 450, sy + 450], color="black", linewidth=1.2, zorder=8)
ax.text(sx0 + bar_m / 2, sy + 900, "5 km", ha="center", va="bottom", fontsize=11, zorder=8)

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Study area: Western Cape, South Africa", fontsize=13)

out = os.path.join(HERE, "study_area_map.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
