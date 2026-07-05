#!/usr/bin/env python
"""
Graphical abstract for the TGRS manuscript: a bump chart showing the
in-region -> spatial-transfer ranking inversion.

Dense temporal nets lead in-region and collapse under transfer; sparse,
axis-aligned feature-selection models (TabNet, L-TAE-S) climb to the top,
and the linear baseline is robust. Numbers are from Table I of tgrs-article.tex.

Colors come from the shared figstyle.py (Okabe & Ito colorblind-safe palette,
same per-family colors as every other figure in the paper).

Outputs vector PDF + 600-dpi PNG to writeup/figures/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from figstyle import apply_style, FAMILY_COLORS

# ---- data: (name, in-region F1, spatial-transfer F1, family) -----------------
# family keys match figstyle.FAMILY_COLORS so colors stay consistent with the
# paper: dense=vermillion (falls), sparse=skyblue (rises), linear=orange.
MODELS = [
    ("L-TAE",         0.78, 0.58, "dense"),
    ("Transformer",   0.78, 0.58, "dense"),
    ("L-TAE-S",       0.77, 0.60, "sparse"),   # our contribution -> starred
    ("TempCNN",       0.77, 0.56, "dense"),
    ("TabNet",        0.73, 0.60, "sparse"),
    ("CNN-BiLSTM",    0.72, 0.46, "dense"),
    ("Logistic Reg.", 0.61, 0.56, "linear"),
]
OURS = "L-TAE-S"   # highlighted with a star marker

def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    r = [0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r

left_f1 = [m[1] for m in MODELS]
right_f1 = [m[2] for m in MODELS]
left_rank = ranks(left_f1)
right_rank = ranks(right_f1)

apply_style()
plt.rcParams.update({"axes.spines.left": False, "axes.spines.bottom": False})

fig, ax = plt.subplots(figsize=(7.6, 6.4))
x_left, x_right = 0.0, 1.0
n = len(MODELS)


def bezier(x0, y0, x1, y1, color, lw, alpha, z):
    """Smooth S-curve between two rank positions."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), fc="none", ec=color,
                                    lw=lw, alpha=alpha, zorder=z, capstyle="round"))


# connecting bands: soft halo + solid line
for i, (name, lf, rf, fam) in enumerate(MODELS):
    color = FAMILY_COLORS[fam]
    y0, y1 = left_rank[i], right_rank[i]
    star = name == OURS
    bezier(x_left, y0, x_right, y1, color, lw=10, alpha=0.16, z=1)          # halo
    bezier(x_left, y0, x_right, y1, color, lw=4.2 if star else 3.0,
           alpha=0.97, z=2)                                                 # line

# nodes + labels
for i, (name, lf, rf, fam) in enumerate(MODELS):
    color = FAMILY_COLORS[fam]
    star = name == OURS
    marker = "*" if star else "o"
    s = 340 if star else 130
    weight = "bold" if star else "normal"
    ax.scatter(x_left, left_rank[i], s=s, color=color, edgecolor="white",
               linewidth=1.6, zorder=4, marker=marker)
    ax.text(x_left - 0.05, left_rank[i], f"{name}  {lf:.2f}", ha="right",
            va="center", fontsize=13, fontweight=weight, color="#222")
    ax.scatter(x_right, right_rank[i], s=s, color=color, edgecolor="white",
               linewidth=1.6, zorder=4, marker=marker)
    ax.text(x_right + 0.05, right_rank[i], f"{rf:.2f}  {name}", ha="left",
            va="center", fontsize=13, fontweight=weight, color="#222")

# column headers, centered over the model-name blocks (which hang off the
# dots at x=0 / x=1), not over the dots themselves
hdr_left, hdr_right = -0.30, 1.30
ax.text(hdr_left, 0.18, "IN-REGION", ha="center", va="bottom",
        fontsize=15, fontweight="bold", color="#333")
ax.text(hdr_left, 0.46, "(field-wise CV)", ha="center", va="bottom",
        fontsize=10.5, style="italic", color="#777")
ax.text(hdr_right, 0.18, "SPATIAL TRANSFER", ha="center", va="bottom",
        fontsize=15, fontweight="bold", color="#333")
ax.text(hdr_right, 0.46, "(disjoint holdout tile)", ha="center", va="bottom",
        fontsize=10.5, style="italic", color="#777")

# rank markers on the far left
for r in range(1, n + 1):
    ax.text(-0.66, r, f"#{r}", ha="right", va="center", fontsize=9.5,
            color="#bbb")

# takeaway banner (below the chart)
ax.text(0.5, n + 0.95, "The ranking inverts under spatial transfer",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1a1a1a")
ax.text(0.5, n + 1.45,
        "Dense temporal nets lead in-region, then collapse;\n"
        "sparse feature-selection models rise to the top.",
        ha="center", va="center", fontsize=11.5, color="#555")
ax.text(0.5, n + 0.52, "node value = macro-F1", ha="center", va="center",
        fontsize=9.5, style="italic", color="#999")

# legend (arrows encode direction; star marks our model)
handles = [
    Line2D([0], [0], color=FAMILY_COLORS["dense"], lw=3.4,
           label="Dense temporal net  ▼ falls"),
    Line2D([0], [0], color=FAMILY_COLORS["sparse"], lw=3.4,
           label="Sparse feature selection  ▲ rises"),
    Line2D([0], [0], color=FAMILY_COLORS["linear"], lw=3.4,
           label="Linear baseline (robust)"),
    Line2D([0], [0], color=FAMILY_COLORS["sparse"], lw=0, marker="*",
           markersize=15, markeredgecolor="white",
           label="L-TAE-S (ours)"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.14),
          ncol=2, frameon=False, fontsize=10.5, handlelength=1.6,
          columnspacing=1.8, labelspacing=0.7)

ax.set_xlim(-0.78, 1.78)
ax.set_ylim(n + 1.95, 0.05)   # inverted: rank 1 at top; headroom for banner
ax.axis("off")
fig.tight_layout()

fig.savefig("graphical_abstract.pdf", bbox_inches="tight")
fig.savefig("graphical_abstract.png", dpi=600, bbox_inches="tight")
print("wrote graphical_abstract.pdf and graphical_abstract.png")
