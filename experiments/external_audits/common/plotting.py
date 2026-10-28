"""Matplotlib theme for the audit figures (dataviz skill: light surface, fixed
categorical order, thin recessive axes, 300 dpi, PNG plus PDF)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Validated categorical palette (light surface), fixed slot order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "regular",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "figure.dpi": 110,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> Path:
    """Save PNG at 300 dpi and a PDF beside it; returns the PNG path."""
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path
