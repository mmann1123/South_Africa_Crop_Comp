"""Shared figure style: colorblind-safe palette, fixed per-category colors, and
large readable fonts. Imported by all plot_*.py scripts so colors stay consistent
across figures (a given model/family/condition keeps its color everywhere).

Palette: Okabe & Ito (2008), the standard colorblind-safe qualitative set.
"""
import matplotlib as mpl

OKABE = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "grey":       "#999999",
}

# Fixed colors per model — used wherever that model appears.
MODEL_COLORS = {
    "TabNet":              OKABE["blue"],
    "XGBoost":             OKABE["green"],
    "LightGBM":            OKABE["skyblue"],
    "Logistic Regression": OKABE["orange"],
    "L-TAE":               OKABE["vermillion"],
    "TempCNN":             OKABE["purple"],
    "CNN-BiLSTM":          OKABE["black"],
    "LassoNet":            OKABE["grey"],
    "Random Forest":       OKABE["yellow"],
}

# Fixed colors per inductive-bias family.
FAMILY_COLORS = {
    "tree":   OKABE["green"],
    "linear": OKABE["orange"],
    "dense":  OKABE["vermillion"],
    "sparse": OKABE["skyblue"],
    "aug":    OKABE["purple"],
}

# Fixed colors per crop class — used across the EDA figures.
CROP_COLORS = {
    "Lucerne/Medics":      OKABE["blue"],
    "Wheat":               OKABE["orange"],
    "Barley":              OKABE["green"],
    "Canola":              OKABE["vermillion"],
    "Small grain grazing": OKABE["purple"],
}

# Fixed colors for the two recurring binary contrasts.
COND_INREGION = OKABE["orange"]   # in-region validation
COND_HOLDOUT  = OKABE["blue"]     # out-of-sample holdout
COND_PIXEL    = OKABE["orange"]   # pixel level
COND_FIELD    = OKABE["blue"]     # field level


def color_for_model(name):
    """Return the fixed color for a model given any of its label variants."""
    n = name.lower()
    table = [
        ("lightgbm", "LightGBM"), ("l-tae", "L-TAE"), ("tabnet", "TabNet"),
        ("xgboost", "XGBoost"), ("tempcnn", "TempCNN"), ("cnn-bilstm", "CNN-BiLSTM"),
        ("lassonet", "LassoNet"), ("random forest", "Random Forest"),
        ("logistic", "Logistic Regression"), ("base lr", "Logistic Regression"),
        (" lr", "Logistic Regression"), ("rf", "Random Forest"),
    ]
    for key, model in table:
        if key in n:
            return MODEL_COLORS[model]
    return OKABE["grey"]


def apply_style():
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "0.9",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.2,
        "lines.markersize": 8,
    })
