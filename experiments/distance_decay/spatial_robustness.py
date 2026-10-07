"""
Spatial robustness within the holdout tile (candidate SI material, reviewers R2-Q4 / R2-Q7).

For every Table 1 model, scores holdout fields by distance band from the training
region (macro-F1 with field-bootstrap 95% CI, accuracy), fits a class-adjusted logit of
per-field correctness on distance, summarizes by inductive-bias family, and draws:
  figures/f1_by_distance.pdf   macro-F1 by distance band (representative models + family means)
  figures/error_maps.pdf       ground truth and correct/incorrect maps for five models

Read-only on out_of_sample/; writes to experiments/distance_decay/{results,figures}/.
Run: ~/miniconda3/envs/deep_field/bin/python experiments/distance_decay/spatial_robustness.py
"""
import glob
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
from shapely.ops import unary_union
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OOS = os.path.join(REPO, "out_of_sample")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
BASE = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp"
sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
from figstyle import apply_style, FAMILY_COLORS  # noqa: E402

BINS = [0, 5, 10, 15, 25]
BIN_LABELS = ["0–5", "5–10", "10–15", "15–24"]
B = 2000

# Table 1 label -> prediction CSV
MODELS = {
    "TabNet (pixel)": "predictions_tabnet.csv",
    "L-TAE-S (field)": "predictions_ltae_s_field.csv",
    "L-TAE-S (pixel)": "predictions_ltae_s.csv",
    "L-TAE (field)": "predictions_ltae_field.csv",
    "Transformer (pixel)": "predictions_transformer.csv",
    "TempCNN (pixel)": "predictions_tempcnn.csv",
    "XGBoost (field)": "predictions_xgboost.csv",
    "LightGBM (pixel)": "predictions_base_lgbm.csv",
    "Logistic Regression (pixel)": "predictions_base_lr.csv",
    "LightGBM (field)": "predictions_lgbm.csv",
    "Voting (pixel)": "predictions_voting.csv",
    "Random Forest (pixel)": "predictions_base_rf.csv",
    "XGBoost (pixel)": "predictions_base_xgb.csv",
    "L-TAE (pixel)": "predictions_ltae.csv",
    "TempCNN (field)": "predictions_tempcnn_field.csv",
    "Stacking (field)": "predictions_stacking.csv",
    "CNN-BiLSTM (pixel)": "predictions_cnn_bilstm.csv",
    "CNN-BiLSTM (field)": "predictions_cnn_bilstm_field.csv",
    "3D CNN (patch)": "predictions_3d_cnn.csv",
    "SMOTE Stacked (field)": "predictions_smote_stacked.csv",
    "Multi-Channel 2D CNN (patch)": "predictions_multi_channel_cnn.csv",
    "TabNet raw field-avg (field)": "predictions_tabnet_temporal_field.csv",
    "TabNet xr_fresh (field)": "predictions_tabnet_field.csv",
}
# Inductive-bias families, as in plot_inductive_bias.py (Fig. 4)
FAMILY = {
    "Tree-based": ["Random Forest (pixel)", "XGBoost (pixel)", "LightGBM (pixel)",
                   "XGBoost (field)", "LightGBM (field)", "TabNet (pixel)"],
    "Linear": ["Logistic Regression (pixel)"],
    "Dense temporal / patch DL": ["CNN-BiLSTM (pixel)", "TempCNN (pixel)", "L-TAE (pixel)",
                                  "Transformer (pixel)", "3D CNN (patch)", "Multi-Channel 2D CNN (patch)"],
    "Sparse-attention": ["L-TAE-S (pixel)", "L-TAE-S (field)"],
}
FAM_COLOR = {"Tree-based": FAMILY_COLORS["tree"], "Linear": FAMILY_COLORS["linear"],
             "Dense temporal / patch DL": FAMILY_COLORS["dense"], "Sparse-attention": FAMILY_COLORS["sparse"]}
SHOW = [("TabNet (pixel)", "TabNet"), ("L-TAE-S (pixel)", "L-TAE-S"), ("XGBoost (field)", "XGBoost (field)"),
        ("L-TAE (pixel)", "L-TAE"), ("CNN-BiLSTM (pixel)", "CNN-BiLSTM")]


def load_geometry():
    tr = pd.concat([gpd.read_file(f) for f in glob.glob(
        f"{BASE}/ref_fusion_competition_south_africa_train_labels/*/labels.geojson")])
    te = gpd.read_file(glob.glob(f"{BASE}/ref_fusion_competition_south_africa_test_labels/*/labels.geojson")[0])
    crs = tr.estimate_utm_crs()
    tr, te = tr.to_crs(crs), te.to_crs(crs)
    union = unary_union(tr.geometry.values)
    te = te.drop_duplicates("fid").copy()
    te["dist_km"] = te.geometry.centroid.distance(union) / 1e3
    te["band"] = pd.cut(te["dist_km"], BINS, labels=BIN_LABELS, include_lowest=True)
    return tr, te, union


def score(te, rng):
    rows, slopes, preds = [], [], {}
    labels = sorted(te["crop_name"].unique())
    for name, f in MODELS.items():
        p = pd.read_csv(os.path.join(OOS, f)).drop_duplicates("fid").set_index("fid")["crop_name"]
        m = te.set_index("fid")[["crop_name", "dist_km", "band"]].join(p.rename("pred"), how="inner")
        m["correct"] = (m["pred"] == m["crop_name"]).astype(int)
        preds[name] = m["correct"]
        for band, g in m.groupby("band", observed=True):
            yt, yp = g["crop_name"].values, g["pred"].values
            f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
            bs = [f1_score(yt[i], yp[i], labels=labels, average="macro", zero_division=0)
                  for i in (rng.integers(0, len(g), len(g)) for _ in range(B))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"model": name, "band_km": band, "n_fields": len(g), "macro_f1": f1,
                         "ci_lo": lo, "ci_hi": hi, "accuracy": g["correct"].mean()})
        fit = smf.logit("correct ~ I(dist_km/10) + C(crop_name)", data=m).fit(disp=0)
        lo, hi = fit.conf_int().iloc[1]
        slopes.append({"model": name, "logodds_per_10km": fit.params.iloc[1], "ci_lo": lo, "ci_hi": hi,
                       "p": fit.pvalues.iloc[1]})
    return pd.DataFrame(rows), pd.DataFrame(slopes), preds


def fig_bands(res):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(BIN_LABELS))
    ax = axes[0]
    for (name, lab), mk in zip(SHOW, ["o", "s", "^", "D", "v"]):
        d = res[res.model == name].set_index("band_km").loc[BIN_LABELS]
        fam = next(k for k, v in FAMILY.items() if name in v)
        ax.errorbar(x, d.macro_f1, yerr=[d.macro_f1 - d.ci_lo, d.ci_hi - d.macro_f1], marker=mk, capsize=3,
                    color=FAM_COLOR[fam], label=lab, lw=1.4, alpha=0.9)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel("Distance to nearest training field (km)", fontsize=10)
    ax.set_ylabel("Out-of-region macro-F1", fontsize=10)
    ax.set_title("(a) Representative models", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for fam, members in FAMILY.items():
        d = res[res.model.isin(members)].groupby("band_km", observed=True)["macro_f1"].agg(["mean", "min", "max"]).loc[BIN_LABELS]
        ax.plot(x, d["mean"], marker="o", color=FAM_COLOR[fam], label=f"{fam} (n={len(members)})", lw=1.6)
        ax.fill_between(x, d["min"], d["max"], color=FAM_COLOR[fam], alpha=0.15)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel("Distance to nearest training field (km)", fontsize=10)
    ax.set_title("(b) Inductive-bias family mean (shading = range)", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"f1_by_distance.{ext}"), dpi=300)
    plt.close(fig)


def fig_maps(tr, te, union, preds):
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    xmin, ymin, xmax, ymax = te.total_bounds
    pad = 1000
    rings = [gpd.GeoSeries([union.buffer(k * 1000).boundary], crs=te.crs) for k in (5, 10, 15)]
    crops = sorted(te["crop_name"].unique())
    cmap = plt.get_cmap("tab10")
    ax = axes[0, 0]
    for i, c in enumerate(crops):
        te[te.crop_name == c].plot(ax=ax, color=cmap(i), linewidth=0)
    ax.legend(handles=[Patch(color=cmap(i), label=c) for i, c in enumerate(crops)], fontsize=6.5,
              loc="lower left", frameon=True)
    ax.set_title("(a) Ground truth", fontsize=10)
    for k, (ax, (name, lab)) in enumerate(zip(axes.flat[1:], SHOW)):
        g = te.set_index("fid").join(preds[name].rename("ok"))
        g[g.ok == 1].plot(ax=ax, color="#bfbfbf", linewidth=0)
        g[g.ok == 0].plot(ax=ax, color="#d55e00", linewidth=0)
        acc = g.ok.mean()
        ax.set_title(f"({chr(98 + k)}) {lab}: {acc:.0%} fields correct", fontsize=10)
    for ax in axes.flat:
        tr.plot(ax=ax, color="#4c72b0", alpha=0.25, linewidth=0)
        for r, ls in zip(rings, [":", "--", "-."]):
            r.plot(ax=ax, color="black", linewidth=0.6, linestyle=ls)
        ax.set_xlim(xmin - pad - 6000, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5)
    fig.legend(handles=[Patch(color="#bfbfbf", label="Correct"), Patch(color="#d55e00", label="Incorrect"),
                        Patch(color="#4c72b0", alpha=0.25, label="Training fields (edge)"),
                        plt.Line2D([], [], color="black", lw=0.6, ls=":", label="5 km"),
                        plt.Line2D([], [], color="black", lw=0.6, ls="--", label="10 km"),
                        plt.Line2D([], [], color="black", lw=0.6, ls="-.", label="15 km")],
               loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"error_maps.{ext}"), dpi=300)
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    tr, te, union = load_geometry()
    comp = pd.crosstab(te["band"], te["crop_name"])
    comp["n_fields"] = comp.sum(axis=1)
    comp.to_csv(os.path.join(RES, "class_counts_by_band.csv"))
    res, slopes, preds = score(te, np.random.default_rng(42))
    res.round(4).to_csv(os.path.join(RES, "f1_by_distance_all_models.csv"), index=False)
    slopes.round(4).to_csv(os.path.join(RES, "distance_effect_all_models.csv"), index=False)
    piv = res.pivot(index="model", columns="band_km", values="macro_f1")[BIN_LABELS]
    piv["range"] = piv.max(axis=1) - piv.min(axis=1)
    piv["min_band"] = piv[BIN_LABELS].idxmin(axis=1)
    piv = piv.loc[list(MODELS)]
    piv.round(3).to_csv(os.path.join(RES, "f1_by_distance_wide.csv"))
    fam = []
    for f, members in FAMILY.items():
        sub = piv.loc[members]
        fam.append({"family": f, "n_models": len(members),
                    **{b: sub[b].mean() for b in BIN_LABELS}, "mean_range": sub["range"].mean()})
    fam = pd.DataFrame(fam).round(3)
    fam.to_csv(os.path.join(RES, "f1_by_distance_family.csv"), index=False)
    fig_bands(res)
    fig_maps(tr, te, union, preds)
    print(comp.to_string(), "\n")
    print(piv.round(2).to_string(), "\n")
    print(fam.to_string(index=False), "\n")
    print(slopes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
