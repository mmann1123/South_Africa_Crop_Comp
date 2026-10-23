"""Is the "south-central error zone" (CEA supplement Sec. S-spatial: fields 8 to 16 km from the
training tiles where L-TAE and CNN-BiLSTM fell to about 30 to 36% correct) the footprint of the
September cloud gap?

Per holdout field: the share of pixels with September missing (B2_September is NaN; the six
bands are missing together in all but 0.96% of pixels). Maps that share next to correct/incorrect
fields for L-TAE and CNN-BiLSTM under the zero fill (the published inputs) and the training-median
fill, and for TabNet (pixel) as published. Fits, per model and fill, a logit of field correctness
on September-missing share, distance to the training region (per 10 km) and crop class, and
tabulates accuracy inside and outside the 8 to 16 km zone split by September status.

Writes figures/sept_gap_error_map.{pdf,png} and results/sept_gap_{logit,zone}.csv.
Run in the deep_field env: python experiments/missingness/sept_gap_map.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "distance_decay"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
import spatial_robustness as sr  # noqa: E402  (load_geometry, figstyle)
from config import MERGED_DL_TEST_PATH  # noqa: E402

PRED = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
ZONE = (8, 16)  # km, as described in the CEA supplement
# label -> prediction file stem in holdout_fill_check/predictions; fills scored for each
RUNS = [("L-TAE", "ltae", ["orig", "trainmed"]), ("CNN-BiLSTM", "cnn_bilstm", ["orig", "trainmed"]),
        ("TempCNN", "tempcnn", ["orig", "trainmed"]), ("TabNet (pixel)", "tabnet", ["orig", "trainmed"]),
        ("XGBoost (field)", "xgboost", ["orig", "trainmed"])]
FILL_LABEL = {"orig": "published inputs", "trainmed": "training-median fill"}


def sept_share():
    t = pq.read_table(MERGED_DL_TEST_PATH, columns=["fid", "B2_September"]).to_pandas()
    return t["B2_September"].isna().groupby(t["fid"]).mean().rename("sept_missing")


def correctness(te, stem, tag):
    p = pd.read_csv(f"{PRED}/{stem}_{tag}.csv").drop_duplicates("fid").set_index("fid")["crop_name"]
    g = te.set_index("fid")
    return (g["crop_name"] == p.reindex(g.index)).where(p.reindex(g.index).notna()).rename("ok")


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tr, te, union = sr.load_geometry()
    te = te.join(sept_share(), on="fid")
    assert te["sept_missing"].notna().mean() > 0.99, "fields without pixels"
    te["sept_any"] = te["sept_missing"] > 0
    te["zone"] = te["dist_km"].between(*ZONE)

    logit, zone = [], []
    ok = {}
    for lab, stem, tags in RUNS:
        for tag in tags:
            d = te.set_index("fid")[["crop_name", "dist_km", "sept_missing", "sept_any", "zone"]].join(
                correctness(te, stem, tag)).dropna(subset=["ok"])
            ok[(lab, tag)] = d["ok"]
            d["ok"] = d["ok"].astype(int)
            for spec, f in [("distance only", "ok ~ I(dist_km/10) + C(crop_name)"),
                            ("distance + September", "ok ~ I(dist_km/10) + sept_missing + C(crop_name)")]:
                fit = smf.logit(f, data=d).fit(disp=0)
                for term in ("I(dist_km / 10)", "sept_missing"):
                    if term in fit.params:
                        lo, hi = fit.conf_int().loc[term]
                        logit.append({"model": lab, "inputs": FILL_LABEL[tag], "spec": spec, "term": term,
                                      "coef": fit.params[term], "ci_lo": lo, "ci_hi": hi, "p": fit.pvalues[term]})
            for (z, s), g in d.groupby(["zone", "sept_any"]):
                zone.append({"model": lab, "inputs": FILL_LABEL[tag], "zone_8_16km": z, "sept_missing_any": s,
                             "n_fields": len(g), "pct_correct": 100 * g["ok"].mean()})
    lg, zn = pd.DataFrame(logit).round(4), pd.DataFrame(zone).round(2)
    lg.to_csv(os.path.join(HERE, "results", "sept_gap_logit.csv"), index=False)
    zn.to_csv(os.path.join(HERE, "results", "sept_gap_zone.csv"), index=False)

    # figure
    sr.apply_style()
    panels = [("L-TAE", "orig"), ("CNN-BiLSTM", "orig"), ("TabNet (pixel)", "orig"),
              ("L-TAE", "trainmed"), ("CNN-BiLSTM", "trainmed")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    xmin, ymin, xmax, ymax = te.total_bounds
    pad = 1000
    rings = [sr.gpd.GeoSeries([union.buffer(k * 1000).boundary], crs=te.crs) for k in ZONE]
    ax = axes[0, 0]
    te.plot(ax=ax, color="#e6e6e6", linewidth=0)
    te[te.sept_any].plot(ax=ax, column="sept_missing", cmap="Blues", vmin=0, vmax=1, linewidth=0)
    cax = ax.inset_axes([0.62, 0.06, 0.32, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1)), cax=cax,
                      orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Share of field's pixels missing", fontsize=7)
    ax.set_title(f"(a) September missing ({100 * te.sept_any.mean():.0f}% of fields)", fontsize=10)
    for k, (axx, (lab, tag)) in enumerate(zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]],
                                              panels)):
        g = te.set_index("fid").join(ok[(lab, tag)])
        g[g.ok == 1].plot(ax=axx, color="#bfbfbf", linewidth=0)
        g[g.ok == 0].plot(ax=axx, color="#d55e00", linewidth=0)
        axx.set_title(f"({chr(98 + k)}) {lab}, {FILL_LABEL[tag]}: {g.ok.mean():.0%} correct", fontsize=9)
    for ax in axes.flat:
        tr.plot(ax=ax, color="#4c72b0", alpha=0.25, linewidth=0)
        for r, ls in zip(rings, ["--", "-."]):
            r.plot(ax=ax, color="black", linewidth=0.6, linestyle=ls)
        ax.set_xlim(xmin - pad - 6000, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5)
    fig.legend(handles=[Patch(color="#bfbfbf", label="Correct"), Patch(color="#d55e00", label="Incorrect"),
                        Patch(color="#4c72b0", alpha=0.25, label="Training fields (edge)"),
                        plt.Line2D([], [], color="black", lw=0.6, ls="--", label=f"{ZONE[0]} km"),
                        plt.Line2D([], [], color="black", lw=0.6, ls="-.", label=f"{ZONE[1]} km")],
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"sept_gap_error_map.{ext}"), dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 200)
    zf = te.groupby("zone")["sept_any"].agg(["size", "mean"])
    print("September-missing share of fields, outside/inside the 8-16 km zone:\n", zf.round(3).to_string())
    print(zn.pivot_table(index=["model", "inputs"], columns=["zone_8_16km", "sept_missing_any"],
                         values="pct_correct").round(0).to_string())
    print(lg.to_string(index=False))


if __name__ == "__main__":
    main()
