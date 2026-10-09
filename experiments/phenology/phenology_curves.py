"""
Reviewer R2-Q2: phenological profiles of the five crops, training vs. holdout.

Field-level monthly profiles (mean of each field's pixels) for EVI, SWIR (B11, B12) and
hue over the ten months the models use (May and June are excluded for cloud). For each
crop, plots the median across fields with an interquartile band, training region solid
and holdout tile dashed. Also writes a per-month separability table: for each band and
month, the standardized difference (Cohen's d) between crop pairs, summarized as the
largest |d| over the season for each pair.

Read-only on data/; writes experiments/phenology/{figures,results}/.
Run: ~/miniconda3/envs/deep_field/bin/python experiments/phenology/phenology_curves.py
"""
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
from config import MERGED_DL_PATH, MERGED_DL_TEST_PATH  # noqa: E402
from figstyle import apply_style  # noqa: E402

MONTHS = ["January", "February", "March", "April", "July", "August",
          "September", "October", "November", "December"]
MPOS = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
PANELS = [("EVI", "EVI"), ("B11", "SWIR 1 (B11)"), ("B12", "SWIR 2 (B12)"), ("hue", "Hue")]
CROPS = ["Lucerne/Medics", "Canola", "Wheat", "Barley", "Small grain grazing"]
COLORS = {"Lucerne/Medics": "#009E73", "Canola": "#E69F00", "Wheat": "#0072B2",
          "Barley": "#56B4E9", "Small grain grazing": "#D55E00"}


def field_profiles(path, labels=None):
    cols = [f"{b}_{m}" for b, _ in PANELS for m in MONTHS]
    df = pd.read_parquet(path, columns=["fid"] + cols + ([] if labels is not None else ["crop_name"]))
    f = df.groupby("fid")[cols].mean()
    if labels is not None:
        f = f.join(labels.rename("crop_name"), how="inner")
    else:
        f = f.join(df.groupby("fid")["crop_name"].agg(lambda x: x.mode()[0]))
    return f


def main():
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    from compare_predictions import load_ground_truth
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    tr = field_profiles(MERGED_DL_PATH)
    te = field_profiles(MERGED_DL_TEST_PATH, labels=gt)
    print("fields: training", len(tr), " holdout", len(te))

    # summary table: median per crop/band/month, both regions
    rows = []
    for region, d in (("training", tr), ("holdout", te)):
        for c in CROPS:
            g = d[d.crop_name == c]
            for b, _ in PANELS:
                for m, p in zip(MONTHS, MPOS):
                    v = g[f"{b}_{m}"]
                    rows.append({"region": region, "crop": c, "band": b, "month": p, "n_fields": len(g),
                                 "median": v.median(), "q25": v.quantile(.25), "q75": v.quantile(.75)})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(HERE, "results", "phenology_medians.csv"), index=False)

    # separability: max |Cohen's d| over months, per band and crop pair (training region)
    sep = []
    for a, c in itertools.combinations(CROPS, 2):
        for b, _ in PANELS:
            ds = []
            for m in MONTHS:
                x, y = tr.loc[tr.crop_name == a, f"{b}_{m}"], tr.loc[tr.crop_name == c, f"{b}_{m}"]
                sp = np.sqrt((x.var() + y.var()) / 2)
                ds.append(abs(x.mean() - y.mean()) / sp if sp > 0 else 0)
            k = int(np.argmax(ds))
            sep.append({"pair": f"{a} vs {c}", "band": b, "max_abs_d": ds[k], "month": MPOS[k]})
    sep = pd.DataFrame(sep)
    best = sep.loc[sep.groupby("pair")["max_abs_d"].idxmax()].sort_values("max_abs_d", ascending=False)
    sep.round(3).to_csv(os.path.join(HERE, "results", "separability_by_band.csv"), index=False)
    best.round(3).to_csv(os.path.join(HERE, "results", "separability_best_band.csv"), index=False)
    print(best.round(2).to_string(index=False))

    # figure
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (b, lab) in zip(axes.flat, PANELS):
        ax.axvspan(4.5, 6.5, color="0.92", zorder=0)
        for c in CROPS:
            for region, d, ls in (("training", tr, "-"), ("holdout", te, "--")):
                g = d[d.crop_name == c][[f"{b}_{m}" for m in MONTHS]]
                med = g.median().values
                ax.plot(MPOS, med, ls=ls, color=COLORS[c], lw=1.8 if ls == "-" else 1.3, marker="o" if ls == "-" else None, ms=3)
                if region == "training":
                    ax.fill_between(MPOS, g.quantile(.25).values, g.quantile(.75).values, color=COLORS[c], alpha=0.10, lw=0)
        ax.set_title(lab, fontsize=11)
        ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
    axes[0, 0].text(5.5, axes[0, 0].get_ylim()[1], "May–Jun\nexcluded", ha="center", va="top", fontsize=7.5, color="0.4")
    handles = [plt.Line2D([], [], color=COLORS[c], lw=2, label=c) for c in CROPS]
    handles += [plt.Line2D([], [], color="0.3", ls="-", lw=1.8, label="Training region (median, IQR)"),
                plt.Line2D([], [], color="0.3", ls="--", lw=1.3, label="Holdout tile (median)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"phenology_curves.{ext}"), dpi=300)
    print("wrote figures/phenology_curves.pdf")


if __name__ == "__main__":
    main()
