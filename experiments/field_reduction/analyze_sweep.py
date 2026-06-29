"""Summarize + plot the subsample-seed variance sweep (results/seed_sweep_var.csv).

For each (model, fraction) computes mean / std / min / max of OOS macro-F1 across the
subsample draws, then plots the single-seed mean +/- band vs training fraction. Overlays
the published single-draw (seed-42) 5-seed values for reference, so we can see whether the
0.50 dip is a draw artifact (band covers it / seed-42 sits low) or a real effect.

Usage: python analyze_sweep.py [--csv results/seed_sweep_var.csv] [--out results/seed_sweep_band.png]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Published 5-seed single-draw (seed 42) values from field_reduction_results.csv, for reference.
PUBLISHED = {
    "ltae_sparse_pixel": {1.00: 0.599, 0.75: 0.614, 0.50: 0.558, 0.25: 0.581},
    "ltae_pixel":        {1.00: 0.579, 0.75: 0.572, 0.50: 0.526, 0.25: 0.570},
    "tabnet_pixel":      {1.00: 0.603, 0.75: 0.620, 0.50: 0.620, 0.25: 0.586},
    "xgboost_field":     {1.00: 0.565, 0.75: 0.575, 0.50: 0.537, 0.25: 0.533},
    "base_lr_pixel":     {1.00: 0.563, 0.75: 0.593, 0.50: 0.597, 0.25: 0.581},
}
LABEL = {"ltae_sparse_pixel": "L-TAE-S", "ltae_pixel": "L-TAE", "tabnet_pixel": "TabNet",
         "xgboost_field": "XGBoost", "base_lr_pixel": "Logistic Reg."}
COLOR = {"ltae_sparse_pixel": "C0", "ltae_pixel": "C1", "tabnet_pixel": "C2",
         "xgboost_field": "C3", "base_lr_pixel": "C4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/seed_sweep_var.csv")
    ap.add_argument("--out", default="results/seed_sweep_band.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    g = (df.groupby(["model", "fraction"])["oos_f1_macro"]
           .agg(["mean", "std", "min", "max", "count"]).reset_index())
    g["std"] = g["std"].fillna(0.0)
    print("Per (model, fraction) across subsample draws:")
    print(g.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in [m for m in LABEL if m in g["model"].values]:
        sub = g[g["model"] == model].sort_values("fraction")
        x, m, s = sub["fraction"].values, sub["mean"].values, sub["std"].values
        c = COLOR[model]
        # capped error bars (bound ticks) at +/- 1 s.d., plus a faint band for continuity
        ax.errorbar(x, m, yerr=s, fmt="-o", color=c, capsize=5, capthick=1.4, elinewidth=1.4,
                    markersize=5, label=f"{LABEL[model]} (single-seed mean +/- 1 s.d., n={int(sub['count'].max())} draws)")
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.10)
        # published 5-seed single-draw reference (dashed, same color)
        pub = PUBLISHED.get(model, {})
        px = sorted(f for f in pub if f in set(x))
        if px:
            ax.plot(px, [pub[f] for f in px], "--^", color=c, alpha=0.50,
                    label=f"{LABEL[model]} (published 5-seed, seed-42 draw)")

    ax.set_xlabel("Fraction of training fields retained")
    ax.set_ylabel("Spatial-transfer macro-F1")
    ax.set_title("Data efficiency: single-seed mean ± 1 s.d. over subsample draws")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
