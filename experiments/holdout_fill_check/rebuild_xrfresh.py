"""Rebuild the holdout xr_fresh features from the pixel time series.

The published holdout features (features/X_testing_{band}_34S_20E_259N.parquet) were made
by light_learn/2_SA_xr_fresh_extraction.py from the monthly rasters. This recomputes the
same 29 features per band with the same xr_fresh 0.2.2 functions, on the same 11 monthly
values per pixel (Jan-Apr, May, Jul-Dec; May is missing everywhere), for two inputs:

  orig      the raw holdout series (September missing for ~41% of pixels)
  trainmed  September filled with training-tile medians (make_filled_holdout.py)

Faithfulness check: the 'orig' rebuild must equal the published feature file for every
pixel and feature (relative difference <= 1e-4), allowing at most 0.001% of pixels to
differ on a feature: float32 summation order can flip a value lying exactly at the
series mean (seen on 4 of 3.4M pixels for two count features). Otherwise it stops.

Note: in the original extraction, function_mapping.get("autocorr") returns None, so the
three "autocorr_lag_k" features silently reuse the previous function,
absolute_sum_of_changes. The published files confirm this; the rebuild reproduces it.

Run in the xr_fresh env:
  ~/miniconda3/envs/xr_fresh/bin/python experiments/holdout_fill_check/rebuild_xrfresh.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import jax.numpy as jnp
import xr_fresh.feature_calculator_series as fc

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/features"
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUTS = {"orig": os.path.join(REPO, "data", "merged_dl_test.parquet"),
          "trainmed": os.path.join(HERE, "data", "merged_dl_test_trainmedian.parquet")}
# Fill-value sweep variants (experiments/missingness/make_filled_variants.py); opt in with --variants.
for _m in ["zero", "self_median", "train_mean", "train_median", "interp"]:
    INPUTS[f"fill_{_m}"] = os.path.join(HERE, "data", f"merged_dl_test_fill_{_m}.parquet")
DEFAULT_VARIANTS = ["orig", "trainmed"]
BANDS = ["B12", "B11", "B2", "B6", "EVI", "hue"]
MONTHS = ["January", "February", "March", "April", "May", "July", "August", "September",
          "October", "November", "December"]
DOY = [int(datetime(2017, m, 1).strftime("%j")) for m in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
META = ["id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN"]
CHUNK = 200_000

ASC = fc.absolute_sum_of_changes()
SPEC = {  # published column suffix -> function instance (order as in the published files)
    "abs_energy": fc.abs_energy(),
    "absolute_sum_of_changes": ASC,
    "autocorr_lag_1": ASC, "autocorr_lag_2": ASC, "autocorr_lag_3": ASC,  # original bug, see docstring
    "count_above_mean": fc.count_above_mean(),
    "count_below_mean": fc.count_below_mean(),
    "doy_of_maximum_dates": fc.doy_of_maximum(dates=DOY),
    "doy_of_minimum_dates": fc.doy_of_minimum(dates=DOY),
    "kurtosis": fc.kurtosis(),
    "large_standard_deviation": fc.large_standard_deviation(),
    "maximum": fc.maximum(),
    "mean": fc.mean(),
    "mean_abs_change": fc.mean_abs_change(),
    "mean_change": fc.mean_change(),
    "mean_second_derivative_central": fc.mean_second_derivative_central(),
    "median": fc.median(),
    "minimum": fc.minimum(),
    "quantile_q_0.05": fc.quantile(q=0.05),
    "quantile_q_0.95": fc.quantile(q=0.95),
    "ratio_beyond_r_sigma_r_1": fc.ratio_beyond_r_sigma(r=1),
    "ratio_beyond_r_sigma_r_2": fc.ratio_beyond_r_sigma(r=2),
    "skewness": fc.skewness(),
    "standard_deviation": fc.standard_deviation(),
    "sum": fc.sum(),
    "symmetry_looking": fc.symmetry_looking(),
    "ts_complexity_cid_ce": fc.ts_complexity_cid_ce(),
    "variance": fc.variance(),
    "variance_larger_than_standard_deviation": fc.variance_larger_than_standard_deviation(),
}


def features(series):
    """series: (N, T) float -> dict of (N,) arrays. Arrays passed as (T, 1, N, 1) like a
    one-band geowombat series block."""
    out = {k: np.empty(len(series), dtype=np.float32) for k in SPEC}
    for i in range(0, len(series), CHUNK):
        a = jnp.asarray(series[i:i + CHUNK].T[:, None, :, None])
        for k, f in SPEC.items():
            out[k][i:i + CHUNK] = np.asarray(f.calculate(a)).reshape(-1)
    return out


def main(variants=DEFAULT_VARIANTS):
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    report = []
    for band in BANDS:
        pub = pq.read_table(os.path.join(FEAT, f"X_testing_{band}_34S_20E_259N.parquet")).to_pandas()
        cols = [f"{band}_{k}" for k in SPEC]
        assert list(pub.columns) == META + cols, f"{band}: unexpected published columns"
        for variant in variants:
            path = INPUTS[variant]
            src = pq.read_table(path, columns=META + [f"{band}_{m}" for m in MONTHS]).to_pandas()
            assert (src["id"].values == pub["id"].values).all() and (src["point"].values == pub["point"].values).all()
            feats = features(src[[f"{band}_{m}" for m in MONTHS]].to_numpy(np.float32))  # rasters were float32
            df = src[META].copy()
            for k in SPEC:
                df[f"{band}_{k}"] = feats[k].astype(pub[f"{band}_{k}"].dtype)
            if variant == "orig":
                for c in cols:
                    a, b = df[c].to_numpy(np.float64), pub[c].to_numpy(np.float64)
                    nan_same = bool((np.isnan(a) == np.isnan(b)).all())
                    m = ~np.isnan(b)
                    d = float(np.max(np.abs(a[m] - b[m]) / np.maximum(1.0, np.abs(b[m])))) if m.any() else 0.0
                    rel = np.zeros_like(a)
                    rel[m] = np.abs(a[m] - b[m]) / np.maximum(1.0, np.abs(b[m]))
                    frac = float((rel > 1e-4).mean())
                    report.append({"band": band, "feature": c, "nan_pattern_equal": nan_same,
                                   "max_rel_diff": d, "share_pixels_differing": frac})
                bad = [r for r in report if r["band"] == band
                       and (not r["nan_pattern_equal"] or r["share_pixels_differing"] > 1e-5)]
                pd.DataFrame(report).to_csv(os.path.join(HERE, "results", "xrfresh_rebuild_check.csv"), index=False)
                if bad:
                    print(pd.DataFrame(bad).to_string())
                    sys.exit(f"{band}: rebuild does not reproduce the published features; stopped.")
                print(f"{band}: rebuild matches published features (largest share of pixels differing: "
                      f"{max(r['share_pixels_differing'] for r in report if r['band'] == band):.1e})", flush=True)
            df.to_parquet(os.path.join(HERE, "data", f"xrfresh_{variant}_{band}.parquet"), index=False)
            print(f"  wrote {variant} {band}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(INPUTS))
    main(ap.parse_args().variants)
