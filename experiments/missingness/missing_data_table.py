"""Per-month missing-data rates, training tiles vs holdout tile, at pixel and field level.

For each band x month: share of pixels missing, share of fields with any pixel missing, and
share of fields with every pixel missing. Computed from the pixel files every raw-input model
reads (data/merged_dl_train.parquet, data/merged_dl_test.parquet).

Writes experiments/missingness/results/missing_data_by_month.csv (long) and prints a
month x region summary (B2), and how often the six bands of a month are only partly missing.
Run in the deep_field env: python experiments/missingness/missing_data_table.py
"""
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
from config import MERGED_DL_PATH, MERGED_DL_TEST_PATH  # noqa: E402
from missing import MONTHS, _band_month  # noqa: E402

BANDS = ["B2", "B6", "B11", "B12", "EVI", "hue"]


def rates(path, region):
    names = pq.ParquetFile(path).schema.names
    cols = [c for c in names if _band_month(c) is not None]
    t = pq.read_table(path, columns=["fid"] + cols).to_pandas()
    miss = t[cols].isna()
    g = miss.groupby(t["fid"])
    any_f, all_f = g.any(), g.all()
    rows = []
    for c in cols:
        band, m = _band_month(c)
        rows.append({"region": region, "band": band, "month": MONTHS[m], "month_idx": m,
                     "pixels": len(t), "fields": t["fid"].nunique(),
                     "pct_pixels_missing": 100 * miss[c].mean(),
                     "pct_fields_any_missing": 100 * any_f[c].mean(),
                     "pct_fields_all_missing": 100 * all_f[c].mean()})
    # how often the bands of one month disagree (some bands missing, others observed)
    for m in sorted({_band_month(c)[1] for c in cols}):
        mc = [c for c in cols if _band_month(c)[1] == m]
        part = miss[mc].any(axis=1) & ~miss[mc].all(axis=1)
        for r in rows:
            if r["month_idx"] == m:
                r["pct_pixels_some_bands_missing"] = 100 * part.mean()
                r["pct_pixels_all_bands_missing"] = 100 * miss[mc].all(axis=1).mean()
    return rows


if __name__ == "__main__":
    out = pd.DataFrame(rates(MERGED_DL_PATH, "training tiles") + rates(MERGED_DL_TEST_PATH, "holdout tile"))
    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    out.round(3).to_csv(os.path.join(res, "missing_data_by_month.csv"), index=False)
    b2 = out[out.band == "B2"].sort_values("month_idx")
    wide = b2.pivot_table(index=["month_idx", "month"], columns="region",
                          values=["pct_pixels_missing", "pct_fields_any_missing", "pct_fields_all_missing"])
    pd.set_option("display.width", 200)
    print(wide.round(1).droplevel(0).to_string())
    sep = out[(out.month == "September")].groupby("region")[
        ["pct_pixels_all_bands_missing", "pct_pixels_some_bands_missing"]].first()
    print("\nSeptember, % of pixels:"); print(sep.round(2).to_string())
    print("\nSeptember, % pixels missing by band:")
    print(out[out.month == "September"].pivot(index="band", columns="region", values="pct_pixels_missing").round(2).to_string())
