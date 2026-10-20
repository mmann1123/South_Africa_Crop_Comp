"""Field-level holdout xr_fresh features (mean per field), as out_of_sample/combine_test_parquets.py
does, from the rebuilt per-band pixel features (rebuild_xrfresh.py). Also links the per-band pixel
files under the published names for voting_hybrid_oos.py. The 'orig' result must equal
data/combined_test_features.parquet. Run in any env with pandas/pyarrow.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
from config import COMBINED_TEST_FEATURES_PATH  # noqa: E402

BANDS = ["B12", "B11", "B2", "B6", "EVI", "hue"]
META = ["id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN"]
pub = pd.read_parquet(COMBINED_TEST_FEATURES_PATH)
for variant in ["orig", "trainmed"]:
    pixdir = os.path.join(HERE, "data", f"xtest_{variant}")
    os.makedirs(pixdir, exist_ok=True)
    field = None
    for b in BANDS:
        src = os.path.join(HERE, "data", f"xrfresh_{variant}_{b}.parquet")
        link = os.path.join(pixdir, f"X_testing_{b}_34S_20E_259N.parquet")
        if not os.path.exists(link):
            os.symlink(src, link)
        d = pd.read_parquet(src)
        f = d.groupby("fid")[[c for c in d.columns if c not in META]].mean().reset_index()
        field = f if field is None else field.merge(f, on="fid", how="inner")
    out = field[[c for c in pub.columns if c in field.columns]]
    if variant == "orig":
        p = pub.set_index("fid").loc[out["fid"]]
        cols = [c for c in out.columns if c != "fid"]
        missing = [c for c in pub.columns if c not in out.columns]
        diff = np.nanmax(np.abs(out[cols].to_numpy(float) - p[cols].to_numpy(float)) /
                         np.maximum(1.0, np.abs(p[cols].to_numpy(float))))
        print(f"orig vs published combined: fields {len(out)} vs {len(pub)}, "
              f"max rel diff {diff:.2e}, published columns not rebuilt: {missing}")
        out = pub.drop(columns=cols).merge(out, on="fid")[pub.columns]  # keep crop_name etc.
    else:
        extra = pub[[c for c in pub.columns if c not in out.columns or c == "fid"]]
        out = extra.merge(out, on="fid")[pub.columns]
    out.to_parquet(os.path.join(HERE, "data", f"combined_test_features_{variant}.parquet"), index=False)
    print("wrote", variant, out.shape)
