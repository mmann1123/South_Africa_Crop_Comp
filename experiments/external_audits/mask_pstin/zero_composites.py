"""Count zero-filled 20-day composites in the Mask-PSTIN Auvergne 2021 release.

Usage: python -I zero_composites.py <path to a clone of github.com/ZhiwenCai-RS/Mask-PSTIN>

The released arrays are uint16 (n, 19*5, 5, 5): 19 composites of five bands over a
5 by 5 patch. The loader sets NaN to 0 and the arrays cannot hold NaN, so a composite
with no valid Cloud Score+ observation is stored as all five bands equal to zero.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Others", "Wheat", "Corn", "Rye", "Barley", "Sunflower", "Rapeseed"]
START = date(2020, 11, 1)  # paper section 3.2, region A: 2020-11-01 to 2021-11-01
OUT = Path(__file__).resolve().parent / "results"


def load(root, split):
    d = np.load(root / "data" / split / "data.npy", allow_pickle=False)
    lbl = np.load(root / "data" / split / "lbl.npy", allow_pickle=False).astype(int).ravel()
    n, tc, h, w = d.shape
    return d.reshape(n, tc // 5, 5, h, w), lbl


def main(root):
    step_rows, class_rows, split_rows = [], [], []
    for split in ["train", "val", "test"]:
        x, lbl = load(root, split)
        centre_zero = (x[:, :, :, 2, 2] == 0).all(axis=2)
        patch_zero = (x == 0).all(axis=(2, 3, 4))
        per_sample = centre_zero.sum(axis=1)
        split_rows.append({
            "split": split, "samples": len(lbl), "steps": x.shape[1],
            "centre_zero_share": centre_zero.mean(), "patch_zero_share": patch_zero.mean(),
            "samples_with_any_zero_step": (per_sample > 0).mean(),
            "mean_zero_steps_per_sample": per_sample.mean(), "max_zero_steps": int(per_sample.max()),
        })
        for t in range(x.shape[1]):
            first = START + timedelta(days=20 * t)
            step_rows.append({"split": split, "step": t + 1, "first_day": first.isoformat(),
                              "centre_zero_share": centre_zero[:, t].mean()})
        for c in np.unique(lbl):
            class_rows.append({"split": split, "class": CLASSES[c], "samples": int((lbl == c).sum()),
                               "centre_zero_share": centre_zero[lbl == c].mean()})
    OUT.mkdir(exist_ok=True)
    for name, rows in [("split_summary", split_rows), ("by_step", step_rows), ("by_class", class_rows)]:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
