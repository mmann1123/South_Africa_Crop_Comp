"""Per-input table and aggregation for the Kerner audit (pure, tested)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.external_audits.kerner import kernerlib as kl

CLASSES = ("corn", "soy", "other")


def decode_states(strings) -> np.ndarray:
    """Digit strings (one per pixel, equal length) -> (N, A) int8 array."""
    arr = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8) - ord("0")
    return arr.reshape(len(strings), -1).astype(np.int8)


def input_table(sample: pd.DataFrame, states: np.ndarray, acq_doy: np.ndarray, year: int, offset: int) -> pd.DataFrame:
    """One row per pixel x stage x basis for one year.

    sample: that year's rows (same order as states). Growth-stage targets are the
    released phenology index plus `offset`; fixed targets are the paper's DOYs.
    """
    rows = []
    for basis, stages in (("fixed", kl.FIXED_STAGES), ("growth", kl.GROWTH_STAGES)):
        for k, stage in enumerate(stages):
            if basis == "fixed":
                target = np.full(len(sample), kl.FIXED_DOYS[year][k])
            else:
                target = sample[stage].to_numpy() + offset
            n_clear, n_flag, exact = kl.window_counts(states, acq_doy, target)
            rows.append(
                pd.DataFrame(
                    {"year": year, "pid": sample["pid"].to_numpy(), "class": sample["class"].to_numpy(), "basis": basis, "stage": stage, "stage_idx": k, "target_doy": target, "n_clear": n_clear, "n_flagged": n_flag, "exact": exact}
                )
            )
    return pd.concat(rows, ignore_index=True)


def _row(g: pd.DataFrame, n_boot: int, seed: int) -> dict:
    pids = g["pid"].unique()
    zero = (g["n_clear"] == 0)
    flagged = g["exact"] == kl.FLAGGED
    mat = lambda s: s.groupby(g["pid"]).apply(lambda x: x.to_numpy()).reindex(pids)  # noqa: E731
    zm = np.vstack([v.astype(float) for v in mat(zero).to_numpy()])
    fm = np.vstack([v.astype(float) for v in mat(flagged).to_numpy()])
    z, zlo, zhi = kl.bootstrap_mean_ci(zm, n_boot, seed)
    f, flo, fhi = kl.bootstrap_mean_ci(fm, n_boot, seed)
    return {
        "n_pixels": int(len(pids)),
        "n_inputs": int(len(g)),
        "share_zero_clear": z,
        "share_zero_clear_lo": zlo,
        "share_zero_clear_hi": zhi,
        "share_le1_clear": float((g["n_clear"] <= 1).mean()),
        "median_n_clear": float(g["n_clear"].median()),
        "mean_n_clear": float(g["n_clear"].mean()),
        "share_exact_clear": float((g["exact"] == kl.CLEAR).mean()),
        "share_exact_cloudy": f,
        "share_exact_cloudy_lo": flo,
        "share_exact_cloudy_hi": fhi,
        "share_exact_no_acq": float((g["exact"] == kl.NODATA).mean()),
        "share_exact_not_clear": float((g["exact"] != kl.CLEAR).mean()),
    }


def aggregate(inputs: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """year x basis x stage ('all' pools the three) x class ('all' pools the three)."""
    out = []
    for (year, basis), gyb in inputs.groupby(["year", "basis"]):
        for stage in ["all", *sorted(gyb["stage"].unique(), key=lambda s: gyb[gyb["stage"] == s]["stage_idx"].iloc[0])]:
            gs = gyb if stage == "all" else gyb[gyb["stage"] == stage]
            for cls in ("all", *CLASSES):
                g = gs if cls == "all" else gs[gs["class"] == cls]
                if g.empty:
                    continue
                r = _row(g, n_boot, seed)
                r.update({"year": year, "basis": basis, "stage": stage, "class": cls, "median_target_doy": float(g["target_doy"].median())})
                out.append(r)
    cols = ["year", "basis", "stage", "class", "median_target_doy"]
    df = pd.DataFrame(out)
    return df[cols + [c for c in df.columns if c not in cols]]


def diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """mean(a) - mean(b) for per-pixel indicator vectors, bootstrap over pixels in each group."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ba = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    bb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = ba - bb
    return float(a.mean() - b.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def test_minus_train(inputs: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """2019 minus pooled 2017+2018 (pixels weighted equally), for the share with zero clear, flagged on the exact day, and not clear on the exact day (flagged or unobserved), per basis x stage x class."""
    rows = []
    for basis in sorted(inputs["basis"].unique()):
        gb = inputs[inputs["basis"] == basis]
        for stage in ["all", *gb.sort_values("stage_idx")["stage"].unique()]:
            gs = gb if stage == "all" else gb[gb["stage"] == stage]
            for cls in ("all", *CLASSES):
                g = gs if cls == "all" else gs[gs["class"] == cls]
                row = {"basis": basis, "stage": stage, "class": cls}
                for name, ind in (("zero_clear", g["n_clear"] == 0), ("exact_cloudy", g["exact"] == kl.FLAGGED), ("exact_not_clear", g["exact"] != kl.CLEAR)):
                    per = ind.astype(float).groupby([g["year"], g["pid"]]).mean()
                    tr = per[per.index.get_level_values(0).isin([2017, 2018])].to_numpy()
                    te = per[per.index.get_level_values(0) == 2019].to_numpy()
                    d, lo, hi = diff_ci(te, tr, n_boot, seed)
                    row.update({f"{name}_train": float(tr.mean()), f"{name}_test2019": float(te.mean()), f"{name}_diff": d, f"{name}_diff_lo": lo, f"{name}_diff_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)
