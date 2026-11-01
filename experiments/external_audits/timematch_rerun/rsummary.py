"""Pure pandas summaries of the per-run condition table."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("macro_f1", "accuracy")


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (direction, fold, cloud_source, threshold, condition): metric means over
    random draws (conditions e, e_shift) and the SD over draws."""
    g = df.groupby(["direction", "fold", "cloud_source", "threshold", "condition"], dropna=False)
    agg = g.agg(n_draws=("draw", "size"), **{m: (m, "mean") for m in METRICS}, **{f"{m}_draw_sd": (m, lambda s: s.std(ddof=0)) for m in METRICS},
                mean_dates_kept=("mean_dates_kept", "mean"), mean_removed=("mean_removed", "mean")).reset_index()
    return agg


def condition_summary(ft: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD (over folds) per direction, source, threshold, condition."""
    g = ft.groupby(["direction", "cloud_source", "threshold", "condition"], dropna=False)
    out = g.agg(n_folds=("fold", "nunique"), **{f"{m}_mean": (m, "mean") for m in METRICS},
                **{f"{m}_fold_sd": (m, lambda s: s.std(ddof=0)) for m in METRICS},
                **{f"{m}_draw_sd_mean": (f"{m}_draw_sd", "mean") for m in METRICS},
                mean_dates_kept=("mean_dates_kept", "mean"), mean_removed=("mean_removed", "mean")).reset_index()
    return out


def effects(ft: pd.DataFrame, upper: dict, timematch: dict | None = None) -> pd.DataFrame:
    """Cloud effects per fold, then mean and SD over folds.
    upper: {(direction, fold): macro_f1 of the target-trained model}; timematch: same for TimeMatch.
    effect_b = b - e, effect_d = d - e_shift. share_of_shifted_gap = effect_d / (upper - c);
    share_of_unshifted_gap = effect_b / (upper - a). Metrics on the 0-1 scale."""
    rows = []
    for (d, f), g in ft.groupby(["direction", "fold"]):
        a = g[g.condition == "a"]["macro_f1"].iloc[0]
        c = g[g.condition == "c"]["macro_f1"].iloc[0]
        a_oa = g[g.condition == "a"]["accuracy"].iloc[0]
        c_oa = g[g.condition == "c"]["accuracy"].iloc[0]
        for (src, thr), h in g[g.cloud_source != "none"].groupby(["cloud_source", "threshold"]):
            v = h.set_index("condition")
            ub = upper[(d, f)]
            row = {"direction": d, "fold": f, "cloud_source": src, "threshold": thr,
                   "a": a, "b": v.loc["b", "macro_f1"], "c": c, "d": v.loc["d", "macro_f1"],
                   "e": v.loc["e", "macro_f1"], "e_shift": v.loc["e_shift", "macro_f1"],
                   "e_draw_sd": v.loc["e", "macro_f1_draw_sd"], "e_shift_draw_sd": v.loc["e_shift", "macro_f1_draw_sd"],
                   "oa_a": a_oa, "oa_b": v.loc["b", "accuracy"], "oa_c": c_oa, "oa_d": v.loc["d", "accuracy"],
                   "oa_e": v.loc["e", "accuracy"], "oa_e_shift": v.loc["e_shift", "accuracy"], "upper": ub}
            row["effect_b"] = row["b"] - row["e"]
            row["effect_d"] = row["d"] - row["e_shift"]
            row["oa_effect_b"] = row["oa_b"] - row["oa_e"]
            row["oa_effect_d"] = row["oa_d"] - row["oa_e_shift"]
            row["share_of_unshifted_gap"] = row["effect_b"] / (ub - a) if ub > a else np.nan
            row["share_of_shifted_gap"] = row["effect_d"] / (ub - c) if ub > c else np.nan
            if timematch:
                tm = timematch[(d, f)]
                row["timematch"] = tm
                row["share_of_timematch_gap"] = row["effect_d"] / (ub - tm) if ub > tm else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def effects_summary(ef: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ef.columns if c not in ("direction", "fold", "cloud_source", "threshold")]
    g = ef.groupby(["direction", "cloud_source", "threshold"])
    mean = g[cols].mean().add_suffix("_mean")
    sd = g[cols].agg(lambda s: s.std(ddof=0)).add_suffix("_fold_sd")
    return pd.concat([mean, sd], axis=1).reset_index()
