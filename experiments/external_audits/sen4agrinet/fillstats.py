"""Per-patch fill table and composition-weighted split summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.external_audits.sen4agrinet import fillrule

WINDOWS = {"PAD_Apr-Sep": fillrule.PAD_MONTHS, "OAD_May-Nov": fillrule.OAD_MONTHS}
GROUP = ["scenario", "split", "country", "year", "tile"]


def patch_fill_table(sample: pd.DataFrame, stacks: pd.DataFrame) -> pd.DataFrame:
    """One row per sampled patch: observations, month presence, fill counts per window."""
    st = stacks.set_index("stem")
    rows = []
    for r in sample.itertuples(index=False):
        t = pd.DatetimeIndex(pd.to_datetime(st.loc[r.stem, "times"].split(";")))
        pres = fillrule.present_months(t, int(r.year))
        d = {c: getattr(r, c) for c in GROUP + ["stem"]}
        d["n_obs"] = int(len(t))
        d["n_obs_in_year"] = int((fillrule.month_bins(t, int(r.year)) > 0).sum())
        for m in range(12):
            d[f"present_{m + 1:02d}"] = int(pres[m])
        for name, months in WINDOWS.items():
            s = fillrule.window_stats(pres, months)
            d[f"{name}_filled"] = s["filled"]
            d[f"{name}_interpolated"] = s["interpolated"]
            d[f"{name}_extrapolated"] = s["extrapolated"]
        rows.append(d)
    return pd.DataFrame(rows)


def _weights(pf: pd.DataFrame, comp: pd.DataFrame) -> pd.Series:
    """Patch weight so each tile-year group counts in proportion to its number of
    patches in the published split (group size over sampled size)."""
    n_pop = comp.set_index(GROUP)["n_patches"]
    n_s = pf.groupby(GROUP).size()
    w = (n_pop / n_s).rename("w")
    return pf.join(w, on=GROUP)["w"]


def split_summary(pf: pd.DataFrame, comp: pd.DataFrame, boot: int = 300, seed: int = 42) -> pd.DataFrame:
    pf = pf.copy()
    pf["w"] = _weights(pf, comp)
    rows = []
    rng = np.random.default_rng(seed)
    for (scen, split), g in pf.groupby(["scenario", "split"]):
        for name, months in WINDOWS.items():
            nm = len(months)
            row = {"scenario": scen, "split": split, "window": name, "country": "+".join(sorted(g["country"].unique())), "years": "+".join(map(str, sorted(g["year"].unique()))), "n_patches_split": int(comp[(comp.scenario == scen) & (comp.split == split)]["n_patches"].sum()), "n_patches_sampled": len(g)}

            def stat(df, col):
                return float((df[col] / nm * df["w"]).sum() / df["w"].sum())

            def any_fill(df):
                return float(((df[f"{name}_filled"] > 0) * df["w"]).sum() / df["w"].sum())

            row["share_filled"] = stat(g, f"{name}_filled")
            row["share_interpolated"] = stat(g, f"{name}_interpolated")
            row["share_extrapolated"] = stat(g, f"{name}_extrapolated")
            row["share_patches_any_fill"] = any_fill(g)
            row["mean_obs_per_patch_year"] = float((g["n_obs"] * g["w"]).sum() / g["w"].sum())
            # stratified bootstrap over sampled patches inside each tile-year group
            reps = []
            groups = [d for _, d in g.groupby(GROUP)]
            for _ in range(boot):
                b = pd.concat([d.iloc[rng.integers(0, len(d), len(d))] for d in groups])
                reps.append(stat(b, f"{name}_filled"))
            row["share_filled_ci_low"], row["share_filled_ci_high"] = np.percentile(reps, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def month_missing(pf: pd.DataFrame, comp: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Weighted share of patches with no acquisition in each calendar month."""
    pf = pf.copy()
    pf["w"] = _weights(pf, comp)
    out = []
    for key, g in pf.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(by, key))
        for m in range(1, 13):
            miss = 1 - g[f"present_{m:02d}"]
            out.append({**base, "month": m, "share_missing": float((miss * g["w"]).sum() / g["w"].sum()), "n_patches_sampled": len(g)})
    return pd.DataFrame(out)


def obs_by_country_year(pf: pd.DataFrame) -> pd.DataFrame:
    u = pf.drop_duplicates("stem")
    g = u.groupby(["country", "year"])["n_obs"]
    return g.agg(n_patches_sampled="size", mean="mean", median="median", p10=lambda s: s.quantile(0.1), p90=lambda s: s.quantile(0.9), min="min", max="max").reset_index()


def combine_tile_years(mm: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Country-year monthly missing share from tile-year rows (columns country,
    year, tile, month, share_missing), weighting each tile-year by its patch count
    in `pop` (columns country, year, tile, n_patches)."""
    d = mm.merge(pop, on=["country", "year", "tile"], how="left")
    if d["n_patches"].isna().any():
        raise ValueError("tile-year without a population weight")
    d["wx"] = d["share_missing"] * d["n_patches"]
    g = d.groupby(["country", "year", "month"]).agg(wx=("wx", "sum"), w=("n_patches", "sum")).reset_index()
    g["share_missing"] = g["wx"] / g["w"]
    return g[["country", "year", "month", "share_missing"]]
