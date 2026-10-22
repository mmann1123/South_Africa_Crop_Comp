"""Imputation-free check with uncertainty: every model's PUBLISHED holdout predictions scored on
the fields whose September is observed and on the fields whose September is missing, with
percentile field-bootstrap 95% CIs (B = 10,000, seed 42), each model's rank within each subset,
and the Spearman correlation between the two subset rankings and the all-field ranking.

A field counts as September-missing if any of its pixels lacks B2_September (44.0% of fields).

Writes experiments/missingness/results/subset_bootstrap.csv. Run in the deep_field env:
  python experiments/missingness/subset_bootstrap.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "holdout_fill_check"))
sys.path.insert(0, os.path.join(REPO, "out_of_sample"))
sys.path.insert(0, os.path.join(REPO, "deep_learn", "src"))
import score as sc  # noqa: E402  (MODELS: display name -> published prediction file)
from config import MERGED_DL_TEST_PATH  # noqa: E402

B, SEED = 10_000, 42

if __name__ == "__main__":
    gt = sc.load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    cl = sorted(gt.unique())
    ci = {c: i for i, c in enumerate(cl)}
    K = len(cl)
    t = pq.read_table(MERGED_DL_TEST_PATH, columns=["fid", "B2_September"]).to_pandas()
    gap = t["B2_September"].isna().groupby(t["fid"]).any()
    subsets = {"all": gt.index, "sept_observed": gap.index[~gap], "sept_missing": gap.index[gap]}
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (pub_file, _) in sc.MODELS.items():
        pred = sc.read(os.path.join(sc.OOS, pub_file))
        r = {"model": name}
        for s, ids in subsets.items():
            idx = gt.index.intersection(pred.index).intersection(ids)
            yt, yp = gt.reindex(idx).map(ci).values, pred.reindex(idx).map(ci).values
            N = len(idx)
            boot = np.array([sc.metrics(sc.confusion(yt, yp, K, np.bincount(rng.integers(0, N, N), minlength=N).astype(float)))[0]
                             for _ in range(B)])
            r[f"{s}_n"] = N
            r[f"{s}_f1m"] = sc.metrics(sc.confusion(yt, yp, K))[0]
            r[f"{s}_lo"], r[f"{s}_hi"] = np.percentile(boot, [2.5, 97.5])
        rows.append(r)
        print(f"{name:32s} all {r['all_f1m']:.3f}  observed {r['sept_observed_f1m']:.3f} "
              f"[{r['sept_observed_lo']:.3f},{r['sept_observed_hi']:.3f}]  missing {r['sept_missing_f1m']:.3f} "
              f"[{r['sept_missing_lo']:.3f},{r['sept_missing_hi']:.3f}]", flush=True)
    df = pd.DataFrame(rows)
    for s in subsets:
        df[f"{s}_rank"] = df[f"{s}_f1m"].rank(ascending=False, method="min").astype(int)
    df.round(4).to_csv(os.path.join(HERE, "results", "subset_bootstrap.csv"), index=False)
    print(f"\nfields: observed {df['sept_observed_n'].iloc[0]}, missing {df['sept_missing_n'].iloc[0]}")
    for a, b in [("all", "sept_observed"), ("all", "sept_missing"), ("sept_observed", "sept_missing")]:
        print(f"Spearman {a} vs {b}: {spearmanr(df[f'{a}_f1m'], df[f'{b}_f1m'])[0]:.2f}")
