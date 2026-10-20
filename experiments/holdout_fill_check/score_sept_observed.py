"""Score every model's PUBLISHED holdout predictions on the fields whose September values
are fully observed (no pixel with missing September), so no imputation is involved.

Run in the deep_field env:  python experiments/holdout_fill_check/score_sept_observed.py
"""
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as sc  # noqa: E402  (MODELS, read, OOS, metrics helpers)
from config import MERGED_DL_TEST_PATH  # noqa: E402

t = pq.read_table(MERGED_DL_TEST_PATH, columns=["fid", "B2_September"]).to_pandas()
miss = t["B2_September"].isna().groupby(t["fid"]).any()
observed = miss.index[~miss]
gt = sc.load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
classes = sorted(gt.unique())
cidx = {c: i for i, c in enumerate(classes)}
K = len(classes)
rows = []
for name, (pub_file, _) in sc.MODELS.items():
    pub = sc.read(os.path.join(sc.OOS, pub_file))
    for subset, ids in [("all", gt.index), ("sept_observed", observed), ("sept_missing", miss.index[miss])]:
        idx = gt.index.intersection(pub.index).intersection(ids)
        f1, kap, acc = sc.metrics(sc.confusion(gt.reindex(idx).map(cidx).values,
                                               pub.reindex(idx).map(cidx).values, K))
        rows.append({"model": name, "subset": subset, "n_fields": len(idx), "f1m": f1, "acc": acc})
df = pd.DataFrame(rows).pivot(index="model", columns="subset", values="f1m")
n = pd.DataFrame(rows).groupby("subset")["n_fields"].max().to_dict()
df = df[["all", "sept_observed", "sept_missing"]].sort_values("sept_observed", ascending=False)
df["rank_all"] = df["all"].rank(ascending=False, method="min").astype(int)
df["rank_observed"] = df["sept_observed"].rank(ascending=False, method="min").astype(int)
df.round(4).to_csv(os.path.join(HERE, "results", "sept_observed_subset.csv"))
print("fields:", n)
print(df.round(3).to_string())
