#!/usr/bin/env python
"""DENETHOR Sentinel-2 random forest (random_forest.py, Table 3: accuracy 0.59,
macro F1 0.42 on the 2019 test tile), re-run with the year effect separated
and density matched.

Features exactly as the authors' S2 transform: field-mean reflectance x 1e-4
of the 12 bands plus NDVI; an acquisition counts when its field-mean CLP is
below 40 (the literal `cld < 40`, on CLP's 0 to 255 scale); max, min, median,
mean and std over the counted acquisitions; and the positions of the first
acquisition (over all 144, counted or not) equal to that max and min, divided
by 144. RandomForestClassifier(random_state=0), sklearn defaults otherwise,
trained on all 18E-242N 2018 fields.

Conditions, each evaluated on 17E-243N with 2019 imagery (the benchmark) and
with 2018 imagery (the paper's ablation design; only 2019 labels are public,
so they are reused, and crop rotation makes them wrong for an unknown share
of fields: the 2018-imagery rows cannot estimate a year effect and serve only
for within-2018 contrasts such as B1 against B1ctrl):
  B0       as published.
  B1       test-time density match: in each 2018 test-tile field, clear
           acquisitions are dropped at random, month by month, down to that
           field's clear count in the same month of 2019 (10 draws).
  B1ctrl   the same number of clear acquisitions dropped per field, at random
           across months (10 draws).
  B1train  training-side match: each 2018 training field is thinned month by
           month to its own clear count in 2019 (CLP of 18E-242N 2019), the
           forest is retrained (10 draws), and tested on both test years.
  B1trainctrl  the same number of clear training acquisitions dropped per
           field, at random across months, forest retrained (10 draws).
  B0seeds  B0 with forest seeds 0 to 9, the seed-to-seed spread that the
           B1train contrast has to exceed.
  B2       B0 under the s2cloudless 40 percent rule (CLP < 102), used
           elsewhere in this audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402

PAPER = "denethor_rerun"
CROP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
B04, B08 = 3, 7  # BANDS order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
N_DRAWS = 10


def load(tile: str, year: int) -> dict:
    z = np.load(paths.paper_root(PAPER) / f"s2_{tile}_{year}.npz", allow_pickle=True)
    ts = pd.to_datetime(z["timestamps"])
    return {"X": z["X"], "cld": z["cld"], "fid": z["fid"], "y": np.array([CROP_IDS.index(int(c)) for c in z["crop_id"]]),
            "month": ts.month.values}


def features(X: np.ndarray, clear: np.ndarray) -> np.ndarray:
    """X (F, T, 12) raw DN means, clear (F, T) bool -> (F, 91), the authors' transform per field."""
    out = []
    for x, m in zip(X, clear):
        x = x * 1e-4
        n = (x[:, B08] - x[:, B04]) / (x[:, B08] + x[:, B04])
        x = np.hstack([x, n[:, None]])
        x_ = x[m, :]
        with np.errstate(all="ignore"):
            feat = [f(x_, axis=0) for f in (np.max, np.min, np.median, np.mean, np.std)]
        t_max = np.argmax(x == feat[0], axis=0) / x.shape[0]
        t_min = np.argmax(x == feat[1], axis=0) / x.shape[0]
        out.append(np.hstack([np.stack(feat).reshape(-1), t_min, t_max]))
    return np.nan_to_num(np.stack(out), nan=0.0, posinf=0.0, neginf=0.0)


def thin_by_month(clear: np.ndarray, month: np.ndarray, target: np.ndarray, rng) -> np.ndarray:
    """clear (F, T) bool; target (F, 12) clear counts to keep at most, per month."""
    out = clear.copy()
    for f in range(clear.shape[0]):
        for m in range(1, 13):
            idx = np.flatnonzero(clear[f] & (month == m))
            k = len(idx) - int(target[f, m - 1])
            if k > 0:
                out[f, rng.choice(idx, k, replace=False)] = False
    return out


def thin_random(clear: np.ndarray, n_drop: np.ndarray, rng) -> np.ndarray:
    out = clear.copy()
    for f in range(clear.shape[0]):
        idx = np.flatnonzero(clear[f])
        k = min(int(n_drop[f]), len(idx))
        if k > 0:
            out[f, rng.choice(idx, k, replace=False)] = False
    return out


def month_counts(clear: np.ndarray, month: np.ndarray) -> np.ndarray:
    return np.stack([(clear & (month == m)).sum(1) for m in range(1, 13)], axis=1)


def aligned(src: dict, ref: dict) -> np.ndarray:
    """Row index into ref for every fid of src (same tile, other year)."""
    pos = {f: i for i, f in enumerate(ref["fid"])}
    return np.array([pos[f] for f in src["fid"]])


def score(y, p) -> dict:
    return {"accuracy": skm.accuracy_score(y, p), "f1_macro": skm.f1_score(y, p, average="macro", labels=np.arange(9), zero_division=0),
            "kappa": skm.cohen_kappa_score(y, p)}


def main() -> None:
    with RunLog("04_rf", {"n_draws": N_DRAWS}, paper=PAPER) as log:
        tr, te19, te18 = load("18E-242N", 2018), load("17E-243N", 2019), load("17E-243N", 2018)
        tr19 = load("18E-242N", 2019)  # CLP only
        rows, rng = [], np.random.default_rng(42)
        for rule, thr in (("literal_cld_lt_40", 40), ("s2cloudless_40pct", 102)):
            ctr, c19, c18 = tr["cld"] < thr, te19["cld"] < thr, te18["cld"] < thr
            clf = RandomForestClassifier(random_state=0, n_jobs=-1).fit(features(tr["X"], ctr), tr["y"])
            for test, X, cl, d in (("test_tile_2019", te19["X"], c19, te19), ("test_tile_2018", te18["X"], c18, te18)):
                rows.append({"rule": rule, "condition": "B0", "test": test, "draw": -1, **score(d["y"], clf.predict(features(X, cl))),
                             "median_clear": float(np.median(cl.sum(1)))})
            if rule != "literal_cld_lt_40":
                continue
            i19 = aligned(te18, te19)
            if not np.array_equal(te18["y"], te19["y"][i19]):
                raise ValueError("test-tile labels differ between the 2018 and 2019 series")
            target = month_counts(c19, te19["month"])[i19]
            for k in range(N_DRAWS):
                th = thin_by_month(c18, te18["month"], target, rng)
                rows.append({"rule": rule, "condition": "B1", "test": "test_tile_2018", "draw": k,
                             **score(te18["y"], clf.predict(features(te18["X"], th))), "median_clear": float(np.median(th.sum(1)))})
                n_drop = c18.sum(1) - th.sum(1)
                tc = thin_random(c18, n_drop, rng)
                rows.append({"rule": rule, "condition": "B1ctrl", "test": "test_tile_2018", "draw": k,
                             **score(te18["y"], clf.predict(features(te18["X"], tc))), "median_clear": float(np.median(tc.sum(1)))})
            j = aligned(tr, tr19)
            ttarget = month_counts(tr19["cld"] < thr, tr19["month"])[j]
            for k in range(N_DRAWS):
                th = thin_by_month(ctr, tr["month"], ttarget, rng)
                c2 = RandomForestClassifier(random_state=0, n_jobs=-1).fit(features(tr["X"], th), tr["y"])
                tc = thin_random(ctr, ctr.sum(1) - th.sum(1), rng)
                c3 = RandomForestClassifier(random_state=0, n_jobs=-1).fit(features(tr["X"], tc), tr["y"])
                for test, X, cl, d in (("test_tile_2019", te19["X"], c19, te19), ("test_tile_2018", te18["X"], c18, te18)):
                    rows.append({"rule": rule, "condition": "B1train", "test": test, "draw": k, **score(d["y"], c2.predict(features(X, cl))),
                                 "median_clear": float(np.median(th.sum(1)))})
                    rows.append({"rule": rule, "condition": "B1trainctrl", "test": test, "draw": k, **score(d["y"], c3.predict(features(X, cl))),
                                 "median_clear": float(np.median(tc.sum(1)))})
            Ftr = features(tr["X"], ctr)
            for k in range(N_DRAWS):
                c4 = RandomForestClassifier(random_state=k, n_jobs=-1).fit(Ftr, tr["y"])
                for test, X, cl, d in (("test_tile_2019", te19["X"], c19, te19), ("test_tile_2018", te18["X"], c18, te18)):
                    rows.append({"rule": rule, "condition": "B0seeds", "test": test, "draw": k, **score(d["y"], c4.predict(features(X, cl))),
                                 "median_clear": float(np.median(ctr.sum(1)))})
        df = pd.DataFrame(rows)
        write_csv(df, paths.results_dir(PAPER) / "rf_conditions.csv", "04_rf", ["s2_<tile>_<year>.npz from 03_s2_field_series.py"])
        summ = df.groupby(["rule", "condition", "test"]).agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                                                            f1_macro=("f1_macro", "mean"), kappa=("kappa", "mean"),
                                                            median_clear=("median_clear", "mean"), draws=("draw", "count")).reset_index()
        write_csv(summ, paths.results_dir(PAPER) / "rf_summary.csv", "04_rf", ["results/rf_conditions.csv"])
        print(summ.round(4).to_string())
        log.count("rows", len(df))


if __name__ == "__main__":
    main()
