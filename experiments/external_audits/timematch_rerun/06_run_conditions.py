#!/usr/bin/env python
"""Step 2: evaluate the frozen source-only models on the target tile under
(a) as published, (b) cloudy dates removed per parcel, (c) the paper's calendar shift,
(d) both, (e) random removal of as many dates per parcel as (b) removed, (e') random
removal plus shift. Random draws are seeded; e and e' share the same draws (paired).
Cloud flags: Cloud Score+ cs_cdf < t over the parcel footprint (FR1 target, exact), or
the Cloud Score+ emulator / block-level Cloud Score+ (DK1 target, no polygons).
Run in env deep_field: python -I 06_run_conditions.py --direction DK1_to_FR1"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch_rerun import evalcore, rerunlib  # noqa: E402

PAPER = "timematch_rerun"
ROOT = evalcore.ROOT
RES = HERE / "results"
SHIFT = {"DK1_to_FR1": 32, "FR1_to_DK1": -35}  # target-to-source days, paper Fig. 4; positions + shift
THR_IDX = {50: 0, 60: 1, 70: 2}
SOURCES = {"DK1_to_FR1": ["csplus_parcel", "csplus_parcel+nodata"], "FR1_to_DK1": ["emulator", "block", "emulator+nodata"]}
PRIMARY_ONLY = ("block", "csplus_parcel+nodata", "emulator+nodata")  # secondary variants are run at threshold 0.6 only


def positions_for(tile_lab: str) -> np.ndarray:
    d = json.loads((ROOT / f"dates_{tile_lab}.json").read_text())
    start = dt.datetime(2017, 1, 1)
    return np.array([abs((dt.datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])) - start).days) for x in d])


def flags_for(direction, source, test_idx, thr, nodata):
    """(flags (P,T) bool, score (P,T) float) for the test parcels. A '+nodata' source also
    flags dates whose pixels are all zero (no S2 coverage; shipped as zero-filled data)."""
    if source.endswith("+nodata"):
        f, sc = flags_for(direction, source[: -len("+nodata")], test_idx, thr, nodata)
        return rerunlib.add_nodata(f, sc, nodata)
    t = THR_IDX[thr]
    if source == "csplus_parcel":
        files = sorted((ROOT / "ee_chunks").glob("fr1_chunk_*.npz"))
        idx = np.concatenate([np.load(f)["idx"] for f in files])
        frac = np.concatenate([np.load(f)["frac"] for f in files])[:, t, :]
        row = {int(i): k for k, i in enumerate(idx)}
        f = frac[[row[int(i)] for i in test_idx]]
        return rerunlib.cloud_flags(f, 0.5), f
    if source == "emulator":
        prob = np.load(ROOT / "emulator" / f"dk1_prob_t{thr}.npy")[test_idx]
        return prob >= 0.5, prob
    if source == "block":
        b = pd.read_csv(RES / "block_cloud_by_date.csv")
        b = b[b["tile"] == "DK1"]
        dates = json.loads((ROOT / "dates_DK1.json").read_text())
        mat = np.full((100, len(dates)), np.nan)
        for k, d in enumerate(dates):
            ds = f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}"
            g = b[b["date"] == ds]
            mat[g["block"].values, k] = g[f"frac_cloudy_t{thr}"].values
        meta = pd.read_csv(ROOT / "parcels_DK1.csv")
        blocks = meta["block"].values[test_idx]
        f = mat[blocks]
        return rerunlib.block_flags_to_parcels(blocks, mat, 0.5), f
    raise ValueError(source)


def main(direction: str, folds: list[int], thresholds: list[int], draws: int, shifts: list[int] | None, force: bool) -> None:
    import torch

    evalcore.setup_repo()
    from dataset import PixelSetData

    src, tgt, mdir = evalcore.DIRECTIONS[direction]
    sources = SOURCES[direction]
    shift = SHIFT[direction]
    out = RES / f"conditions_{direction}.csv"
    done = set()
    if out.exists() and not force:
        d0 = pd.read_csv(out)
        done = set(zip(d0["fold"], d0["cloud_source"], d0["threshold"].astype(str), d0["condition"], d0["draw"]))
    with RunLog(f"06_run_conditions_{direction}", {"folds": folds, "thresholds": thresholds, "draws": draws, "shift": shift}, paper=PAPER) as log:
        classes = evalcore.source_classes(src)
        s_name, t_name = evalcore.SRC_NAMES[src], evalcore.SRC_NAMES[tgt]
        n_s = len(PixelSetData(str(evalcore.DATA_ROOT), s_name, classes))
        tds = PixelSetData(str(evalcore.DATA_ROOT), t_name, classes)
        folds_all = rerunlib.make_folds(n_s, len(tds), s_name, t_name, seed=1)
        labels_all = tds.get_labels()
        pos = positions_for(tgt)
        T = len(pos)
        log.add("shift_days", shift)
        log.add("min_keep", rerunlib.MIN_KEEP)
        rows_buf = []

        def flush():
            if rows_buf:
                df = pd.DataFrame(rows_buf)
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                rows_buf.clear()

        for fold in folds:
            test_idx = np.array(sorted(folds_all[fold][t_name]["test"]))
            y = labels_all[test_idx]
            t0 = time.time()
            parcels = evalcore.read_parcels(t_name, test_idx)
            print(f"{direction} fold {fold}: loaded {len(parcels)} parcels in {time.time() - t0:.0f}s", flush=True)
            S = np.array([p.shape[2] for p in parcels])
            nodata = rerunlib.nodata_dates(parcels)
            print(f"fold {fold}: all-zero (no coverage) parcel-dates {nodata.mean():.3f}", flush=True)
            model = evalcore.load_model(evalcore.MODELS / mdir, fold, len(classes))
            full = np.ones((len(parcels), T), dtype=bool)

            def run(keep, sh):
                batches = rerunlib.group_by_shape(keep.sum(axis=1), S, 128)
                pred = evalcore.predict(model, parcels, None, pos, keep, sh, batches)
                m = rerunlib.metrics(y, pred, len(classes))
                m["mean_dates_kept"] = float(keep.sum(axis=1).mean())
                return m

            def rec(source, thr, cond, draw, m, sh, nrem=np.nan):
                rows_buf.append({"direction": direction, "fold": fold, "cloud_source": source, "threshold": thr, "condition": cond,
                                 "draw": draw, "shift_days": sh, "n_test": len(y), "mean_removed": nrem, **m})

            for cond, sh in (("a", 0), ("c", shift)):
                if (fold, "none", "nan", cond, -1) not in done:
                    rec("none", np.nan, cond, -1, run(full, sh), sh, 0.0)
            for s_i in (shifts or []):
                if (fold, "none", "nan", f"sweep{s_i}", -1) not in done:
                    rec("none", np.nan, f"sweep{s_i}", -1, run(full, s_i), s_i)
            flush()
            for source in sources:
                for thr in thresholds:
                    if source in PRIMARY_ONLY and thr != 60:
                        continue
                    flags, score = flags_for(direction, source, test_idx, thr, nodata)
                    keep_c = rerunlib.keep_mask_cloud(flags, score)
                    nrem = rerunlib.n_to_remove(flags)
                    key = (fold, source, str(thr / 100))
                    for cond, sh in (("b", 0), ("d", shift)):
                        if key + (cond, -1) not in done:
                            rec(source, thr / 100, cond, -1, run(keep_c, sh), sh, float(nrem.mean()))
                    for k in range(draws):
                        rng = np.random.default_rng([42, fold, thr, k, sources.index(source)])
                        keep_r = rerunlib.keep_mask_random(nrem, T, rng)
                        for cond, sh in (("e", 0), ("e_shift", shift)):
                            if key + (cond, k) not in done:
                                rec(source, thr / 100, cond, k, run(keep_r, sh), sh, float(nrem.mean()))
                    flush()
                    print(f"{direction} fold {fold} {source} thr {thr / 100}: done {time.time() - t0:.0f}s, mean removed {nrem.mean():.1f}/{T}", flush=True)
            del parcels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=list(evalcore.DIRECTIONS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--thresholds", type=int, nargs="+", default=[60, 50, 70])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--sweep", type=int, nargs="*", default=None, help="extra shifts (days) for condition a-like sweep")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.direction, a.folds, a.thresholds, a.draws, a.sweep, a.force)
