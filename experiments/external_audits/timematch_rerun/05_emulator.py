#!/usr/bin/env python
"""Cloud Score+ emulator for DK1 (no parcel polygons shipped). A gradient-boosted
classifier maps a parcel-date's band medians and means to the Earth Engine label
'at least half of the parcel's pixels have cs_cdf < t' (t = 0.5, 0.6, 0.7), trained
on FR1 parcels with exact polygon labels (parcel-wise split) and validated (a) on
held-out FR1 parcels and (b) against block-level Cloud Score+ cropland fractions on
DK1 (emulated cloudy fraction per block and date vs the Earth Engine fraction).
Outputs: results/emulator_validation.csv, results/dk1_block_agreement.csv and
<root>/emulator/dk1_flags_t{50,60,70}.npy, dk1_prob_t*.npy ((60332, 52) arrays).
Run in env deep_field: python -I 05_emulator.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch_rerun import rerunlib  # noqa: E402

PAPER = "timematch_rerun"
ROOT = Path("/mnt/bigdrive/external_audits/timematch_rerun")
RES = HERE / "results"
THRS = (50, 60, 70)


def load_fr1_labels():
    files = sorted((ROOT / "ee_chunks").glob("fr1_chunk_*.npz"))
    idx = np.concatenate([np.load(f)["idx"] for f in files])
    frac = np.concatenate([np.load(f)["frac"] for f in files])  # (P, 3, T)
    return idx, frac


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    with RunLog("05_emulator", {}, paper=PAPER) as log:
        idx, frac = load_fr1_labels()
        fz = np.load(ROOT / "features" / "features_FR1_union.npz")
        assert (fz["idx"] == idx).all()
        Xf = fz["feats"]  # (P, T, 20)
        P, T, F = Xf.shape
        rng = np.random.default_rng(42)
        is_train = rng.random(P) < 0.7
        dk = np.load(ROOT / "features" / "features_DK1_all.npz")["feats"]  # (60332, 52, 20)
        meta = pd.read_csv(ROOT / "parcels_DK1.csv")
        blocks = pd.read_csv(RES / "block_cloud_by_date.csv")
        dk_dates = json.loads((ROOT / "dates_DK1.json").read_text())
        out_dir = ROOT / "emulator"
        out_dir.mkdir(exist_ok=True)
        val_rows, blk_rows = [], []
        for ti, t in enumerate(THRS):
            y = rerunlib.cloud_flags(frac[:, ti, :], 0.5)  # (P, T)
            valid = ~np.isnan(frac[:, ti, :])
            tr = is_train[:, None] & valid
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42)
            clf.fit(Xf[tr], y[tr])
            te = (~is_train)[:, None] & valid
            p_te = clf.predict_proba(Xf[te])[:, 1]
            yte = y[te]
            pred = p_te >= 0.5
            tp = (pred & yte).sum(); fp = (pred & ~yte).sum(); fn = (~pred & yte).sum()
            val_rows.append({"threshold": t / 100, "n_train_parcel_dates": int(tr.sum()), "n_heldout_parcel_dates": int(te.sum()),
                             "heldout_base_rate": float(yte.mean()), "auc": float(roc_auc_score(yte, p_te)),
                             "accuracy": float((pred == yte).mean()), "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1))})
            prob = clf.predict_proba(dk.reshape(-1, F))[:, 1].reshape(dk.shape[0], dk.shape[1])
            prob = np.where(dk[..., 10:].max(-1) == 0, 0.0, prob)  # all-zero (no coverage) rows: not a cloud call
            np.save(out_dir / f"dk1_prob_t{t}.npy", prob.astype(np.float32))
            np.save(out_dir / f"dk1_flags_t{t}.npy", prob >= 0.5)
            # agreement with block-level Earth Engine cropland fractions
            b = blocks[blocks["tile"] == "DK1"].copy()
            col = f"frac_cloudy_t{t}"
            fl = prob >= 0.5
            for d_i, d in enumerate(dk_dates):
                ds = f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}"
                for blk, g in meta.groupby("block"):
                    ee_v = b[(b["block"] == blk) & (b["date"] == ds)][col]
                    blk_rows.append({"threshold": t / 100, "date": ds, "block": int(blk), "n_parcels": len(g),
                                     "emulated_frac_cloudy": float(fl[g["idx"].values, d_i].mean()),
                                     "ee_block_frac_cloudy": float(ee_v.iloc[0]) if len(ee_v) and pd.notna(ee_v.iloc[0]) else np.nan})
        bdf = pd.DataFrame(blk_rows)
        write_csv(bdf, RES / "dk1_block_agreement.csv", "05_emulator", ["features_DK1_all.npz", "block_cloud_by_date.csv"])
        for r in val_rows:
            g = bdf[(bdf["threshold"] == r["threshold"])].dropna()
            r["dk1_block_pearson_r"] = float(np.corrcoef(g["emulated_frac_cloudy"], g["ee_block_frac_cloudy"])[0, 1])
            r["dk1_mean_emulated_frac"] = float(g["emulated_frac_cloudy"].mean())
            r["dk1_mean_ee_block_frac"] = float(g["ee_block_frac_cloudy"].mean())
        write_csv(pd.DataFrame(val_rows), RES / "emulator_validation.csv", "05_emulator", ["features_FR1_union.npz", "ee_chunks/*.npz"])
        print(pd.DataFrame(val_rows).T)


if __name__ == "__main__":
    main()
