#!/usr/bin/env python
"""Join the date counts, Cloud Score+ date labels and raster checks: composition of the 71 input slots per
area-year, pooled for the authors' training years (L1+L2 2016-18), test Y (L1+L2 2019) and test A (L3 2019),
slot calendar shifts, EE-versus-raster agreement, and the figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths, plotting  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.gallo import gallolib  # noqa: E402

PAPER = gallolib.PAPER
RES = paths.results_dir(PAPER)
FIG = paths.figures_dir(PAPER)
ORDER = [(a, y) for a in ("L1", "L2", "L3") for y in gallolib.YEARS]
POOLS = {
    "train L1+L2 2016-18": lambda a, y: a in ("L1", "L2") and y <= 2018,
    "test Y L1+L2 2019": lambda a, y: a in ("L1", "L2") and y == 2019,
    "test A L3 2019": lambda a, y: a == "L3" and y == 2019,
}
METRICS = ["n_dates", "pad_share", "zero_share", "clear_share", "cloudy_share", "nodata_dates", "mean_crop_cloudy", "px_cloud", "px_nodata_dates"]


def area_year_table() -> pd.DataFrame:
    dates = pd.read_csv(RES / "dates_by_area_year.csv", dtype={"date": str})
    cloud = pd.read_csv(RES / "cloud_by_date.csv", dtype={"date": str})
    px = pd.read_csv(RES / "pixel_state_raw.csv", dtype={"date": str})
    rows = []
    for a, y in ORDER:
        c = cloud[(cloud["area"] == a) & (cloud["year"] == y)].sort_values("date")
        p = px[(px["area"] == a) & (px["year"] == y)]
        rn = c["date"].map(p.groupby("date")["nodata"].mean()).fillna(0.0)
        labels = gallolib.classify_dates(c["crop_cloudy"].tolist(), c["n_images"].tolist(), rn.tolist())
        s = gallolib.series_summary(labels)
        per_patch = p.groupby("patch").agg(nodata_dates=("nodata", lambda v: float((v >= 0.99).sum())), cloud=("cloud", "mean"))
        rows.append({
            "area": a, "year": y, **s, "pad_share": s["pad_slots"] / gallolib.SEQ_LEN, "subsample_keep": min(1.0, gallolib.SEQ_LEN / s["n_dates"]),
            "clear_dates": labels.count("clear"), "cloudy_dates": labels.count("cloudy"), "nodata_dates": labels.count("nodata"),
            "mean_crop_cloudy": float(c["crop_cloudy"].mean()), "n_ee_dates_unmatched": int((c["n_images"] == 0).sum()),
            "px_cloud": float(per_patch["cloud"].mean()), "px_nodata_dates": float(per_patch["nodata_dates"].mean()), "px_patches": int(len(per_patch)),
        })
    return pd.DataFrame(rows)


def slot_shifts(dates: pd.DataFrame) -> pd.DataFrame:
    """Same-slot calendar offset: each training area-year against the 2019 series of its own area (test Y),
    and the L3 2019 series against every training area-year of L1 and L2 (test A)."""
    ds = {(a, y): sorted(g["date"]) for (a, y), g in dates.groupby(["area", "year"])}
    rows = []
    for a in ("L1", "L2"):
        for y in (2016, 2017, 2018):
            rows.append({"pair": f"{a} {y} vs {a} 2019 (test Y)", "pool": "test Y", **gallolib.slot_shift(ds[(a, y)], ds[(a, 2019)])})
            rows.append({"pair": f"{a} {y} vs L3 2019 (test A)", "pool": "test A", **gallolib.slot_shift(ds[(a, y)], ds[("L3", 2019)])})
    return pd.DataFrame(rows)


def pooled(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, f in POOLS.items():
        sub = t[[f(a, y) for a, y in zip(t["area"], t["year"])]]
        rows.append({"pool": name, "series": len(sub), **gallolib.pool_mean(sub, METRICS)})
    for y in (2016, 2017, 2018):
        sub = t[(t["area"].isin(["L1", "L2"])) & (t["year"] == y)]
        rows.append({"pool": f"train L1+L2 {y}", "series": len(sub), **gallolib.pool_mean(sub, METRICS)})
    return pd.DataFrame(rows)


def agreement() -> pd.DataFrame:
    """Date-level agreement between the Earth Engine cropland cloudy fraction and the raster bright-pixel share."""
    cloud = pd.read_csv(RES / "cloud_by_date.csv", dtype={"date": str})
    px = pd.read_csv(RES / "pixel_state_raw.csv", dtype={"date": str}).groupby(["area", "year", "date"])[["cloud", "nodata"]].mean().reset_index()
    m = cloud.merge(px, on=["area", "year", "date"])
    rows = []
    for (a, y), g in m.groupby(["area", "year"]):
        both = g.dropna(subset=["crop_cloudy"])
        rows.append({"area": a, "year": y, "dates": len(g), "pearson_r": float(np.corrcoef(both["crop_cloudy"], both["cloud"])[0, 1]) if len(both) > 2 else float("nan"),
                     "ee_unmatched_dates": int((g["n_images"] == 0).sum()), "unmatched_and_raster_nodata": int(((g["n_images"] == 0) & (g["nodata"] >= 0.99)).sum()),
                     "raster_nodata_but_ee_image": int(((g["n_images"] > 0) & (g["nodata"] >= 0.99)).sum())})
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, out: Path) -> Path:
    plotting.theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    parts = [("clear_slots", "clear (Cloud Score+ cloudy share < 0.5)", plotting.SERIES[2]), ("cloudy_slots", "cloudy", plotting.SERIES[1]),
             ("nodata_slots", "date with no scene or all-zero raster", plotting.INK_SECONDARY), ("pad_slots", "zero padding", plotting.GRID)]
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for col, label, color in parts:
        v = t[col].to_numpy()
        ax.bar(x, v, bottom=bottom, color=color, label=label, width=0.78, edgecolor=plotting.SURFACE, linewidth=0.4)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{y}" for a, y in zip(t["area"], t["year"])], fontsize=7)
    ax.set_ylabel("slots of the 71-image input")
    ax.set_ylim(0, 71)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    role = {(a, y): ("train" if a != "L3" and y <= 2018 else "test Y" if a != "L3" else "test A" if y == 2019 else "not used") for a, y in ORDER}
    for i, (a, y) in enumerate(zip(t["area"], t["year"])):
        ax.text(i, 72, role[(a, y)], ha="center", va="bottom", fontsize=6.5, color=plotting.INK_SECONDARY)
    fig.tight_layout()
    return plotting.save(fig, out)


def main() -> None:
    with RunLog("06_summarise", {}, paper=PAPER) as log:
        t = area_year_table()
        dates = pd.read_csv(RES / "dates_by_area_year.csv", dtype={"date": str})
        src = [str(RES / f) for f in ("dates_by_area_year.csv", "cloud_by_date.csv", "pixel_state_raw.csv")]
        write_csv(t, RES / "gallo_area_year.csv", "06_summarise", src)
        write_csv(pooled(t), RES / "gallo_pools.csv", "06_summarise", src)
        write_csv(slot_shifts(dates), RES / "gallo_slot_shift.csv", "06_summarise", src[:1])
        write_csv(agreement(), RES / "gallo_ee_vs_raster.csv", "06_summarise", src[1:])
        figure(t, FIG / "gallo_slot_composition")
        log.count("area_years", len(t))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
