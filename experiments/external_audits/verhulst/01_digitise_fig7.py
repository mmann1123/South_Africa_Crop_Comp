#!/usr/bin/env python
"""Digitise Figure 7 of Verhulst et al. 2024: per-scenario overall accuracy (panel a: RF and MLP box
medians, SVM asterisks, for every training/validation year pair) and the mean accuracy loss per
validation year (panels b-d, bars). The OA values are read off the raster (about 0.1 to 0.3 points
per marker); the bars give an independent check of the per-scenario losses."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.verhulst import catalogue, verhulstlib as vl  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
YEARS = catalogue.YEARS
COLOURS = {"RF": (102, 150, 152), "MLP": (187, 131, 80), "SVM": (228, 127, 161)}
BAR_COLOURS = {"RF": (102, 150, 152), "SVM": (205, 125, 154), "MLP": (187, 131, 80)}
Y_AXIS = (50.0, 90.0)  # OA axis limits of every panel-a subplot
X_ZERO_PX, X_PER_PT = 156.0, 72.8  # loss axis: 0 at column 156, 2.5 points per 182 px


def load_rasters(pdf: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    doc = fitz.open(pdf)
    page = doc[catalogue.FIG7_PAGE - 1]
    imgs = {}
    for im in page.get_images(full=True):
        pix = fitz.Pixmap(doc, im[0])
        if pix.n - pix.alpha != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        imgs[(pix.w, pix.h)] = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()
    top, bottom = imgs[(1639, 954)], imgs[(1639, 475)]
    info = {"pdf_page": catalogue.FIG7_PAGE, "top_sha256": hashlib.sha256(top.tobytes()).hexdigest(), "bottom_sha256": hashlib.sha256(bottom.tobytes()).hexdigest()}
    return top, bottom, info


def panel_geometry(rgb: np.ndarray) -> tuple[list[list[int]], float, float]:
    """Slot boundaries (6 columns per panel) and the row calibration of panel a.

    Dividers and frames are the non-white columns on a data-free row near the x axis; the
    panel groups are the five runs separated by the wide gaps between subplots.
    """
    row = rgb[720].astype(int)
    cols = np.flatnonzero(row.sum(1) < 700)
    groups = [(a + b) // 2 for a, b in vl.runs(np.isin(np.arange(rgb.shape[1]), cols))]
    panels, cur = [], [groups[0]]
    for g in groups[1:]:
        if g - cur[-1] > 70 and len(cur) >= 6 or len(cur) == 6:
            panels.append(cur)
            cur = [g]
        else:
            cur.append(g)
    panels.append(cur)
    if len(panels) != 5 or any(len(p) != 6 for p in panels):
        raise RuntimeError(f"expected 5 panels with 6 column bounds, got {[len(p) for p in panels]}")
    col = rgb[:, 75:120].astype(int)
    frame_rows = [y for y in range(40, 800) if (np.abs(col[y] - col[y, 0]).sum(1) < 8).mean() > 0.9 and col[y, 0].sum() < 690]
    y_top, y_bot = frame_rows[0], frame_rows[-1]
    return panels, float(y_top), float(y_bot)


def oa_of_row(row: float, y_top: float, y_bot: float) -> float:
    return Y_AXIS[1] - (row - y_top) / (y_bot - y_top) * (Y_AXIS[1] - Y_AXIS[0])


def digitise_panel_a(rgb: np.ndarray) -> pd.DataFrame:
    panels, y_top, y_bot = panel_geometry(rgb)
    out = []
    for vy, bounds in zip(YEARS, panels):
        for ty, x0, x1 in zip(YEARS, bounds[:-1], bounds[1:]):
            x0, x1 = x0 + 3, x1 - 3
            plot = rgb[int(y_top) + 3 : int(y_bot) - 3, x0:x1]
            off = int(y_top) + 3
            for algo in ("RF", "MLP"):
                box = vl.box_from_mask(vl.colour_mask(plot, COLOURS[algo]))
                if box is None:  # a box with no fill (all runs equal): its median is a short dark line
                    line = (plot.astype(int).sum(-1) < 420) & (np.ptp(plot.astype(int), axis=-1) < 30)
                    line_rows = [r for r in range(line.shape[0]) if line[r].sum() >= 6 and line[r].sum() <= 25]
                    row = float(np.median(line_rows)) if line_rows else None
                    out.append({"algo": algo, "validation_year": vy, "training_year": ty, "oa": round(oa_of_row(row + off, y_top, y_bot), 2) if row is not None else np.nan, "degenerate_box": True})
                    continue
                t, m, b = (oa_of_row(r + off, y_top, y_bot) for r in box)
                out.append({"algo": algo, "validation_year": vy, "training_year": ty, "oa": round(m, 2), "box_q3": round(t, 2), "box_q1": round(b, 2)})
            pink = (plot[..., 0] > 200) & (plot[..., 1] < 150) & (plot[..., 2] > 100) & (plot[..., 2] < 200)
            c = vl.blob_centroid(pink)
            out.append({"algo": "SVM", "validation_year": vy, "training_year": ty, "oa": round(oa_of_row(c[0] + off, y_top, y_bot), 2) if c else np.nan})
    return pd.DataFrame(out)


def digitise_bars(top: np.ndarray, bottom: np.ndarray) -> pd.DataFrame:
    """Bar end of each validation year (panels b, c, d), measured on rows just off the error-bar line."""
    out = []
    for algo, img, y0, y1 in (("RF", top, 800, 954), ("SVM", bottom, 60, 210), ("MLP", bottom, 270, 420)):
        m = vl.colour_mask(img[y0:y1], BAR_COLOURS[algo], tol=16)
        m[:, :150] = False
        rows = vl.runs(m[:, 160:200].sum(1) > 25)
        merged = []
        for r in rows:  # an error-bar line across the bar edge can split it
            if merged and r[0] - merged[-1][1] <= 8:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        rows = [r for r in merged if r[1] - r[0] >= 10]
        if len(rows) != 5:
            raise RuntimeError(f"{algo}: found {len(rows)} bars, expected 5")
        for vy, (a, b) in zip(YEARS, rows):
            ends = []
            for r in (a + 3, a + 5, b - 5, b - 3):
                xs = np.flatnonzero(m[r])
                ends.append(xs.max())
            out.append({"algo": algo, "validation_year": vy, "mean_loss_bars": round((float(np.median(ends)) - X_ZERO_PX) / X_PER_PT, 2)})
    return pd.DataFrame(out)


def main(force: bool = False) -> None:
    out = RES / "fig7_oa_digitised.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("01_digitise_fig7", {"force": force}, paper=PAPER) as log:
        pdf = catalogue.paper_pdf()
        top, bottom, info = load_rasters(pdf)
        log.add("figure", info)
        oa = digitise_panel_a(top)
        bars = digitise_bars(top, bottom)
        write_csv(oa, out, "01_digitise_fig7", [str(pdf), catalogue.PDF_URL])
        write_csv(bars, RES / "fig7_bar_losses_digitised.csv", "01_digitise_fig7", [str(pdf)])
        log.count("oa_markers", int(oa["oa"].notna().sum()))
        print(oa.pivot_table(index=["algo", "validation_year"], columns="training_year", values="oa").to_string())
        print(bars.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
