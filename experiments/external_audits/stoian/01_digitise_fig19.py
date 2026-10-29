#!/usr/bin/env python
"""Digitise Figure 19 (kappa per tile, FG-UNET and RF, 11 tiles vs 10 tiles + 1 test)
from the raster embedded in the HAL deposit of Stoian et al. 2019."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.stoian import catalogue, fig19  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
YTICKS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
PLOT_Y = (34, 266)  # rows of the plot area (below the legend, above the x tick labels)
LABEL_X = (36, 62)
XLABEL_Y = (268, 284)
MARKER_PX = 15  # marker box size in pixels
FULL_VISIBLE = 0.9


def extract_figure(pdf: Path, out_png: Path) -> tuple[np.ndarray, dict]:
    doc = fitz.open(pdf)
    page = doc[catalogue.FIG19_PAGE - 1]
    found = [im for im in page.get_images(full=True) if (im[2], im[3]) == (739, 312)]
    if len(found) != 1:
        raise RuntimeError(f"expected one 739x312 raster on page {catalogue.FIG19_PAGE}, found {len(found)}")
    xref = found[0][0]
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_png)
    return arr, {"pdf_page": catalogue.FIG19_PAGE, "xref": xref, "size": [pix.w, pix.h], "rgb_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def largest_window(mask: np.ndarray, xc: float, y_range: tuple[int, int]) -> np.ndarray | None:
    y0, y1 = y_range
    x0, x1 = int(round(xc)) - 10, int(round(xc)) + 11
    sub = mask[y0:y1, x0:x1]
    lab, n = ndi.label(sub, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndi.sum(sub, lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(big)
    return big[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def flag_hidden(df: pd.DataFrame, kappa_per_px: float) -> pd.DataFrame:
    """Mark markers whose box overlaps another series' marker in the same column and whose
    visible template share is below FULL_VISIBLE times that series' median among
    non-overlapping markers (the expected share of an unobstructed marker)."""
    df = df.copy()
    overlap = []
    for _, r in df.iterrows():
        others = df[(df["tile"] == r["tile"]) & (df["series"] != r["series"])]
        overlap.append(bool((abs(others["marker_row_px"] - r["marker_row_px"]) < MARKER_PX).any()))
    df["overlaps_other_marker"] = overlap
    ref = df[~df["overlaps_other_marker"]].groupby("series")["visible_share"].median()
    df["partly_hidden"] = df["overlaps_other_marker"] & (df["visible_share"] < df["series"].map(ref) * FULL_VISIBLE)
    df["uncertainty_kappa"] = [round(max(0.015 if h else 0.005, (sp / 2 + 1) * kappa_per_px), 3) for h, sp in zip(df["partly_hidden"], df["row_span_px"])]
    return df.drop(columns="row_span_px")


def digitise(rgb: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = fig19.tick_label_rows(rgb, *LABEL_X)
    if len(rows) != len(YTICKS):
        raise RuntimeError(f"found {len(rows)} y tick labels, expected {len(YTICKS)}")
    cal = fig19.calibrate_y(rows, YTICKS)
    resid = [fig19.row_to_value((a + b) / 2, cal) - v for (a, b), v in zip(rows, YTICKS)]
    xs = fig19.column_centres(rgb, *XLABEL_Y)
    if len(xs) != len(catalogue.TILES):
        raise RuntimeError(f"found {len(xs)} x tick labels, expected {len(catalogue.TILES)}")
    out = []
    masks = {}
    for name, colour in fig19.SERIES.items():
        m = fig19.alpha_mask(rgb, colour)
        m[: PLOT_Y[0]] = False
        m[PLOT_Y[1] :] = False
        m[:, : LABEL_X[1] + 6] = False
        masks[name] = m
    for name, mask in masks.items():
        others = np.zeros_like(mask)
        for n2, m2 in masks.items():
            if n2 != name:
                others |= m2
        others = ndi.binary_dilation(others, iterations=1)
        windows = [w for w in (largest_window(mask, x, PLOT_Y) for x in xs) if w is not None]
        template = fig19.make_template(windows)
        for tile, x in zip(catalogue.TILES, xs):
            loc = fig19.locate(mask, template, x, PLOT_Y, others)
            val = fig19.row_to_value(loc["row"], cal)
            out.append(
                {
                    "tile": tile,
                    "series": name,
                    "kappa": round(val, 3),
                    "marker_row_px": round(loc["row"], 1),
                    "marker_col_px": round(loc["col"], 1),
                    "visible_share": round(loc["visible_share"], 2),
                    "row_span_px": loc["row_span"],
                }
            )
    df = pd.DataFrame(out)
    df = flag_hidden(df, abs(cal[0]))
    info = {"y_calibration_slope_per_px": cal[0], "y_calibration_intercept": cal[1], "tick_label_rows": rows, "tick_fit_residual_max": float(np.abs(resid).max()), "x_centres_px": xs}
    return df, info


def wide(df: pd.DataFrame) -> pd.DataFrame:
    k = df.pivot(index="tile", columns="series", values="kappa").reindex(catalogue.TILES)
    h = df.pivot(index="tile", columns="series", values="partly_hidden").reindex(catalogue.TILES)
    w = k.copy()
    for c in k.columns:
        w[f"{c}_partly_hidden"] = h[c]
    return w.reset_index()


def main(force: bool = False) -> None:
    out = RES / "fig19_kappa_digitised.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("01_digitise_fig19", {"force": force}, paper=PAPER) as log:
        pdf = catalogue.paper_pdf()
        rgb, info = extract_figure(pdf, paths.paper_root(PAPER) / "fig19_raster.png")
        log.add("figure", info)
        df, cal = digitise(rgb)
        log.add("calibration", cal)
        write_csv(df, out, "01_digitise_fig19", [str(pdf), catalogue.HAL_PDF_URL])
        write_csv(wide(df), RES / "fig19_kappa_digitised_wide.csv", "01_digitise_fig19", [str(out)])
        log.count("markers", len(df))
        log.count("partly_hidden", int(df["partly_hidden"].sum()))
        print(wide(df).to_string(index=False))
        print(df[df["partly_hidden"]][["tile", "series", "kappa", "visible_share"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
