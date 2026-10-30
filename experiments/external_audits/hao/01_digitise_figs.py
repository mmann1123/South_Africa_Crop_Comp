#!/usr/bin/env python
"""Digitise the overall-accuracy bars of Figs. 4 (Hengshui), 6 (Alberta) and 8 (Nebraska) of Hao et al.
2020: transfer learning (TL, left) and local training (LO, right) for time series DOY 60 to 75 ... 330.
Bar tops are read against the y-axis label positions; the values quoted in the text are in
results/fig_text_check.csv as a check (the read-off agrees to about 1 point)."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.hao import catalogue, haolib  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)

# Overall accuracies (percent) printed in the text of Section 4.1: (region, series, doy_end, value)
TEXT_VALUES = [
    ("HS", "TL", 330, 97.79),
    ("AB", "TL", 330, 86.45), ("AB", "LO", 330, 96.55),
    ("AB", "TL", 135, 50.12), ("AB", "TL", 150, 72.91), ("AB", "LO", 135, 63.05), ("AB", "LO", 150, 78.45),
    ("AB", "TL", 195, 76.11), ("AB", "TL", 225, 83.25), ("AB", "LO", 195, 80.30), ("AB", "LO", 225, 94.09),
    ("NE", "TL", 330, 94.86), ("NE", "LO", 330, 96.73), ("NE", "TL", 180, 80.72), ("NE", "LO", 165, 92.06), ("NE", "TL", 285, 92.99),
]


def page_figure(doc: fitz.Document, page_no: int) -> tuple[np.ndarray, str]:
    """The largest raster on the page (the figure) as RGB."""
    page = doc[page_no - 1]
    best = max(page.get_images(full=True), key=lambda im: im[2] * im[3])
    pix = fitz.Pixmap(doc, best[0])
    if pix.n - pix.alpha != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()
    return rgb, hashlib.sha256(rgb.tobytes()).hexdigest()


def digitise(rgb: np.ndarray) -> dict[str, list[float]]:
    bars = haolib.orange_bars(rgb)
    if len(bars) != 2 * catalogue.N_WINDOWS:
        raise RuntimeError(f"expected {2 * catalogue.N_WINDOWS} overall-accuracy bars, found {len(bars)}")
    out = {}
    for name, bx in (("TL", bars[: catalogue.N_WINDOWS]), ("LO", bars[catalogue.N_WINDOWS :])):
        x0, base = bx[0][0], bx[0][3]
        x_lo, x_hi = (20, x0 - 45) if name == "TL" else (x0 - 200, x0 - 45)
        rows = haolib.axis_label_rows(rgb, base, x_lo, x_hi)
        out[name] = haolib.bar_values(rows, [b[2] for b in bx])
    return out


def main(force: bool = False) -> None:
    out = RES / "fig_oa_digitised.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("01_digitise_figs", {"force": force}, paper=PAPER) as log:
        doc = fitz.open(catalogue.PDF)
        log.add("pdf", str(catalogue.PDF))
        rows = []
        for region, page in catalogue.FIG_PAGES.items():
            rgb, sha = page_figure(doc, page)
            log.add(f"figure_{region}", {"pdf_page": page, "raster_sha256": sha, "shape": list(rgb.shape)})
            vals = digitise(rgb)
            for series, v in vals.items():
                for end, val in zip(catalogue.WINDOW_ENDS, v):
                    rows.append({"region": region, "series": series, "doy_end": end, "n_windows": (end - 60) // 15, "oa": round(val, 2)})
        df = pd.DataFrame(rows)
        write_csv(df, out, "01_digitise_figs", [str(catalogue.PDF)])
        chk = pd.DataFrame(TEXT_VALUES, columns=["region", "series", "doy_end", "oa_text"]).merge(df, on=["region", "series", "doy_end"]).rename(columns={"oa": "oa_digitised"})
        chk["diff"] = (chk["oa_digitised"] - chk["oa_text"]).round(2)
        write_csv(chk[["region", "series", "doy_end", "oa_text", "oa_digitised", "diff"]], RES / "fig_text_check.csv", "01_digitise_figs", [str(catalogue.PDF)])
        log.count("bars", len(df))
        log.add("text_check_mean_abs_diff", float(chk["diff"].abs().mean()))
        log.add("text_check_max_abs_diff", float(chk["diff"].abs().max()))
        print(df.pivot_table(index="doy_end", columns=["region", "series"], values="oa").to_string())
        print(chk.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
