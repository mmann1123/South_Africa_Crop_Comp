#!/usr/bin/env python
"""Transcribe Table 1 (class areas per tile, ha) of Stoian et al. 2019 into a CSV.

Source: paper p. 6 of 26 (page 7 of the HAL PDF with its cover page). Every printed row total
is checked against the recomputed sum (all 17 reconcile). The printed column totals
exceed the sum of the 17 rows by 5 to 9 ha per tile (74 ha overall, 0.004 percent); this
is in the paper and is logged, not corrected."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.stoian import catalogue, table1  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)


def main(force: bool = False) -> None:
    out = RES / "table1_class_areas.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("02_transcribe_table1", {"force": force}, paper=PAPER) as log:
        pdf = catalogue.paper_pdf()
        text = fitz.open(pdf)[catalogue.TABLE1_PAGE - 1].get_text()
        df = table1.parse_table1(text, catalogue.TILES)
        problems, col_diff = table1.check_totals(df, catalogue.TILES)
        log.add("row_total_problems", problems)
        log.add("printed_minus_summed_column_totals_ha", col_diff)
        if problems:
            raise RuntimeError(f"Table 1 row totals do not reconcile (transcription error): {problems}")
        df.insert(1, "class_name", df["class"].map(table1.CLASSES))
        df["source"] = "Stoian et al. 2019, Remote Sens. 11:1986, Table 1, p. 6 of 26 (HAL PDF page 7)"
        write_csv(df, out, "02_transcribe_table1", [str(pdf), catalogue.HAL_PDF_URL])
        log.count("classes", len(df))
        log.add("grand_total_ha", int(df["total_ha"].sum()))
        print(df.drop(columns="source").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
