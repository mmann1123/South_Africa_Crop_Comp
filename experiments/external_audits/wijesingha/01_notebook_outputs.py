#!/usr/bin/env python
"""Fetch the authors' notebooks and download script (SHA-256 manifest) and record what their
saved outputs and code show: printed array shapes after dropna(), the year and region split,
and whether anything fills or counts gaps."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.wijesingha import catalogue  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
REPO = paths.paper_root(PAPER) / "repo"
FILL_WORDS = ("fillna", "interpolate", "SimpleImputer", "KNNImputer", "ffill", "bfill", "unmask")
SHAPE = re.compile(r"(Train|Val) X: \((\d+), 10, 4, 1\)")


def source_and_outputs(nb: dict) -> tuple[str, list[tuple[int, str]]]:
    src, outs = [], []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src.append("".join(c["source"]))
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain") or ""
            outs.append((i, "".join(t)))
    return "\n".join(src), outs


def main(force: bool = False) -> None:
    out = RES / "notebook_facts.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("01_notebook_outputs", {"force": force}, paper=PAPER) as log:
        files = list(catalogue.NOTEBOOKS.values()) + [catalogue.DOWNLOAD_SCRIPT]
        for f in files:
            fetch.download(catalogue.REPO_RAW + f, REPO / f, paths.manifest_path(PAPER), f"repo/{f}")
        rows = []
        for region, f in catalogue.NOTEBOOKS.items():
            nb = json.loads((REPO / f).read_text())
            src, outs = source_and_outputs(nb)
            shapes = {}
            for _, text in outs:
                for kind, n in SHAPE.findall(text):
                    shapes.setdefault(kind, int(n))
            test_n = [int(n) for n in re.findall(r"Shape of the test dataset\s*\n\s*Val X: \((\d+),", "\n".join(t for _, t in outs))]
            facts = {
                "dropna_in_code": "dropna()" in src,
                "fill_words_found": ",".join(w for w in FILL_WORDS if w in src) or "none",
                "excluded_years": re.search(r"no_years = (\[.*?\])", src).group(1),
                "holdout_year": re.search(r"test_years = (\d+)", src).group(1),
                "holdout_nuts": re.search(r"test_nuts = '(\w+)'", src).group(1),
                "rows_after_dropna_train_sample": shapes.get("Train"),
                "rows_after_dropna_val": shapes.get("Val"),
                "rows_after_dropna_holdout_test": test_n[0] if test_n else None,
                "rows_before_dropna_printed": bool(re.search(r"print\(.*(isna|isnull)", src)),
            }
            for k, v in facts.items():
                rows.append({"region": region, "notebook": f, "fact": k, "value": v})
        write_csv(pd.DataFrame(rows), out, "01_notebook_outputs", [catalogue.REPO_URL, str(paths.manifest_path(PAPER))])
        log.count("facts", len(rows))
        print(pd.DataFrame(rows)[["region", "fact", "value"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
