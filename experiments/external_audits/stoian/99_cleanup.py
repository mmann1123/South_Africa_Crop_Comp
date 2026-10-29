#!/usr/bin/env python
"""Delete downloaded stoian data under the external root once results are complete; keeps manifest.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import cleanup, paths  # noqa: E402

PAPER = "stoian"
# Used only if common/cleanup.py's EXPECTED registry has no entry for this paper yet.
LOCAL_EXPECTED = [
    ("fig19_kappa_digitised.csv", 44),
    ("table1_class_areas.csv", 17),
    ("scl_tile_summary.csv", 11),
    ("l1c_tile_summary.csv", 11),
    ("stoian_tile_table.csv", 11),
    ("stoian_correlations.csv", 30),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="delete even if results are incomplete")
    args = ap.parse_args()
    cleanup.EXPECTED.setdefault(PAPER, LOCAL_EXPECTED)
    ok, problems = cleanup.complete(PAPER)
    print(f"{PAPER}: results complete = {ok}" + (f" ({problems})" if problems else ""))
    freed = cleanup.purge(PAPER, force=args.force)
    print(f"{PAPER}: freed {freed / 1e9:.2f} GB under {paths.paper_root(PAPER)}; kept {cleanup.KEEP}")
