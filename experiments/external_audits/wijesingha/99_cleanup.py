#!/usr/bin/env python
"""Delete downloaded wijesingha data under the external root once results are complete; keeps manifest.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import cleanup, paths  # noqa: E402

PAPER = "wijesingha"
# Used only if common/cleanup.py's EXPECTED registry has no entry for this paper yet.
LOCAL_EXPECTED = [
    ("notebook_facts.csv", 18),
    ("cropland_points.csv", 4000),
    ("obs_counts_points.csv.gz", 36000),
    ("year_summary.csv", 18),
    ("period_contrast.csv", 2),
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
