#!/usr/bin/env python
"""Delete downloaded maleki data under the external root once results are complete; keeps manifest.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import cleanup, paths  # noqa: E402

PAPER = "maleki"
# Used only if common/cleanup.py's EXPECTED registry has no entry for this paper yet.
LOCAL_EXPECTED = [
    ("site_inventory.csv", 4),
    ("interp_summary.csv", 30),
    ("monthly_valid.csv", 24),
    ("pixel_metrics.csv", 3000),
    ("maleki_direction_table.csv", 2),
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
