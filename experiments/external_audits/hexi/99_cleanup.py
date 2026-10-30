#!/usr/bin/env python
"""Delete downloaded hexi data under the external root once results are complete; keeps manifest.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import cleanup, paths  # noqa: E402

PAPER = "hexi"
# Used only if common/cleanup.py's EXPECTED registry has no entry for this paper yet.
LOCAL_EXPECTED = [
    ("points.csv", 2900),
    ("weekly_gap_summary.csv", 96),
    ("residual_cloud.csv", 24),
    ("weekly_valid_share.csv", 1200),
    ("monthly_qa60.csv", 72),
    ("discard_rules.csv", 24),
    ("qa60_images.csv", 100),
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
