#!/usr/bin/env python
"""Fetch the Kerner et al. 2020 phenology arrays and CDL label arrays out of the
7.27 GB Zenodo zip (10.5281/zenodo.3997350, file record 3997351) by HTTP range
requests. Only six small members are read; the growth-stage input cubes are not."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.timematch import zipindex  # noqa: E402  (generic zip range reader, shared)

PAPER = "kerner"
ZIP_URL = "https://zenodo.org/records/3997351/files/data.zip?download=1"
EXPECTED_ZIP_BYTES = 7_269_005_245
WANTED = [f"data/phenology/16TBL_{y}_phenology.npy" for y in (2017, 2018, 2019)] + [f"data/cdl_labels_16TBL_{y}.npy" for y in (2017, 2018, 2019)]


def main(force: bool = False) -> None:
    root = paths.paper_root(PAPER)
    manifest = paths.manifest_path(PAPER)
    read = lambda s, l: fetch.range_get(ZIP_URL, s, l)  # noqa: E731
    with RunLog("01_fetch_zenodo", {"force": force}, paper=PAPER) as log:
        total = fetch.head_size(ZIP_URL)
        log.add("zip_bytes", total)
        if total != EXPECTED_ZIP_BYTES:
            log.add("warning", f"zip size {total} differs from the record's {EXPECTED_ZIP_BYTES}")
        members = {m.name: m for m in zipindex.read_central_directory(read, total)}
        log.count("zip_members_total", len(members))
        fetched = 0
        for name in WANTED:
            m = members[name]
            dest = root / name
            if dest.exists() and dest.stat().st_size > 0 and not force:
                continue
            zipindex.extract_member(read, m, dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            entries = [e for e in fetch.manifest_entries(manifest) if e["key"] != name]
            entries.append({"key": name, "url": ZIP_URL, "bytes": m.uncompressed_size, "sha256": sha, "fetched_at": pd.Timestamp.now("UTC").isoformat(), "zip_header_offset": m.header_offset, "zip_compressed_bytes": m.compressed_size})
            manifest.write_text(json.dumps(entries, indent=2))
            fetched += 1
            print(f"fetched {name} ({m.compressed_size / 1e6:.1f} MB compressed)", flush=True)
        log.count("members_fetched", fetched)
        if fetched == 0:
            print("all members already present; nothing to do (use --force to refetch)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
