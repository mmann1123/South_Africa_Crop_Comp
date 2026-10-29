#!/usr/bin/env python
"""Stream the authors' Dropbox S4A-Models.zip once and keep only the nine COCO
split files and the OAD test CSVs of the published runs. The archive is built on
the fly by Dropbox (no Range support), so the whole 13.7 GB is read, but only the
selected members touch disk. A member listing (name, size, kept or not) is saved
to results/zip_member_listing.csv and every kept file goes into manifest.json
with its SHA-256."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import zipstream  # noqa: E402

PAPER = "sen4agrinet"
SHARE_URL = "https://www.dropbox.com/sh/kvgo4r2vin7sbwt/AACzDLNbnSouuZYMk8Y9I4sha?dl=1"
EXPECTED_ZIP_BYTES = 13_728_117_098
COCO_RE = re.compile(r"^coco_files/exp[123]_patches5000_strat_coco_(train|val|test)\.json$")
CSV_RE = re.compile(r"\.csv(\.gz)?$")


def keep(name: str, size: int) -> bool:
    """COCO split files of all three scenarios, and OAD CSVs whose name marks them
    as a test split (the train CSVs are 2 to 3 GB each and are not needed)."""
    if COCO_RE.match(name):
        return True
    return bool(CSV_RE.search(name)) and "test" in Path(name).name.lower()


def main(force: bool = False) -> None:
    root = paths.paper_root(PAPER) / "dropbox"
    manifest = paths.manifest_path(PAPER)
    listing_path = paths.results_dir(PAPER) / "zip_member_listing.csv"
    with RunLog("02_fetch_splits", {"force": force}, paper=PAPER) as log:
        if listing_path.exists() and not force:
            have = pd.read_csv(listing_path)
            kept = have[have["kept"]]
            if len(kept) and all((root / n).exists() for n in kept["name"]):
                print("selected members already on disk; use --force to re-stream")
                log.discard()
                return
        s = requests.get(SHARE_URL, stream=True, timeout=fetch.TIMEOUT)
        s.raise_for_status()
        total = int(s.headers.get("Content-Length", 0))
        log.add("zip_bytes", total)
        if total != EXPECTED_ZIP_BYTES:
            log.add("warning", f"zip size {total} differs from the recorded {EXPECTED_ZIP_BYTES}")
        rows, entries = [], [e for e in fetch.manifest_entries(manifest) if not e["key"].startswith("dropbox/")]
        t0 = time.time()

        def want(h: zipstream.Header) -> Path | None:
            return root / h.name if keep(h.name, h.uncompressed_size) else None

        def on_member(h: zipstream.Header, dest: Path | None, sha: str | None) -> None:
            rows.append({"name": h.name, "bytes": h.uncompressed_size, "kept": dest is not None, "sha256": sha or ""})
            if dest is not None:
                entries.append({"key": "dropbox/" + h.name, "url": SHARE_URL, "bytes": h.uncompressed_size, "sha256": sha, "fetched_at": pd.Timestamp.now("UTC").isoformat()})
                print(f"kept {h.name} ({h.uncompressed_size / 1e6:.0f} MB) at {(time.time() - t0) / 60:.1f} min", flush=True)

        consumed = zipstream.walk(s.iter_content(1 << 22), want, on_member)
        s.close()
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(entries, indent=2))
        df = pd.DataFrame(rows)
        write_csv(df, listing_path, "02_fetch_splits", [SHARE_URL])
        log.count("members", len(df))
        log.count("members_kept", int(df["kept"].sum()))
        log.add("stream_bytes_consumed", consumed)
        log.add("minutes", round((time.time() - t0) / 60, 1))
        names = df.loc[df["kept"], "name"].tolist()
        log.add("kept_names", names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
