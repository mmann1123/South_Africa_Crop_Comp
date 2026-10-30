#!/usr/bin/env python
"""Crawl the anonymous Kaggle file listing of ignazio/sentinel2-crop-mapping (the authors' deposit)
and save every file name with its size. Image dates are in the file names, so no pixel is downloaded.
Resumable: names are appended to listing.tsv and the next-page token to listing.token."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.gallo import gallolib  # noqa: E402

PAPER = gallolib.PAPER
ROOT = paths.paper_root(PAPER)
URL = "https://www.kaggle.com/api/v1/datasets/list/ignazio/sentinel2-crop-mapping"
LISTING = ROOT / "listing.tsv"
TOKEN = ROOT / "listing.token"
DONE = ROOT / "listing.done"


def fetch_page(token: str | None) -> dict:
    q = "?pageSize=1000" + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
    for attempt in range(12):
        try:
            return json.load(urllib.request.urlopen(URL + q, timeout=60))
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 2 * 2**attempt))
    raise RuntimeError("Kaggle listing page failed")


def main(force: bool = False) -> None:
    if force:
        for p in (LISTING, TOKEN, DONE):
            p.unlink(missing_ok=True)
    if DONE.exists():
        print(f"{DONE} exists; use --force")
        return
    with RunLog("01_list_kaggle", {"force": force}, paper=PAPER) as log:
        token = TOKEN.read_text().strip() if TOKEN.exists() else None
        pages = n = 0
        with open(LISTING, "a") as fh:
            while True:
                d = fetch_page(token)
                for f in d["datasetFiles"]:
                    fh.write(f"{f['name']}\t{f['totalBytes']}\n")
                n += len(d["datasetFiles"])
                pages += 1
                token = d.get("nextPageToken")
                fh.flush()
                time.sleep(0.3)
                if not token:
                    break
                TOKEN.write_text(token)
                if pages % 200 == 0:
                    print(pages, n, d["datasetFiles"][-1]["name"], flush=True)
        DONE.write_text("done")
        log.count("files_this_run", n)
        log.count("pages_this_run", pages)
        log.add("source", URL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
