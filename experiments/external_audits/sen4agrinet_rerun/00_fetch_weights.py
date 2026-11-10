#!/usr/bin/env python
"""Stream the authors' Dropbox S4A-Models.zip once more and keep only the nine
released PAD checkpoints (U-Net, ConvLSTM, ConvSTAR x Scenarios 1 to 3). Same
forward-only reader as ../sen4agrinet/02_fetch_splits.py (Dropbox ignores Range)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.sen4agrinet import zipstream  # noqa: E402

PAPER = "sen4agrinet_rerun"
SHARE_URL = "https://www.dropbox.com/sh/kvgo4r2vin7sbwt/AACzDLNbnSouuZYMk8Y9I4sha?dl=1"


def keep(name: str) -> bool:
    return name.startswith("model_weights/PAD/") and name.endswith(".ckpt")


def main(force: bool = False) -> None:
    root = paths.paper_root(PAPER) / "dropbox"
    manifest = paths.manifest_path(PAPER)
    have = sorted(root.glob("model_weights/PAD/*/*/*.ckpt"))
    with RunLog("00_fetch_weights", {"force": force}, paper=PAPER) as log:
        if len(have) == 9 and not force:
            print("nine PAD checkpoints already on disk")
            log.discard()
            return
        s = requests.get(SHARE_URL, stream=True, timeout=fetch.TIMEOUT)
        s.raise_for_status()
        entries, t0 = [], time.time()

        def want(h):
            return root / h.name if keep(h.name) else None

        def on_member(h, dest, sha):
            if dest is not None:
                entries.append({"key": "dropbox/" + h.name, "url": SHARE_URL, "bytes": h.uncompressed_size,
                                "sha256": sha, "fetched_at": pd.Timestamp.now("UTC").isoformat()})
                print(f"kept {h.name} at {(time.time() - t0) / 60:.1f} min", flush=True)

        zipstream.walk(s.iter_content(1 << 22), want, on_member)
        s.close()
        manifest.write_text(json.dumps(entries, indent=2))
        log.count("checkpoints", len(entries))
        log.add("minutes", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
