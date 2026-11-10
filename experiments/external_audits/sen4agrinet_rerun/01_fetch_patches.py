#!/usr/bin/env python
"""Download the test-split patch NetCDFs of PAD Scenarios 1 to 3 from the
Hugging Face dataset orion-ai-lab/S4A at the revision the descriptive audit
pinned. About 17 MB each, 3,017 files, about 51 GB. Parallel, resumable
(a file is skipped when it exists with the size the server reports), SHA-256
of every file in results/patch_manifest.csv."""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import splits  # noqa: E402

PAPER = "sen4agrinet_rerun"
HF_REVISION = "4eb44157012b9ec251767721d6bae55cf0c564e2"
HF_URL = "https://huggingface.co/datasets/orion-ai-lab/S4A/resolve/" + HF_REVISION + "/{path}"
SPLITS = paths.AUDIT_DIR / "sen4agrinet" / "results" / "split_patches.csv.gz"


def netcdf_dir() -> Path:
    d = paths.paper_root(PAPER) / "netcdf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_path(stem: str) -> Path:
    return netcdf_dir() / splits.hf_path(stem)


def fetch_one(stem: str) -> dict:
    url = HF_URL.format(path=splits.hf_path(stem))
    dest = local_path(stem)
    sha = hashlib.sha256()
    if dest.exists():
        with open(dest, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                sha.update(chunk)
        return {"stem": stem, "bytes": dest.stat().st_size, "sha256": sha.hexdigest(), "fetched": False}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".nc.part")
    for attempt in range(5):
        try:
            sha = hashlib.sha256()
            n = 0
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", 0)) or None
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        sha.update(chunk)
                        n += len(chunk)
            if expected is not None and n != expected:
                raise IOError(f"{stem}: {n} of {expected} bytes")
            part.replace(dest)
            return {"stem": stem, "bytes": n, "sha256": sha.hexdigest(), "fetched": True}
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
    raise RuntimeError("unreachable")


def main(scenarios: list[int], split: str, workers: int, force: bool) -> None:
    with RunLog("01_fetch_patches", {"scenarios": scenarios, "split": split, "workers": workers}, paper=PAPER) as log:
        sp = pd.read_csv(SPLITS)
        stems = sorted(set(sp.loc[sp["scenario"].isin(scenarios) & (sp["split"] == split), "stem"]))
        out = paths.results_dir(PAPER) / f"patch_manifest_{split}.csv"
        rows = []
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch_one, s): s for s in stems}
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 100 == 0:
                    print(f"{i}/{len(stems)}", flush=True)
        df = pd.DataFrame(rows).sort_values("stem")
        write_csv(df, out, "01_fetch_patches", [HF_URL.format(path="data/<year>/<tile>/<stem>.nc")])
        log.count("files", len(df))
        log.count("fetched_now", int(df["fetched"].sum()))
        log.add("gigabytes", round(df["bytes"].sum() / 1e9, 2))
        log.add("hf_revision", HF_REVISION)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
