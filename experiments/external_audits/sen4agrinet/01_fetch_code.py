#!/usr/bin/env python
"""Fetch the exact S4A-Models and S4A source files at the pinned commits, record
SHA-256, and extract the line ranges that carry the code facts (monthly median,
interpolate_na with linear extrapolation, OAD window, OAD train-on-test-file,
masks off by default). The files are read as text only, never run."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import codefacts  # noqa: E402

PAPER = "sen4agrinet"


def main(force: bool = False) -> None:
    out = paths.results_dir(PAPER) / "code_facts.csv"
    with RunLog("01_fetch_code", {"force": force}, paper=PAPER) as log:
        if out.exists() and not force:
            print(f"{out.name} exists; use --force to redo")
            log.discard()
            return
        root = paths.paper_root(PAPER) / "code"
        manifest = paths.manifest_path(PAPER)
        rows, files = [], {}
        for f in codefacts.FACTS:
            key = f"{f.repo}@{f.commit[:7]}/{f.path}"
            if key not in files:
                dest = root / f.repo.replace("/", "_") / f.commit[:7] / f.path
                fetch.download(codefacts.raw_url(f.repo, f.commit, f.path), dest, manifest, key, force=force)
                files[key] = (dest, hashlib.sha256(dest.read_bytes()).hexdigest())
            dest, sha = files[key]
            text = dest.read_text()
            rows.append(
                {
                    "fact_id": f.fact_id,
                    "repo": f.repo,
                    "commit": f.commit,
                    "path": f.path,
                    "sha256": sha,
                    "line_start": f.start,
                    "line_end": f.end,
                    "pattern_found": codefacts.check(f, text),
                    "note": f.note,
                    "text": codefacts.extract(text, f.start, f.end),
                }
            )
        df = pd.DataFrame(rows)
        if not df["pattern_found"].all():
            log.add("failed_facts", df.loc[~df["pattern_found"], "fact_id"].tolist())
        write_csv(df, out, "01_fetch_code", [codefacts.raw_url(f.repo, f.commit, f.path) for f in codefacts.FACTS])
        log.count("facts", len(df))
        log.count("facts_pattern_found", int(df["pattern_found"].sum()))
        log.count("files", len(files))
        if not df["pattern_found"].all():
            raise SystemExit("some code facts did not match at their pinned lines: " + ",".join(df.loc[~df["pattern_found"], "fact_id"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
