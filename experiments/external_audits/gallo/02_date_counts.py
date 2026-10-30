#!/usr/bin/env python
"""Images per patch folder and area-year from the Kaggle listing: the series the authors' loader sees
before its length harmonisation to 71 (zero padding below, random subsampling above)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.gallo import gallolib  # noqa: E402

PAPER = gallolib.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)


def main(force: bool = False) -> None:
    out = RES / "date_counts_by_area_year.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("02_date_counts", {"force": force}, paper=PAPER) as log:
        dates: dict = defaultdict(lambda: defaultdict(set))
        labels = defaultdict(set)
        n_lines = 0
        for line in open(ROOT / "listing.tsv"):
            n_lines += 1
            p = gallolib.parse_name(line.split("\t")[0])
            if p is None:
                continue
            key = (p["area"], p["year"])
            if p["kind"] == "image":
                dates[key][p["patch"]].add(p["date"])
            else:
                labels[key].add(p["patch"])
        rows, drows = [], []
        for (area, year), per_patch in sorted(dates.items()):
            st = gallolib.date_stats({k: sorted(v) for k, v in per_patch.items()})
            union = sorted(set().union(*per_patch.values()))
            common = set.intersection(*per_patch.values())
            rows.append({"area": area, "year": year, **st, "dates_union": len(union), "dates_in_all_patches": len(common), "patches_with_label": len(labels[(area, year)])})
            cnt = pd.Series([d for s in per_patch.values() for d in s]).value_counts()
            drows += [{"area": area, "year": year, "date": d, "patches_with_date": int(cnt.get(d, 0)), "n_patches": st["n_patches"]} for d in union]
        write_csv(pd.DataFrame(rows), out, "02_date_counts", [str(ROOT / "listing.tsv")])
        write_csv(pd.DataFrame(drows), RES / "dates_by_area_year.csv", "02_date_counts", [str(ROOT / "listing.tsv")])
        h = hashlib.sha256()
        with open(ROOT / "listing.tsv", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        man = paths.manifest_path(PAPER)
        entries = [e for e in fetch.manifest_entries(man) if e["key"] != "listing.tsv"]
        entries.append({"key": "listing.tsv", "url": "https://www.kaggle.com/api/v1/datasets/list/ignazio/sentinel2-crop-mapping (all pages, pageSize=1000)", "bytes": (ROOT / "listing.tsv").stat().st_size, "sha256": h.hexdigest(), "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        man.write_text(json.dumps(entries, indent=2))
        log.count("listing_lines", n_lines)
        log.count("area_years", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
