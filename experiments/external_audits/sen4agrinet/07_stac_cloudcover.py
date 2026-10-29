#!/usr/bin/env python
"""Optional context: granule counts per month and tile-year from the AWS Earth
Search STAC catalogue (sentinel-2-l1c), total and with granule eo:cloud_cover < 10,
for the 16 tile-years of the dataset. This is the scene-level availability the
authors' 10 percent filter acts on; it is compared with the patch-level
presence in 08_summarise.py."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import splits  # noqa: E402

PAPER = "sen4agrinet"
API = "https://earth-search.aws.element84.com/v1/search"
TILE_YEARS = [(t, 2019) for t in sorted(splits.CATALONIA | splits.FRANCE)] + [(t, 2020) for t in sorted(splits.CATALONIA)]


def search_tile_year(tile: str, year: int) -> list[dict]:
    body = {
        "collections": ["sentinel-2-l1c"],
        "limit": 200,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        "query": {"mgrs:utm_zone": {"eq": int(tile[:2])}, "mgrs:latitude_band": {"eq": tile[2]}, "mgrs:grid_square": {"eq": tile[3:]}},
    }
    feats, url = [], API
    while url:
        for attempt in range(5):
            r = requests.post(url, json=body, timeout=120)
            if r.status_code < 500:
                break
            time.sleep(3 * 2**attempt)
        r.raise_for_status()
        d = r.json()
        feats += d["features"]
        nxt = next((link for link in d.get("links", []) if link.get("rel") == "next"), None)
        if nxt:
            url, body = nxt["href"], nxt.get("body", body)
        else:
            url = None
    return feats


def main(force: bool = False) -> None:
    out = paths.results_dir(PAPER) / "stac_granules_by_month.csv"
    with RunLog("07_stac_cloudcover", {"force": force}, paper=PAPER) as log:
        if out.exists() and not force:
            print(f"{out.name} exists; use --force to redo")
            log.discard()
            return
        rows = []
        for tile, year in TILE_YEARS:
            feats = search_tile_year(tile, year)
            log.count(f"granules_{tile}_{year}", len(feats))
            for r in splits.stac_monthly_counts(feats, year):
                rows.append({"country": splits.country_of(tile), "year": year, "tile": tile, **r})
        write_csv(pd.DataFrame(rows), out, "07_stac_cloudcover", [API + " (collection sentinel-2-l1c, one query per tile-year)"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
