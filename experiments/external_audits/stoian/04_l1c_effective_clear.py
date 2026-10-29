#!/usr/bin/env python
"""L1C granule effective clear count per tile for 2016 from the Copernicus Data Space
OData catalogue: sum over granules of (1 - cloudCover/100) x footprint share of the tile."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.stoian import catalogue, l1c  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)


def query_tile(tile: str) -> list[dict]:
    flt = (
        f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') and contains(Name,'_{tile}_') "
        f"and ContentDate/Start ge {catalogue.YEAR}-01-01T00:00:00.000Z and ContentDate/Start lt {catalogue.YEAR + 1}-01-01T00:00:00.000Z"
    )
    params = {"$filter": flt, "$top": 1000, "$expand": "Attributes", "$orderby": "ContentDate/Start"}
    out: list[dict] = []
    url, p = catalogue.CDSE_ODATA, params
    while url:
        r = requests.get(url, params=p, timeout=180)
        r.raise_for_status()
        j = r.json()
        out += j["value"]
        url, p = j.get("@odata.nextLink"), None
    return out


def main(force: bool = False) -> None:
    out = RES / "l1c_tile_summary.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    scl = pd.read_csv(RES / "scl_tile_summary.csv").set_index("tile")
    with RunLog("04_l1c_effective_clear", {"force": force}, paper=PAPER) as log:
        summary, granules = [], []
        for tile in catalogue.TILES:
            prods = query_tile(tile)
            raw = ROOT / f"cdse_{tile}.json"
            raw.write_text(json.dumps(prods))
            b = [float(scl.loc[tile, c]) for c in ("bounds_left", "bounds_bottom", "bounds_right", "bounds_top")]
            epsg = int(scl.loc[tile, "epsg"])
            rows = []
            for p in prods:
                attrs = {a["Name"]: a["Value"] for a in p["Attributes"]}
                rows.append(
                    {
                        "tile": tile,
                        "name": p["Name"],
                        "date": p["ContentDate"]["Start"][:10],
                        "processing_baseline": p["Name"].split("_")[3],
                        "cloud_cover": float(attrs["cloudCover"]),
                        "share": l1c.footprint_share(l1c.parse_footprint(p["Footprint"]), tuple(b), epsg),
                    }
                )
            g = pd.DataFrame(rows)
            granules.append(g)
            y = (f"{catalogue.YEAR}-01-01", f"{catalogue.YEAR}-12-31")
            s = (f"{catalogue.YEAR}-{catalogue.SEASON[0]}", f"{catalogue.YEAR}-{catalogue.SEASON[1]}")
            summary.append(
                {
                    "tile": tile,
                    "n_granules": len(g),
                    "n_dates": int(g["date"].nunique()),
                    "baselines": ";".join(sorted(g["processing_baseline"].unique())),
                    "mean_cloud_cover_pct": round(g["cloud_cover"].mean(), 2),
                    "l1c_eff_clear_year": round(l1c.effective_clear(g, *y), 3),
                    "l1c_eff_clear_marOct": round(l1c.effective_clear(g, *s), 3),
                }
            )
            sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            man = paths.manifest_path(PAPER)
            entries = [e for e in fetch.manifest_entries(man) if e["key"] != raw.name]
            entries.append({"key": raw.name, "url": f"{catalogue.CDSE_ODATA} (SENTINEL-2 MSIL1C, {tile}, 2016)", "bytes": raw.stat().st_size, "sha256": sha, "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            man.write_text(json.dumps(entries, indent=2))
            print(summary[-1], flush=True)
        write_csv(pd.DataFrame(summary), out, "04_l1c_effective_clear", [catalogue.CDSE_ODATA, str(RES / "scl_tile_summary.csv")])
        write_csv(pd.concat(granules, ignore_index=True), RES / "l1c_granules.csv", "04_l1c_effective_clear", [catalogue.CDSE_ODATA])
        log.count("granules", int(sum(len(g) for g in granules)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
