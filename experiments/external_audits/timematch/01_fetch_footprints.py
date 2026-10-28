#!/usr/bin/env python
"""Fetch TimeMatch block and parcel shapefiles out of the Zenodo zip by HTTP
range requests (no 74 GB download) and write the shipped acquisition dates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.timematch import zipindex  # noqa: E402

PAPER = "timematch"
ZIP_URL = "https://zenodo.org/records/5636422/files/timematch_data.zip?download=1"
EXPECTED_ZIP_BYTES = 73_904_625_097
RECON_JSON = paths.NOTES_DIR / "recon_timematch_metadata_summary.json"
TILES = {  # country, tile -> short label used in the paper
    ("austria", "33UVP"): "AT1",
    ("denmark", "32VNH"): "DK1",
    ("france", "30TXT"): "FR1",
    ("france", "31TCJ"): "FR2",
}


def write_dates(log: RunLog) -> Path:
    d = json.loads(RECON_JSON.read_text())
    rows = []
    for key, meta in d.items():
        _, country, tile, year, *_ = key.split("/")
        for date, pct in zip(meta["dates"], meta["cloudy_pct"]):
            rows.append(
                {
                    "country": country,
                    "tile": tile,
                    "label": TILES[(country, tile)],
                    "date": pd.to_datetime(str(date), format="%Y%m%d").strftime("%Y-%m-%d"),
                    "tile_cloudy_pct": pct,
                }
            )
        log.count(f"dates_{TILES[(country, tile)]}", meta["n_dates"])
    df = pd.DataFrame(rows).sort_values(["country", "tile", "date"]).reset_index(drop=True)
    return write_csv(df, paths.results_dir(PAPER) / "timematch_dates.csv", "01_fetch_footprints", [str(RECON_JSON)])


def wanted(name: str) -> bool:
    return ("/meta/blocks/" in name or "/meta/parcels/" in name) and not name.endswith("/")


def main(force: bool = False) -> None:
    root = paths.paper_root(PAPER)
    manifest = paths.manifest_path(PAPER)
    cache = root / "central_directory.json"
    with RunLog("01_fetch_footprints", {"force": force}, paper=PAPER) as log:
        write_dates(log)
        read = lambda s, l: fetch.range_get(ZIP_URL, s, l)  # noqa: E731
        if cache.exists() and not force:
            members = [zipindex.ZipMember(**m) for m in json.loads(cache.read_text())["members"]]
            total = json.loads(cache.read_text())["zip_bytes"]
        else:
            total = fetch.head_size(ZIP_URL)
            log.add("zip_bytes", total)
            if total != EXPECTED_ZIP_BYTES:
                log.add("warning", f"zip size {total} differs from recon {EXPECTED_ZIP_BYTES}")
            all_members = zipindex.read_central_directory(read, total)
            log.count("zip_members_total", len(all_members))
            members = [m for m in all_members if wanted(m.name)]
            cache.write_text(json.dumps({"zip_url": ZIP_URL, "zip_bytes": total, "members": [m.__dict__ for m in members]}, indent=1))
        log.count("footprint_members", len(members))
        fetched = 0
        for m in members:
            # timematch_data/<country>/<tile>/<year>/meta/blocks/blocks_x.shp -> footprints/<country>_<tile>/blocks.shp
            parts = m.name.split("/")
            country, tile = parts[1], parts[2]
            kind = parts[5]  # blocks or parcels
            ext = Path(parts[-1]).suffix
            dest = root / "footprints" / f"{country}_{tile}" / f"{kind}{ext}"
            if dest.exists() and dest.stat().st_size == m.uncompressed_size and not force:
                continue
            zipindex.extract_member(read, m, dest)
            fetched += 1
            entries = fetch.manifest_entries(manifest)
            entries = [e for e in entries if e["key"] != m.name] + [
                {"key": m.name, "url": ZIP_URL, "bytes": m.uncompressed_size, "sha256": None, "fetched_at": pd.Timestamp.utcnow().isoformat(), "zip_header_offset": m.header_offset}
            ]
            manifest.write_text(json.dumps(entries, indent=2))
        log.count("footprint_members_fetched", fetched)
        import geopandas as gpd

        for (country, tile), label in TILES.items():
            for kind in ("blocks", "parcels"):
                shp = root / "footprints" / f"{country}_{tile}" / f"{kind}.shp"
                if shp.exists():
                    g = gpd.read_file(shp)
                    log.add(f"{label}_{kind}", {"n": int(len(g)), "crs": str(g.crs), "bounds": [float(b) for b in g.total_bounds]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
