#!/usr/bin/env python
"""Per-pixel 2016 clear-observation counts per tile from Microsoft Planetary Computer
sentinel-2-l2a SCL, read at about 320 m (344 x 344 grid, the 1/16 overview), same-date
granules merged, summarised over land pixels."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.stoian import catalogue, sclstats  # noqa: E402

PAPER = catalogue.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
THREADS = 8


def search_items(tile: str) -> list[dict]:
    """All 2016 sentinel-2-l2a items of one MGRS tile (follows the POST 'next' link)."""
    body = {"collections": ["sentinel-2-l2a"], "query": {"s2:mgrs_tile": {"eq": tile[1:]}}, "datetime": f"{catalogue.YEAR}-01-01T00:00:00Z/{catalogue.YEAR}-12-31T23:59:59Z", "limit": 200}
    items: list[dict] = []
    while True:
        r = requests.post(catalogue.PC_STAC, json=body, timeout=120)
        r.raise_for_status()
        j = r.json()
        items += j["features"]
        nxt = next((l for l in j.get("links", []) if l["rel"] == "next"), None)
        if not nxt:
            return items
        body = nxt["body"]


def get_token() -> str:
    r = requests.get(catalogue.PC_TOKEN, timeout=60)
    r.raise_for_status()
    return r.json()["token"]


def read_scl(href: str, token: str) -> tuple[np.ndarray, tuple, str]:
    last = None
    for attempt in range(5):
        try:
            with rasterio.open(f"{href}?{token}") as ds:
                a = ds.read(1, out_shape=(catalogue.GRID, catalogue.GRID), resampling=Resampling.nearest)
                return a, tuple(ds.bounds), str(ds.crs)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * 2**attempt)
    raise RuntimeError(f"could not read {href}: {last}")


def fetch_tile(tile: str, log: RunLog, force: bool) -> Path:
    npz = ROOT / f"scl_{tile}.npz"
    if npz.exists() and not force:
        return npz
    items = search_items(tile)
    (ROOT / f"items_{tile}.json").write_text(json.dumps([{"id": i["id"], "datetime": i["properties"]["datetime"], "href": i["assets"]["SCL"]["href"]} for i in items], indent=1))
    by_date: dict[str, list[dict]] = {}
    for it in items:
        by_date.setdefault(it["properties"]["datetime"][:10], []).append(it)
    token = get_token()
    jobs = [(d, it) for d, its in sorted(by_date.items()) for it in its]
    with ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(lambda j: read_scl(j[1]["assets"]["SCL"]["href"], token), jobs))
    bounds = {r[1] for r in res}
    crs = {r[2] for r in res}
    if len(bounds) != 1 or len(crs) != 1:
        raise RuntimeError(f"{tile}: granule grids differ: {bounds} {crs}")
    lookup = {id(j[1]): r[0] for j, r in zip(jobs, res)}
    dates = sorted(by_date)
    stack = np.stack([sclstats.merge_same_date([lookup[id(it)] for it in by_date[d]]) for d in dates])
    np.savez_compressed(npz, scl=stack, dates=np.array(dates), bounds=np.array(next(iter(bounds))), crs=np.array(next(iter(crs))))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    man = paths.manifest_path(PAPER)
    entries = [e for e in fetch.manifest_entries(man) if e["key"] != npz.name]
    entries.append({"key": npz.name, "url": f"{catalogue.PC_STAC} (sentinel-2-l2a, s2:mgrs_tile={tile[1:]}, 2016, asset SCL, 1/16 overview)", "bytes": npz.stat().st_size, "sha256": sha, "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    man.write_text(json.dumps(entries, indent=2))
    log.add(f"{tile}_fetch", {"items": len(items), "dates": len(dates), "multi_granule_dates": sum(len(v) > 1 for v in by_date.values()), "bounds": list(next(iter(bounds))), "crs": next(iter(crs))})
    return npz


FRA_URL = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/FRA/ADM0/geoBoundaries-FRA-ADM0.geojson"


def french_mask(npz: Path) -> np.ndarray:
    """Boolean (H, W) mask of grid cells inside France (geoBoundaries gbOpen FRA ADM0, pinned commit)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    z = np.load(npz, allow_pickle=False)
    fra = fetch.download(FRA_URL, ROOT / "geoBoundaries-FRA-ADM0.geojson", paths.manifest_path(PAPER), "geoBoundaries-FRA-ADM0.geojson")
    g = gpd.read_file(fra).to_crs(str(z["crs"]))
    l, b, r, t = [float(v) for v in z["bounds"]]
    return rasterize(((geom, 1) for geom in g.geometry), out_shape=(catalogue.GRID, catalogue.GRID), transform=from_bounds(l, b, r, t, catalogue.GRID, catalogue.GRID), fill=0, dtype="uint8").astype(bool)


def summarise_tile(tile: str, npz: Path, region: np.ndarray | None = None) -> dict:
    z = np.load(npz, allow_pickle=False)
    return sclstats.tile_stats(z["scl"], z["dates"], (f"{catalogue.YEAR}-{catalogue.SEASON[0]}", f"{catalogue.YEAR}-{catalogue.SEASON[1]}"), region)


def main(tiles: list[str] | None = None, force: bool = False, refetch: bool = False) -> None:
    tiles = tiles or catalogue.TILES
    out = RES / "scl_tile_summary.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    with RunLog("03_scl_clear_counts", {"tiles": tiles, "force": force}, paper=PAPER) as log:
        rows, drows, french = [], [], []
        for tile in tiles:
            npz = fetch_tile(tile, log, force=refetch)
            z = np.load(npz, allow_pickle=False)
            st = summarise_tile(tile, npz)
            l, b, r, t = [float(v) for v in z["bounds"]]
            rows.append({"tile": tile, **st, "epsg": int(str(z["crs"]).split(":")[-1]), "bounds_left": l, "bounds_bottom": b, "bounds_right": r, "bounds_top": t})
            if tile == "T30TYN":
                fr = french_mask(npz)
                french.append({"tile": "T30TYN", "region": "French part (geoBoundaries FRA ADM0)", **summarise_tile(tile, npz, fr)})
            land = sclstats.land_mask(z["scl"])
            clear = sclstats.clear_mask(z["scl"])
            for d, c, o in zip(z["dates"], clear.sum((1, 2)), (z["scl"] != 0).sum((1, 2))):
                drows.append({"tile": tile, "date": str(d), "clear_pixels": int(c), "observed_pixels": int(o)})
            print(tile, {k: round(v, 1) for k, v in st.items()}, flush=True)
        write_csv(pd.DataFrame(rows), out, "03_scl_clear_counts", [catalogue.PC_STAC, str(paths.manifest_path(PAPER))])
        if french:
            write_csv(pd.DataFrame(french), RES / "scl_t30tyn_french_part.csv", "03_scl_clear_counts", [FRA_URL, str(paths.manifest_path(PAPER))])
        write_csv(pd.DataFrame(drows), RES / "scl_dates.csv", "03_scl_clear_counts", [catalogue.PC_STAC])
        log.count("tiles", len(rows))
        log.add("clear_classes", list(catalogue.SCL_CLEAR))
        log.add("grid", catalogue.GRID)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild the summary tables from the cached SCL stacks")
    ap.add_argument("--refetch", action="store_true", help="also re-read all tiles from Planetary Computer")
    main(**vars(ap.parse_args()))
