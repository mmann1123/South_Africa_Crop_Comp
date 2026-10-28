#!/usr/bin/env python
"""Fetch DENETHOR Sentinel-2 cloud layers (CLP, SCL, IS_DATA), timestamps,
georeference and crop labels from Source Cooperative. Never BANDS.npy."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.denethor import catalogue  # noqa: E402

PAPER = "denethor"


def fetch_tile_year(tile_year: str, manifest: Path, log: RunLog, force: bool) -> None:
    root = paths.paper_root(PAPER) / tile_year
    keys = catalogue.s2_keys(tile_year)
    for name, key in keys.items():
        dest = root / Path(key).name
        fetch.download(catalogue.url(key), dest, manifest, key, expected_bytes=catalogue.listing_size(key), force=force)
    with open(root / "timestamp.pkl", "rb") as fh:
        ts = pickle.load(fh)
    shapes = {}
    for name in ("CLP", "SCL", "IS_DATA"):
        arr = np.load(root / f"{name}.npy", mmap_mode="r")
        shapes[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.shape[0] != len(ts):
            raise ValueError(f"{tile_year} {name}: T={arr.shape[0]} but {len(ts)} timestamps")
    log.add(tile_year, {"n_timestamps": len(ts), "first": str(ts[0]), "last": str(ts[-1]), **shapes})


def fetch_labels(tile: str, manifest: Path, force: bool) -> Path:
    key = catalogue.LABELS[tile]
    dest = paths.paper_root(PAPER) / "labels" / f"{tile}.geojson"
    return fetch.download(catalogue.url(key), dest, manifest, key, expected_bytes=catalogue.listing_size(key), force=force)


def main(sa: bool = False, force: bool = False) -> None:
    manifest = paths.manifest_path(PAPER)
    tile_years = list(catalogue.S2_TILE_YEARS) + (list(catalogue.SA_TILE_YEARS) if sa else [])
    with RunLog("01_fetch_masks", {"sa": sa, "force": force}, paper=PAPER) as log:
        for ty in tile_years:
            fetch_tile_year(ty, manifest, log, force)
            fetch_labels(catalogue.tile_of(ty), manifest, force)
        log.count("tile_years", len(tile_years))
        log.count("manifest_entries", len(fetch.manifest_entries(manifest)))
        log.add("bytes_on_disk", sum(e["bytes"] for e in fetch.manifest_entries(manifest)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa", action="store_true", help="also fetch the three South Africa 2017 tiles")
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
