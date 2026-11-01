#!/usr/bin/env python
"""Parcel tables and FR1 parcel polygons (WGS84) from the shipped meta/ folders.
Reads metadata.pkl with a restricted unpickler (numpy and builtins only).
Output under the data root: parcels_<tile>.csv (idx, id, label, n_pixels, block),
fr1_parcels_wgs84.parquet (idx, geometry), dates_<tile>.json."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common.runlog import RunLog  # noqa: E402

PAPER = "timematch_rerun"
ROOT = Path("/mnt/bigdrive/external_audits/timematch_rerun")
META = ROOT / "meta_fetch"
TILES = {"DK1": ("denmark", "32VNH"), "FR1": ("france", "30TXT")}


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] in ("numpy", "builtins", "collections", "datetime", "_codecs"):
            return super().find_class(mod, name)
        raise pickle.UnpicklingError(f"blocked {mod}.{name}")


def load_meta(country: str, tile: str) -> dict:
    with open(META / country / tile / "2017" / "meta" / "metadata.pkl", "rb") as f:
        return _U(f).load()


def main() -> None:

    with RunLog("01_prepare_meta", {}, paper=PAPER) as log:
        for lab, (country, tile) in TILES.items():
            m = load_meta(country, tile)
            rows = [
                {"idx": i, "id": str(p["id"]), "label": str(p["label"]), "n_pixels": int(p["n_pixels"]), "block": int(p["block"])}
                for i, p in enumerate(m["parcels"])
            ]
            df = pd.DataFrame(rows)
            df.to_csv(ROOT / f"parcels_{lab}.csv", index=False)
            (ROOT / f"dates_{lab}.json").write_text(json.dumps([int(d) for d in m["dates"]]))
            log.count(f"parcels_{lab}", len(df))
            log.count(f"dates_{lab}", len(m["dates"]))
        import geopandas as gpd

        shp = META / "france" / "30TXT" / "2017" / "meta" / "parcels" / "parcels_france_30TXT_2017.shp"
        g = gpd.read_file(shp)
        df = pd.read_csv(ROOT / "parcels_FR1.csv", dtype={"id": str})
        log.add("fr1_shapefile_order_matches_metadata", bool((g["id_parcel"].astype(str).values == df["id"].values).all()))
        log.add("fr1_block_matches", bool((g["block"].astype(int).values == df["block"].values).all()))
        g = g.set_index(g["id_parcel"].astype(str))
        g = g.loc[df["id"]]  # metadata order == zarr index
        g4326 = g.to_crs(4326)
        out = gpd.GeoDataFrame({"idx": df["idx"].values}, geometry=g4326.geometry.values, crs=4326)
        out.to_parquet(ROOT / "fr1_parcels_wgs84.parquet")
        log.count("fr1_polygons", len(out))
        print(log.data)


if __name__ == "__main__":
    main()
