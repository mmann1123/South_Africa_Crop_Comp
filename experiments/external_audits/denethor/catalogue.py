"""Source Cooperative object keys and URLs for the DENETHOR / AI4Food bucket."""
from __future__ import annotations

import csv
from functools import lru_cache

from experiments.external_audits.common import paths

BASE_URL = "https://data.source.coop/esa/"
PREFIX = "fusion-competition/"
LISTING = paths.NOTES_DIR / "recon_fusion_competition_s3_listing.tsv"

S2_TILE_YEARS = [
    "s2-utm-33N-18E-242N-2018",  # DENETHOR train tile-year
    "s2-utm-33N-17E-243N-2018",
    "s2-utm-33N-18E-242N-2019",
    "s2-utm-33N-17E-243N-2019",  # DENETHOR test tile-year
]
SA_TILE_YEARS = [
    "s2-utm-34S-19E-258N-2017",
    "s2-utm-34S-19E-259N-2017",
    "s2-utm-34S-20E-259N-2017",
]
BENCHMARK = {("18E-242N", 2018): "train", ("17E-243N", 2019): "test"}
LABELS = {
    "18E-242N": PREFIX + "br-18E-242N-crop-labels-train-2018.geojson",
    "17E-243N": PREFIX + "br-17E-243N-crop-labels-test-2019.geojson",
    "19E-258N": PREFIX + "sa-19E-258N-crop-labels-train-2017.geojson",
    "19E-259N": PREFIX + "sa-19E-259N-crop-labels-train-2017.geojson",
    "20E-259N": PREFIX + "sa-20E-259N-crop-labels-test-2017.geojson",
}
LABEL_YEAR = {"18E-242N": 2018, "17E-243N": 2019, "19E-258N": 2017, "19E-259N": 2017, "20E-259N": 2017}


def url(key: str) -> str:
    return BASE_URL + key


def s2_keys(tile_year: str) -> dict[str, str]:
    base = f"{PREFIX}sentinel-2/{tile_year}/"
    return {
        "CLP": base + "data/CLP.npy",
        "SCL": base + "mask/SCL.npy",
        "IS_DATA": base + "mask/IS_DATA.npy",
        "timestamp": base + "timestamp.pkl",
        "bbox": base + "bbox.pkl",
        "meta_info": base + "meta_info.pkl",
    }


def tile_of(tile_year: str) -> str:
    parts = tile_year.split("-")  # s2, utm, 33N, 18E, 242N, 2018
    return f"{parts[3]}-{parts[4]}"


def year_of(tile_year: str) -> int:
    return int(tile_year.split("-")[-1])


def utm_zone_of(tile_year: str) -> str:
    return tile_year.split("-")[2]  # 33N or 34S


def pfqa_key(tile: str, date: str, zone: str = "33N") -> str:
    return f"{PREFIX}planet/UTM-24000/{zone}/{tile}/PF-QA/{date}.tif"


@lru_cache(maxsize=1)
def _listing() -> dict[str, int]:
    out = {}
    with open(LISTING, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("key"):
                out[row["key"]] = int(row["size_bytes"])
    return out


def listing_size(key: str) -> int:
    return _listing()[key]
