"""Parse patch names from the COCO split files and describe the splits."""
from __future__ import annotations

import re

import pandas as pd

PATCH_RE = re.compile(r"(?P<stem>(?P<year>\d{4})_(?P<tile>\d{2}[A-Z]{3})_patch_(?P<row>\d+)_(?P<col>\d+))\.nc$")
CATALONIA = {"31TBF", "31TCF", "31TCG", "31TDF", "31TDG"}
FRANCE = {"31TCJ", "31TCL", "31TDK", "31TDM", "31UCP", "31UDR"}


def country_of(tile: str) -> str:
    if tile in CATALONIA:
        return "Catalonia"
    if tile in FRANCE:
        return "France"
    raise ValueError(f"unknown tile {tile}")


def parse_images(file_names: list[str]) -> pd.DataFrame:
    rows = []
    for fn in file_names:
        m = PATCH_RE.search(fn)
        if m is None:
            raise ValueError(f"unparseable patch name {fn!r}")
        rows.append({"stem": m["stem"], "year": int(m["year"]), "tile": m["tile"], "row": int(m["row"]), "col": int(m["col"]), "country": country_of(m["tile"])})
    return pd.DataFrame(rows)


def composition(patches: pd.DataFrame) -> pd.DataFrame:
    return (
        patches.groupby(["scenario", "split", "country", "year", "tile"]).size().rename("n_patches").reset_index()
    )


def hf_path(stem: str) -> str:
    year, tile = stem.split("_")[:2]
    return f"data/{year}/{tile}/{stem}.nc"


def sample_per_tile_year(patches: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """n random patches per (scenario, split, country, year, tile) group, or all
    of them when the group is smaller. Deterministic for a given seed."""
    parts = []
    for i, (_, g) in enumerate(patches.groupby(["scenario", "split", "country", "year", "tile"], sort=True)):
        parts.append(g.sample(n=min(n, len(g)), random_state=seed + i))
    return pd.concat(parts).sort_values(["scenario", "split", "tile", "year", "row", "col"]).reset_index(drop=True)


def decode_times(units: str, seconds) -> pd.DatetimeIndex:
    """Decode a CF time axis 'seconds since YYYY-MM-DD HH:MM:SS' (proleptic gregorian)."""
    prefix = "seconds since "
    if not units.startswith(prefix):
        raise ValueError(f"unsupported time units {units!r}")
    base = pd.Timestamp(units[len(prefix):])
    return pd.DatetimeIndex([base + pd.Timedelta(int(s), "s") for s in seconds])


def stac_monthly_counts(features: list[dict], year: int, cloud_max: float = 10.0) -> list[dict]:
    """Per calendar month: granules found, and granules with eo:cloud_cover below
    cloud_max, after de-duplicating by sensing time (reprocessed baselines appear
    twice in the catalogue). Features are Earth Search STAC items."""
    seen = {}
    for f in features:
        p = f["properties"]
        key = p["datetime"][:19]
        if key not in seen:
            seen[key] = p
    out = []
    for m in range(1, 13):
        sel = [p for k, p in seen.items() if k.startswith(f"{year}-{m:02d}-")]
        out.append({"month": m, "n_granules": len(sel), "n_cloud_lt10": sum(1 for p in sel if p["eo:cloud_cover"] < cloud_max)})
    return out
