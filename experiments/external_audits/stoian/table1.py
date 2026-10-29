"""Parse Table 1 (class areas in hectares per tile) of Stoian et al. 2019 from the page text."""
from __future__ import annotations

import re

import pandas as pd

CLASSES = {
    "ASC": "Annual summer crops", "AWC": "Annual winter crops", "BLF": "Broad-leaved forest",
    "COF": "Coniferous forest", "NGL": "Natural grasslands", "WOM": "Woody moorlands",
    "CUF": "Continuous urban fabric", "DUF": "Discontinuous urban fabric", "ICU": "Industrial or commercial units",
    "RSF": "Road surfaces", "BRO": "Bare rock", "BDS": "Beaches, dunes and sand plains",
    "WAT": "Water bodies", "GPS": "Glaciers and perpetual snow", "IGL": "Intensive grassland",
    "ORC": "Orchards", "VIN": "Vineyards",
}
_NUM = re.compile(r"^\d{1,3}(,\d{3})*$|^\d+$")


def _to_int(tok: str) -> int:
    return int(tok.replace(",", ""))


def parse_table1(text: str, tiles: list[str]) -> pd.DataFrame:
    """Rows = class codes, columns = tiles plus 'total_ha' and 'total_pct' as printed.

    `text` is the page text with one token per line (PyMuPDF get_text order): the
    header of tile names, then for every class its code, one area per tile and the
    printed total with its percentage; a final 'Total' row follows.
    """
    toks = [t.strip() for t in text.splitlines() if t.strip()]
    start = toks.index(tiles[0])
    pos = start + len(tiles)
    assert toks[pos] == "Total", f"expected Total header, got {toks[pos]!r}"
    pos += 1
    rows = []
    while pos < len(toks) and toks[pos] in CLASSES:
        code = toks[pos]
        vals = []
        for k in range(len(tiles)):
            if not _NUM.match(toks[pos + 1 + k]):
                raise ValueError(f"{code}: bad number {toks[pos + 1 + k]!r}")
            vals.append(_to_int(toks[pos + 1 + k]))
        tot = toks[pos + 1 + len(tiles)]
        m = re.match(r"^([\d,]+)(?: \(([\d.]+)%\))?$", tot)
        if not m:
            raise ValueError(f"{code}: bad total {tot!r}")
        rows.append({"class": code, **dict(zip(tiles, vals)), "total_ha": _to_int(m.group(1)), "total_pct": float(m.group(2)) if m.group(2) else None})
        pos += 2 + len(tiles)
    df = pd.DataFrame(rows)
    if toks[pos] == "Total":
        printed = [_to_int(t) for t in toks[pos + 1 : pos + 1 + len(tiles) + 1]]
        df.attrs["printed_column_totals"] = dict(zip(tiles + ["total_ha"], printed))
    return df


def check_totals(df: pd.DataFrame, tiles: list[str]) -> tuple[list[str], dict[str, int]]:
    """Compare printed totals with recomputed sums.

    Returns (row problems, column differences printed minus recomputed). A row problem
    means a transcription error; the column differences are the paper's own and are
    reported, not enforced.
    """
    rows = []
    for _, r in df.iterrows():
        if sum(r[t] for t in tiles) != r["total_ha"]:
            rows.append(f"row {r['class']}: sum {sum(r[t] for t in tiles)} != printed {r['total_ha']}")
    cols = {name: int(printed - df[name].sum()) for name, printed in df.attrs.get("printed_column_totals", {}).items()}
    return rows, cols
