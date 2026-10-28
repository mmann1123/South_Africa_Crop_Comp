#!/usr/bin/env python
"""Print the six DACCN regions, their areas and sampling frame labels (sanity check)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common.ee_init import init  # noqa: E402
from experiments.external_audits.daccn import regions  # noqa: E402


def main() -> None:
    init()
    for name, country in regions.REGIONS:
        f = regions.lookup(name, country)
        area_km2 = f.geometry().area(1000).divide(1e6).getInfo()
        print(f"{name:14s} {country:26s} {area_km2:10.0f} km2  frame={regions.frame_label(country)}")


if __name__ == "__main__":
    main()
