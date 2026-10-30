#!/usr/bin/env python
"""Site extents (Ramsar polygons from WDPA, the outlines in the paper's Figure 1), 2021 granule and orbit
inventory per site for both Sentinel-2 L2A collections, and the distinct acquisition dates after the
paper's CLOUDY_PIXEL_PERCENTAGE < 50 filter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.maleki import catalogue as C, eelib  # noqa: E402

RES = paths.results_dir(C.PAPER)
ROOT = paths.paper_root(C.PAPER)


def main(force: bool = False) -> None:
    out = RES / "site_inventory.csv"
    if out.exists() and not force:
        print(f"{out} exists; use --force")
        return
    import ee

    with RunLog("01_site_geometry", {"force": force}, paper=C.PAPER) as log:
        log.add("ee_project", ee_init.init())
        rows = []
        for site in C.SITES:
            geom = eelib.site_geometry(site)
            (ROOT / f"{site}_ramsar.geojson").write_text(json.dumps(geom.getInfo()))
            (minx, miny), _, (maxx, maxy), _, _ = geom.bounds().getInfo()["coordinates"][0]
            area = geom.area().divide(1e6).getInfo()
            for product in (C.COLLECTION, C.COLLECTION_ALT):
                allc = ee.ImageCollection(product).filterBounds(geom).filterDate(f"{C.YEAR}-01-01", f"{C.YEAR + 1}-01-01")
                col = eelib.collection(geom, product)
                ts = col.aggregate_array("system:time_start").getInfo()
                dates = sorted({pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d") for t in ts})
                rows.append({
                    "site": site, "product": product, "wdpa_name": C.SITES[site], "area_km2": round(area, 1),
                    "lon_min": round(minx, 3), "lat_min": round(miny, 3), "lon_max": round(maxx, 3), "lat_max": round(maxy, 3),
                    "granules_all": allc.size().getInfo(), "granules_cloudy_lt50": col.size().getInfo(),
                    "dates_cloudy_lt50": len(dates),
                    "mgrs_tiles": " ".join(sorted(allc.aggregate_array("MGRS_TILE").distinct().getInfo())),
                    "orbits": " ".join(map(str, sorted(allc.aggregate_array("SENSING_ORBIT_NUMBER").distinct().getInfo()))),
                })
                if product == C.COLLECTION:
                    (ROOT / f"{site}_dates.json").write_text(json.dumps(dates))
                print(rows[-1], flush=True)
        write_csv(pd.DataFrame(rows), out, "01_site_geometry", [C.WDPA, C.COLLECTION, C.COLLECTION_ALT])
        log.count("rows", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
