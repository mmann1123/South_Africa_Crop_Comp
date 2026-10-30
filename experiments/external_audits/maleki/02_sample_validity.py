#!/usr/bin/env python
"""Per-pixel, per-acquisition-date validity at random points inside each site polygon: coverage, validity under the
paper's masks (cloudy-granule filter, MSK_CLDPRB < 1, MSK_SNWPRB < 10, SCL 2/3/8/9/10) and under SCL alone,
plus the ESA WorldCover 2021 class. Chunks are cached so an interrupted run resumes."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, fetch, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.maleki import catalogue as C, eelib  # noqa: E402

ROOT = paths.paper_root(C.PAPER)


def points(site: str, geom, n: int) -> np.ndarray:
    f = ROOT / f"{site}_points.npy"
    if f.exists():
        return np.load(f)
    import ee

    fc = ee.FeatureCollection.randomPoints(geom, n, C.SEED)
    xy = np.array([ft["geometry"]["coordinates"] for ft in fc.toList(n).getInfo()], dtype=float)
    np.save(f, xy)
    return xy


def sample_chunk(stack, xy: np.ndarray, n_dates: int) -> np.ndarray:
    """(len(xy), 3 * n_dates + 1) int array: c, v, s per date then the WorldCover class (0 if missing)."""
    import ee

    feats = [ee.Feature(ee.Geometry.Point(list(p)), {"i": i}) for i, p in enumerate(xy)]
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map").rename("wc")
    res = stack.addBands(wc).sampleRegions(ee.FeatureCollection(feats), ["i"], C.SCALE, tileScale=4).getInfo()["features"]
    names = [f"{k}{i}" for i in range(n_dates) for k in "cvs"] + ["wc"]
    out = np.full((len(xy), len(names)), -1, dtype=np.int16)
    for ft in res:
        p = ft["properties"]
        out[int(p["i"])] = [p.get(k, -1) if p.get(k) is not None else -1 for k in names]
    return out


def main(sites: list[str], n: int, force: bool) -> None:
    with RunLog("02_sample_validity", {"sites": sites, "n": n, "force": force}, paper=C.PAPER) as log:
        log.add("ee_project", ee_init.init())
        for site in sites:
            final = ROOT / f"{site}_validity.npz"
            if final.exists() and not force:
                print(f"{final} exists; use --force")
                continue
            geom = eelib.site_geometry(site)
            dates = json.loads((ROOT / f"{site}_dates.json").read_text())
            col = eelib.collection(geom)
            stack = eelib.per_date_stack(col, dates)
            xy = points(site, geom, n)
            parts = []
            for k in range(0, len(xy), C.CHUNK):
                part = ROOT / f"{site}_chunk_{k:05d}.npy"
                if not part.exists():
                    np.save(part, sample_chunk(stack, xy[k : k + C.CHUNK], len(dates)))
                parts.append(np.load(part))
                print(site, k, flush=True)
            arr = np.concatenate(parts)
            doy = np.array([dt.date.fromisoformat(d).timetuple().tm_yday for d in dates])
            np.savez_compressed(final, arr=arr, xy=xy, dates=np.array(dates), doy=doy)
            for p in ROOT.glob(f"{site}_chunk_*.npy"):
                p.unlink()
            man = paths.manifest_path(C.PAPER)
            entries = [e for e in fetch.manifest_entries(man) if e["key"] != final.name]
            entries.append({"key": final.name, "url": f"Earth Engine {C.COLLECTION} sampleRegions, {n} random points in {C.WDPA} Ramsar polygon {C.SITES[site]}", "bytes": final.stat().st_size, "sha256": hashlib.sha256(final.read_bytes()).hexdigest(), "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            man.write_text(json.dumps(entries, indent=2))
            log.count(f"{site}_points", len(xy))
            log.count(f"{site}_dates", len(dates))
            log.count(f"{site}_missing_rows", int((arr[:, -1] < 0).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=list(C.SITES))
    ap.add_argument("--n", type=int, default=C.N_POINTS)
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
