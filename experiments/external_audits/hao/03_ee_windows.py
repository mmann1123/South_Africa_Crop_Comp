#!/usr/bin/env python
"""Per sampled point, per 15-day window (DOY 60 to 330), the number of observations that pass each
quality rule, from Earth Engine. Two streams:
  hls   HLS v2.0 L30 + S30 (the paper's CONUS 2017, Alberta 2018 and Nebraska 2018 stream). Counts:
        obs (reflectance present), fm (Fmask cirrus, cloud, adjacent and shadow bits all clear),
        fmsnow (fm and not snow/ice).
  l8s2  Landsat-8 C02 L2 + Sentinel-2 L1C (the paper's Hengshui stream). Counts: l8_obs, l8_clear
        (QA_PIXEL dilated cloud, cloud, shadow clear), s2_obs, s2_qa60 (L1C opaque and cirrus
        bits clear: the L1C cloud mask), s2_cs65 and s2_cs80 (Cloud Score+ cs_cdf >= 0.65, 0.80).
A window is non-empty if a count is above zero. Points are processed in spatial chunks and cached
per chunk under the external root, so the run is resumable."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import ee_init, paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.hao import catalogue as cat, haolib  # noqa: E402

PAPER = cat.PAPER
RES = paths.results_dir(PAPER)
ROOT = paths.paper_root(PAPER)
WORKERS = 8
SCALE_M = 30
CRS = cat.GRID_CRS
DOMAIN_YEAR = {"CONUS": cat.TRAIN_YEAR, "NE": cat.TARGET_YEAR, "AB": cat.TARGET_YEAR, "HS": cat.TARGET_YEAR}
# "CONUS@2018" is the CONUS pool read in 2018, "NE@2017" and "AB@2017" the target boxes read in 2017: year controls
JOBS = {"hls": ["CONUS", "NE", "AB", "CONUS@2018", "NE@2017", "AB@2017"], "l8s2": ["HS", "NE", "AB", "CONUS"]}
CONTROLS = {"CONUS@2018", "NE@2017", "AB@2017"}
BANDS = {"hls": ["obs", "fm", "fmsnow"], "l8s2": ["l8_obs", "l8_clear", "s2_obs", "s2_qa60", "s2_cs65", "s2_cs80"]}


def zone_filter(zones):
    import ee

    return ee.Filter.Or(*[ee.Filter.stringStartsWith("system:index", f"T{z:02d}") for z in zones])


def hls_collection(geom, year, zones):
    import ee

    def per_image(img):
        inside = img.select("B4").mask().rename("obs")
        fm = img.select("Fmask").unmask(255)
        clear = fm.bitwiseAnd(cat.FMASK_CLOUD_BITS).eq(0)
        snow = fm.bitwiseAnd(cat.FMASK_SNOW_BIT).neq(0)
        return ee.Image.cat([inside, inside.And(clear).rename("fm"), inside.And(clear).And(snow.Not()).rename("fmsnow")]).toInt8().copyProperties(img, ["system:time_start"])

    parts = [ee.ImageCollection(cid).filterBounds(geom).filterDate(f"{year}-03-01", f"{year}-12-01").filter(zone_filter(zones)) for cid in cat.HLS.values()]
    return parts[0].merge(parts[1]).map(per_image)


def l8s2_collection(geom, year, with_csp: bool):
    import ee

    l8 = ee.ImageCollection(cat.L8).filterBounds(geom).filterDate(f"{year}-03-01", f"{year}-12-01")

    def l8_image(img):
        inside = img.select("SR_B4").mask()
        clear = img.select("QA_PIXEL").unmask(0).bitwiseAnd(cat.L8_CLOUD_BITS).eq(0)
        z = inside.multiply(0)
        bands = [inside.rename("l8_obs"), inside.And(clear).rename("l8_clear")] + [z.rename(n) for n in BANDS["l8s2"][2:]]
        return ee.Image.cat(bands).toInt8().copyProperties(img, ["system:time_start"])

    s2 = ee.ImageCollection(cat.S2_L1C).filterBounds(geom).filterDate(f"{year}-03-01", f"{year}-12-01")
    if with_csp:
        s2 = s2.linkCollection(ee.ImageCollection(cat.S2_CSP).filterBounds(geom).filterDate(f"{year}-03-01", f"{year}-12-01"), ["cs_cdf"])

    def s2_image(img):
        inside = img.select("B4").mask()
        qa = img.select("QA60").unmask(0).bitwiseAnd(cat.QA60_BITS).eq(0)
        z = inside.multiply(0)
        if with_csp:
            cs = img.select("cs_cdf").unmask(-1)
            c65, c80 = inside.And(cs.gte(cat.CSP_CLEAR)), inside.And(cs.gte(cat.CSP_STRICT))
        else:
            c65 = c80 = z
        bands = [z.rename("l8_obs"), z.rename("l8_clear"), inside.rename("s2_obs"), inside.And(qa).rename("s2_qa60"), c65.rename("s2_cs65"), c80.rename("s2_cs80")]
        return ee.Image.cat(bands).toInt8().copyProperties(img, ["system:time_start"])

    return l8.map(l8_image).merge(s2.map(s2_image))


def window_image(coll, year, names):
    import ee

    layers = []
    for k, (a, b) in enumerate(cat.window_dates(year)):
        s = coll.filterDate(a.isoformat(), b.isoformat()).sum()
        layers.append(s.rename([f"w{k:02d}_{n}" for n in names]))
    return ee.Image.cat(layers).unmask(0)  # sampleRegions drops a point if any band is masked; an empty window is a count of 0


def domain_year(domain: str) -> tuple[str, int]:
    base, _, y = domain.partition("@")
    return base, int(y) if y else DOMAIN_YEAR[base]


def read_chunk(stream: str, domain: str, chunk: pd.DataFrame, with_csp: bool) -> pd.DataFrame:
    import ee

    _, year = domain_year(domain)
    geom = ee.Geometry.MultiPoint([[float(r.lon), float(r.lat)] for r in chunk.itertuples()])
    zones = sorted(set(haolib.utm_zone(chunk["lon"]).tolist()))
    coll = hls_collection(geom, year, zones) if stream == "hls" else l8s2_collection(geom, year, with_csp)
    img = window_image(coll, year, BANDS[stream])
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"pid": int(r.pid)}) for r in chunk.itertuples()])
    for attempt in range(4):
        try:
            feats = img.sampleRegions(fc, scale=SCALE_M, projection=CRS, tileScale=4).getInfo()["features"]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"chunk failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
                return pd.DataFrame()
            time.sleep(10 * 2**attempt)
    df = pd.DataFrame([{"pid": f["properties"]["pid"], **{k: v for k, v in f["properties"].items() if k != "pid"}} for f in feats])
    if len(df) < len(chunk):
        print(f"warning: {len(chunk) - len(df)} of {len(chunk)} points not returned", flush=True)
    df["n_images"] = int(coll.size().getInfo())
    return df


def main(stream: str, domains: list[str] | None, force: bool = False, smoke: bool = False) -> None:
    pts = pd.read_csv(RES / "points.csv")
    domains = domains or JOBS[stream]
    cache = ROOT / "chunks"
    cache.mkdir(exist_ok=True)
    todo = []
    for d in domains:
        p = pts[pts["domain"] == domain_year(d)[0]]
        if (stream == "l8s2" and d != "HS") or d in CONTROLS:
            p = p[p["pid"] % (len(p) // cat.STREAM_CONTROL_N + 1) == 0] if len(p) > cat.STREAM_CONTROL_N else p
        for c, g in p.groupby("chunk"):
            f = cache / f"{stream}_{d.replace('@', '-')}_{int(c):04d}.csv.gz"
            if force or not f.exists():
                todo.append((d, f, g))
    if not todo:
        print("nothing to do; use --force")
        return
    if smoke:
        todo = todo[:1]
    with RunLog(f"03_ee_windows_{stream}", {"stream": stream, "domains": domains, "force": force, "smoke": smoke}, paper=PAPER) as log:
        log.add("ee_project", ee_init.init())
        log.add("collections", {"hls": cat.HLS, "l8": cat.L8, "s2_l1c": cat.S2_L1C, "csp": cat.S2_CSP})
        log.add("design", {"scale_m": SCALE_M, "crs": CRS, "windows": [[a.isoformat(), b.isoformat()] for a, b in cat.window_dates(cat.TARGET_YEAR)], "bands": BANDS[stream]})

        def work(job):
            d, f, g = job
            t = time.time()
            df = read_chunk(stream, d, g, with_csp=(domain_year(d)[0] != "CONUS"))
            if len(df):
                if smoke:
                    print(df.head().T.to_string())
                else:
                    df.to_csv(f, index=False)
            print(f"{stream} {d} {f.name}: {len(g)} points, {df['n_images'].iloc[0] if len(df) else '-'} images, {time.time() - t:.0f} s", flush=True)
            return len(df)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            n = sum(ex.map(work, todo))
        log.count("points_written", n)
        if smoke:
            log.discard()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=list(JOBS), required=True)
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
