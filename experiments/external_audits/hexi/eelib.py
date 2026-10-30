"""Earth Engine helpers for the Hexi audit: per-image validity bands and weekly count images."""
from __future__ import annotations

import ee

from experiments.external_audits.hexi import catalogue as C, hexilib as hl

BANDS = ["n", "q", *C.CS_THRESHOLDS, *[f"q_{k}" for k in C.CS_THRESHOLDS]]


def linked(year: int, geom, start: str, end: str, product: str = C.S2_L1C) -> ee.ImageCollection:
    """Sentinel-2 images of [start, end) over geom with the Cloud Score+ cs_cdf band attached."""
    s2 = ee.ImageCollection(product).filterBounds(geom).filterDate(start, end)
    cs = ee.ImageCollection(C.CSPLUS).filterBounds(geom).filterDate(start, end)
    return s2.linkCollection(cs, ["cs_cdf"])


def validity(img: ee.Image) -> ee.Image:
    """Per image 0/1 bands: n (pixel inside the footprint), q (QA60 clear, the paper's mask), one band per
    Cloud Score+ threshold, and q_<thr> (QA60 clear and Cloud Score+ clear)."""
    inside = img.select("B2").mask().unmask(0).gt(0)
    cloud = img.select("QA60").unmask(0).toInt32().bitwiseAnd(sum(1 << b for b in C.QA60_CLOUD_BITS)).neq(0)
    q = inside.And(cloud.Not())
    cs = img.select("cs_cdf").unmask(0)
    out = [inside.rename("n"), q.rename("q")]
    for name, thr in C.CS_THRESHOLDS.items():
        out.append(inside.And(cs.gte(thr)).rename(name))
    for name, thr in C.CS_THRESHOLDS.items():
        out.append(q.And(cs.gte(thr)).rename(f"q_{name}"))
    return ee.Image.cat(out).toInt8().copyProperties(img, ["system:time_start"])


def weekly_counts(year: int, geom, weeks: list[int], product: str = C.S2_L1C) -> ee.Image:
    """Bands w<kk>_<name>: number of images per week and pixel that are inside / QA60-clear / Cloud Score+ clear."""
    a0, _ = hl.week_bounds(year, weeks[0])
    _, b1 = hl.week_bounds(year, weeks[-1])
    mapped = linked(year, geom, a0.isoformat(), b1.isoformat(), product).map(validity)
    zero = ee.ImageCollection([ee.Image.constant([0] * len(BANDS)).rename(BANDS).toInt8()])
    layers = []
    for k in weeks:
        a, b = hl.week_bounds(year, k)
        w = mapped.filterDate(a.isoformat(), b.isoformat()).merge(zero).sum()
        layers.append(w.rename([f"w{k:02d}_{n}" for n in BANDS]))
    return ee.Image.cat(layers)
