"""Constants for the Verhulst et al. 2024 audit (Remote Sensing 16:2653, doi 10.3390/rs16142653)."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

from experiments.external_audits.common import fetch, paths

PAPER = "verhulst"
DOI = "10.3390/rs16142653"
PDF_URL = "https://www.mdpi.com/2072-4292/16/14/2653/pdf"
FIG7_PAGE = 12  # 1-based PDF page of the MDPI PDF holding Figure 7
YEARS = [2018, 2019, 2020, 2021, 2022]
N_DEKADS = 36  # three per month, January to December of every year (paper Section 2.3.3)
JAN_JUN = 18  # dekads 0..17
SEED = 42

# Facts pinned from the paper text (Section 2.2).
PAPER_PRODUCT = "TERRASCOPE_S2_TOC_V2 (Sen2Cor Level-2A, full Belgian archive)"
PAPER_MASK = "SCL 4 and 5 kept; mask dilated with an 11 x 11 Gaussian kernel, sigma 1.5 (threshold not stated)"
EE_PRODUCT = "COPERNICUS/S2_SR_HARMONIZED"  # Sen2Cor Level-2A as well
GAUL = "FAO/GAUL/2015/level1"
GAUL_NAME = "Vlaams Gewest"
WORLDCOVER = "ESA/WorldCover/v200"
TREE_CLASS = 10
CRS = "EPSG:32631"  # UTM 31N covers all of Flanders (2.5 E to 5.9 E)
SCALE_M = 10
N_POINTS = 3000
KERNEL_RADIUS = 5  # 11 x 11
KERNEL_SIGMA = 1.5
# Dilation threshold on the Gaussian-weighted share of unusable pixels in the 11 x 11 window.
# The paper does not state one. 0.01 reaches about 3 px from an unusable pixel, 0.05 about 1 px.
VARIANTS = {"none": None, "d3": 0.01, "d1": 0.05}
PRIMARY = "d3"


def paper_pdf() -> Path:
    """The MDPI PDF, fetched with a manifest entry; a copy already on disk is adopted, not re-downloaded."""
    dest = paths.paper_root(PAPER) / "dl" / "paper.pdf"
    manifest = paths.manifest_path(PAPER)
    if dest.exists() and not any(e["key"] == "paper.pdf" for e in fetch.manifest_entries(manifest)):
        entry = {"key": "paper.pdf", "url": PDF_URL, "bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        fetch._write_manifest(manifest, fetch.manifest_entries(manifest) + [entry])
    return fetch.download(PDF_URL, dest, manifest, "paper.pdf")
