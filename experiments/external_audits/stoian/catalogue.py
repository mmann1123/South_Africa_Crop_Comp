"""Constants and fetch helpers for the Stoian et al. 2019 audit."""
from __future__ import annotations

from pathlib import Path

from experiments.external_audits.common import fetch, paths

PAPER = "stoian"
TILES = ["T30TWT", "T30TXQ", "T30TYN", "T30UXV", "T31TDJ", "T31TDN", "T31TEL", "T31TGK", "T31UDQ", "T31UDS", "T32ULU"]
MOUNTAIN = ["T30TYN", "T31TGK"]  # the two tiles the authors call mountain tiles
YEAR = 2016
SEASON = ("03-01", "10-31")  # March to October window used for the growing-season measures

HAL_PDF_URL = "https://hal.science/hal-02627773/document"
HAL_PDF_KEY = "hal-02627773/document.pdf"
FIG19_PAGE = 23  # 1-based PDF page of the HAL deposit (cover page included): paper p. 22 of 26
TABLE1_PAGE = 7  # 1-based PDF page of the HAL deposit: paper p. 6 of 26

# Region labels from the paper's Figure 1, carried over from the desk check.
REGION = {
    "T30TWT": "Oceanic coast, Loire estuary",
    "T30TXQ": "Oceanic, Landes forest and Gironde coast",
    "T30TYN": "Mountain, central Pyrenees (about half in Spain)",
    "T30UXV": "Oceanic, Normandy (Caen)",
    "T31TDJ": "South-western basin, Tarn/Albi",
    "T31TDN": "Degraded oceanic, Centre (Bourges)",
    "T31TEL": "Mountain/semi-continental, Massif Central",
    "T31TGK": "Mountain/altered Mediterranean, southern Alps",
    "T31UDQ": "Degraded oceanic, Paris",
    "T31UDS": "Oceanic coast, Calais/Dunkirk",
    "T32ULU": "Semi-continental, Alsace/Vosges",
}

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
SCL_CLEAR = (2, 4, 5, 6, 7, 11)  # dark features, vegetation, not vegetated, water, unclassified, snow/ice
GRID = 344  # read grid per tile, about 320 m


def paper_pdf() -> Path:
    """Download the HAL deposit of the paper (SHA-256 manifest) and return its path."""
    return fetch.download(HAL_PDF_URL, paths.paper_root(PAPER) / "hal-02627773.pdf", paths.manifest_path(PAPER), HAL_PDF_KEY)
