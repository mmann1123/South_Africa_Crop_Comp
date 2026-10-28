"""Per-field clear-observation rules on DENETHOR Sentinel-2 EOPatch layers."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

GRID_M = 24000.0
EPSG = {"33N": "EPSG:32633", "34S": "EPSG:32734"}
SCL_CLEAR = (4, 5, 6, 7)  # vegetation, not vegetated, water, unclassified


def grid_bbox(tile: str, zone: str) -> tuple[float, float, float, float, str]:
    """Bounding box implied by the AI4EO UTM-24000 tile name, e.g. 18E-242N."""
    e, n = tile.split("-")
    minx = float(int(e[:-1]) * GRID_M)
    miny = float(int(n[:-1]) * GRID_M)
    return minx, miny, minx + GRID_M, miny + GRID_M, EPSG[zone]


class _Stub:
    def __init__(self, *a, **k):
        self.args = a

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("sentinelhub") or module.startswith("eolearn"):
            return _Stub
        return super().find_class(module, name)


def _coords(obj) -> tuple[float, float, float, float] | None:
    for attrs in (("min_x", "min_y", "max_x", "max_y"), ("_min_x", "_min_y", "_max_x", "_max_y")):
        if all(hasattr(obj, a) for a in attrs):
            return tuple(float(getattr(obj, a)) for a in attrs)
    for key in ("lower_left", "upper_right"):
        if hasattr(obj, key):
            ll, ur = getattr(obj, "lower_left"), getattr(obj, "upper_right")
            return float(ll[0]), float(ll[1]), float(ur[0]), float(ur[1])
    if hasattr(obj, "args") and len(obj.args) >= 1:
        a = obj.args[0]
        if isinstance(a, (list, tuple)) and len(a) == 4:
            return tuple(float(v) for v in a)
        if isinstance(a, (list, tuple)) and len(a) == 2:
            return float(a[0][0]), float(a[0][1]), float(a[1][0]), float(a[1][1])
    return None


def load_bbox(path: Path, tile: str, zone: str, tol_m: float = 100.0) -> tuple[float, float, float, float, str]:
    """Read bbox.pkl (sentinelhub BBox pickled) without sentinelhub installed and
    cross-check against the 24 km grid implied by the tile name."""
    with open(path, "rb") as fh:
        obj = _Unpickler(fh).load()
    coords = _coords(obj)
    gminx, gminy, gmaxx, gmaxy, gcrs = grid_bbox(tile, zone)
    if coords is None:
        raise ValueError(f"{path}: could not read coordinates from pickled bbox ({type(obj).__name__}: {vars(obj) if hasattr(obj, '__dict__') else obj})")
    if max(abs(a - b) for a, b in zip(coords, (gminx, gminy, gmaxx, gmaxy))) > tol_m:
        raise ValueError(f"{path}: bbox {coords} disagrees with grid {(gminx, gminy, gmaxx, gmaxy)} for tile {tile}")
    return coords[0], coords[1], coords[2], coords[3], gcrs


def detect_clp_scale(clp: np.ndarray) -> int:
    """255 if any sampled value exceeds 100 (Sentinel Hub uint8 CLP), else 100."""
    return 255 if int(np.asarray(clp).max()) > 100 else 100


def clear_rules(clp_mean: np.ndarray, scl_clear_frac: np.ndarray, isdata_frac: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    """Boolean (T, n_fields) arrays for the three clear-observation rules."""
    return {
        "literal40": clp_mean < 40,
        "s2cp40": clp_mean < 0.40 * scale,
        "scl": (scl_clear_frac >= 0.5) & (isdata_frac >= 0.99),
    }
