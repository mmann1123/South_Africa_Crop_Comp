"""Digitise Figure 19 of Stoian et al. 2019 (kappa per tile, four marker series).

Pure functions on an RGB array. Markers are located by colour: each series has a
known colour, anti-aliased marker edges are blends of that colour with white, and a
marker partly covered by a later-drawn marker is placed by sliding a template cut
from the fully visible markers of the same series along the y axis.
"""
from __future__ import annotations

import numpy as np

# Series in the legend order; colours are the Excel defaults used in the figure.
SERIES = {
    "fg_unet_11": (91, 155, 213),    # blue circle: FG-UNET, 11 tiles
    "fg_unet_10p1": (237, 125, 49),  # orange cross: FG-UNET, 10 tiles + 1 test
    "rf_11": (165, 165, 165),        # grey diamond: RF, 11 tiles
    "rf_10p1": (255, 192, 0),        # yellow square: RF, 10 tiles + 1 test
}
ALPHA_MIN = 0.45  # minimum share of the series colour in a pixel to count as marker
RESIDUAL_MAX = 28.0  # max distance from the colour-to-white line


def alpha_mask(rgb: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """Pixels that are a blend of `colour` and white with at least ALPHA_MIN of the colour."""
    d = 255.0 - np.asarray(colour, float)
    p = 255.0 - rgb.astype(float)
    a = (p @ d) / (d @ d)
    resid = np.linalg.norm(p - a[..., None] * d, axis=-1)
    return (a >= ALPHA_MIN) & (a <= 1.25) & (resid <= RESIDUAL_MAX)


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of consecutive True runs in a 1-D boolean array."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def calibrate_y(label_rows: list[tuple[int, int]], values: list[float]) -> tuple[float, float]:
    """Least-squares (slope, intercept) mapping pixel row to value from tick-label row spans.

    A label's centre is the midpoint of its inclusive row span (all labels are digits of
    equal height, so the midpoint sits on the tick value).
    """
    centres = np.array([(a + b) / 2.0 for a, b in label_rows])
    slope, intercept = np.polyfit(centres, np.asarray(values, float), 1)
    return float(slope), float(intercept)


def tick_label_rows(rgb: np.ndarray, x0: int, x1: int, dark: int = 500) -> list[tuple[int, int]]:
    """Row spans of the y tick labels, found as dark-pixel runs in the label column x0:x1."""
    d = rgb[:, x0:x1].astype(int).sum(-1) < dark
    return runs(d.any(axis=1))


def column_centres(rgb: np.ndarray, y0: int, y1: int, gap: int = 12, dark: int = 500) -> list[float]:
    """x centre of each x tick label, from dark-pixel runs in rows y0:y1 merged across gaps < `gap`."""
    d = (rgb[y0:y1].astype(int).sum(-1) < dark).any(axis=0)
    out: list[list[int]] = []
    for a, b in runs(d):
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a + b) / 2.0 for a, b in out]


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(m)
    h, w = m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ys2, xs2 = ys + dy, xs + dx
    ok = (ys2 >= 0) & (ys2 < h) & (xs2 >= 0) & (xs2 < w)
    out[ys2[ok], xs2[ok]] = m[ys[ok], xs[ok]]
    return out


def make_template(windows: list[np.ndarray], full_frac: float = 0.97) -> np.ndarray:
    """Template from the fully visible markers of a series.

    Each window is a boolean crop around one marker; a window is 'full' if its pixel
    count is at least `full_frac` of the largest. The template is the union of the full
    windows aligned on their bounding-box centres, thresholded at half.
    """
    counts = np.array([w.sum() for w in windows])
    full = [w for w, c in zip(windows, counts) if c >= full_frac * counts.max()]
    acc = None
    size = max(max(w.shape) for w in full)
    pad = size + 8
    acc = np.zeros((pad, pad))
    for w in full:
        ys, xs = np.where(w)
        cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
        ty, tx = pad // 2 - int(round(cy)), pad // 2 - int(round(cx))
        acc += _shift(np.pad(w, ((0, pad - w.shape[0]), (0, pad - w.shape[1]))).astype(float), ty, tx)
    return acc / len(full) >= 0.5


def locate(mask: np.ndarray, template: np.ndarray, x_centre: float, y_range: tuple[int, int], occluders: np.ndarray | None = None) -> dict:
    """Find the marker centre (row, column) for one series in one tile column.

    Scores each integer placement of the template (centre within 2 px of the column
    centre in x, anywhere in y_range) by visible marker pixels inside the template
    minus twice those outside it, minus one for each template pixel that is neither this
    series' marker nor covered by another series' marker (`occluders`), using the column
    window only. The last term stops a half-hidden marker sliding under its occluder. Returns the best
    centre, the share of template pixels that are visible at that placement, and
    row_span, the vertical spread in pixels of placements scoring within 2 of the best
    (the position uncertainty; 0 for a fully visible marker).
    """
    y0, y1 = y_range
    xc = int(round(x_centre))
    x0, x1 = xc - 10, xc + 11
    sub = mask[y0:y1, x0:x1]
    occ = np.zeros_like(mask) if occluders is None else occluders
    occ_sub = occ[y0:y1, x0:x1]
    n_t = int(template.sum())
    scores: dict[tuple[int, int], int] = {}
    for cy in range(y0 + 2, y1 - 2):
        for cx in range(xc - 2, xc + 3):
            tsub = _placed(template, cy, cx, mask.shape)[y0:y1, x0:x1]
            scores[(cy, cx)] = int((sub & tsub).sum()) - 2 * int((sub & ~tsub).sum()) - int((tsub & ~sub & ~occ_sub).sum())
    best_key = max(scores, key=lambda k: (scores[k], -abs(k[1] - x_centre)))
    near = [k for k, v in scores.items() if v >= scores[best_key] - 2 and abs(k[1] - best_key[1]) <= 1]
    cy, cx = best_key
    th, tw = template.shape
    ys, xs = np.where(template)
    visible = int((sub & _placed(template, cy, cx, mask.shape)[y0:y1, x0:x1]).sum())
    return {
        "row": float(cy - th // 2 + (ys.min() + ys.max()) / 2.0),
        "col": float(cx - tw // 2 + (xs.min() + xs.max()) / 2.0),
        "visible_share": visible / n_t if n_t else 0.0,
        "row_span": float(max(k[0] for k in near) - min(k[0] for k in near)),
    }


def _placed(template: np.ndarray, cy: int, cx: int, shape: tuple[int, int]) -> np.ndarray:
    t = np.zeros(shape, bool)
    th, tw = template.shape
    oy, ox = cy - th // 2, cx - tw // 2
    ys0, xs0 = max(oy, 0), max(ox, 0)
    ys1, xs1 = min(oy + th, shape[0]), min(ox + tw, shape[1])
    t[ys0:ys1, xs0:xs1] = template[ys0 - oy : ys1 - oy, xs0 - ox : xs1 - ox]
    return t


def row_to_value(row: float, calibration: tuple[float, float]) -> float:
    return calibration[0] * row + calibration[1]
