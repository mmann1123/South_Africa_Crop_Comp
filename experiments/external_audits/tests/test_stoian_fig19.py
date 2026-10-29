import numpy as np

from experiments.external_audits.stoian import fig19

BLUE = fig19.SERIES["fg_unet_11"]


def _canvas(h=120, w=60):
    return np.full((h, w, 3), 255, np.uint8)


def _square(img, cy, cx, colour, half=7):
    img[cy - half : cy + half + 1, cx - half : cx + half + 1] = colour


def test_alpha_mask_picks_colour_and_blends():
    img = _canvas()
    _square(img, 40, 30, BLUE, half=3)
    img[60, 10] = tuple(int(255 - 0.6 * (255 - c)) for c in BLUE)  # 60 percent blend with white
    img[70, 10] = tuple(int(255 - 0.2 * (255 - c)) for c in BLUE)  # faint, below threshold
    img[80, 10] = (0, 0, 0)
    m = fig19.alpha_mask(img, BLUE)
    assert m[40, 30] and m[60, 10] and not m[70, 10] and not m[80, 10]


def test_runs():
    f = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert fig19.runs(f) == [(1, 2), (5, 5), (7, 9)]
    assert fig19.runs(np.zeros(4, bool)) == []


def test_calibrate_y_recovers_scale():
    rows = [(40, 48), (70, 78), (100, 108)]  # centres 44, 74, 104: 30 px per 0.1
    slope, intercept = fig19.calibrate_y(rows, [1.0, 0.9, 0.8])
    assert abs(fig19.row_to_value(89, (slope, intercept)) - 0.85) < 1e-9


def test_locate_full_and_partly_hidden_marker():
    full = _canvas()
    _square(full, 50, 30, BLUE)
    mask = fig19.alpha_mask(full, BLUE)
    tmpl = fig19.make_template([mask[40:61, 20:41]])
    loc = fig19.locate(mask, tmpl, 30, (10, 110))
    assert abs(loc["row"] - 50) <= 0.5 and abs(loc["col"] - 30) <= 0.5 and loc["visible_share"] > 0.95
    hidden = full.copy()
    _square(hidden, 56, 30, (255, 192, 0), half=7)  # another marker covers the lower half
    m2 = fig19.alpha_mask(hidden, BLUE)
    occ = fig19.alpha_mask(hidden, (255, 192, 0))
    loc2 = fig19.locate(m2, tmpl, 30, (10, 110), occ)
    assert abs(loc2["row"] - 50) <= 1.0
    assert loc2["visible_share"] < 0.9
