"""Numpy model of the You et al. 2021 interpolation step for one pixel.

The deposited Earth Engine script interpolates list indices 9..30 of 37
ten-day composites. For target i it mosaics the composites with length > 0 in
LIC[i-9 : i] (before) and LIC[i+1 : i+10] (after), computes Y from them, then
`target.unmask(Y).unmask(0)`. Per pixel, Y is masked where the before or the
after mosaic is masked, so the pixel ends up zero iff the target composite is
invalid for that pixel AND (no valid composite among the 9 before OR no valid
composite among the 9 after).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def zero_flags(valid: np.ndarray, targets: Iterable[int] = range(9, 31), search: int = 9) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(valid, dtype=bool)
    targets = list(targets)
    is_zero = np.zeros(len(targets), dtype=bool)
    is_valid = np.zeros(len(targets), dtype=bool)
    for k, i in enumerate(targets):
        is_valid[k] = v[i]
        if v[i]:
            continue
        before = v[max(0, i - search) : i].any()
        after = v[i + 1 : i + 1 + search].any()
        is_zero[k] = not (before and after)
    return is_zero, is_valid
