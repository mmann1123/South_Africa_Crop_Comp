"""Shared handling of missing values in raw band x month inputs at inference.

The holdout tile is missing September for ~41% of pixels, while the training tiles have no
missing values in any month the models use. The original inference scripts filled such gaps
in different ways (0 for most raw-sequence models, the holdout's own column medians for pixel
TabNet), so models were compared on differently corrupted inputs. This module puts the choice
in one place.

Methods (``fill_missing``):
  published     each script's original behaviour (pass it as ``published=``); the default, so
                every published result reproduces unchanged
  zero          fill with 0
  self_median   fill with the frame's own column medians, then 0; uses statistics of the data
                being predicted (kept only to reproduce the original pixel-TabNet path)
  train_median  fill with column medians over the training-tile pixels
  train_mean    fill with column means over the training-tile pixels
  interp        per row, linear interpolation between the nearest observed months of the same
                band; months before the first or after the last observation, columns that are
                not band_Month (e.g. SHAPE_AREA), and months never observed in training (May)
                fall back to train_median

The method is chosen by the ``method`` argument, else the HOLDOUT_FILL environment variable,
else config.HOLDOUT_FILL ("published"). Training-tile statistics are computed once from
config.MERGED_DL_PATH and cached in config.DATA_OUTPUT_DIR.
"""
import os

import numpy as np
import pandas as pd

METHODS = ("published", "zero", "self_median", "train_median", "train_mean", "interp")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def resolve_method(method=None):
    """Explicit argument, else $HOLDOUT_FILL, else config.HOLDOUT_FILL."""
    if method is None:
        method = os.environ.get("HOLDOUT_FILL")
    if method is None:
        from config import HOLDOUT_FILL
        method = HOLDOUT_FILL
    if method not in METHODS:
        raise ValueError(f"Unknown fill method {method!r}; choose from {METHODS}")
    return method


def train_stats(kind, cols=None, path=None, cache_dir=None):
    """Column medians or means over the training-tile pixels, as a pandas Series.

    Computed from ``path`` (default config.MERGED_DL_PATH) and cached as
    ``train_{kind}s.csv`` in ``cache_dir`` (default config.DATA_OUTPUT_DIR). Columns that are
    missing in every training pixel (May) get 0: no training statistic exists, and every
    training pipeline either dropped those columns or filled them with 0, so 0 is the value
    the models saw. Requesting a column absent from the training file raises.
    """
    if kind not in ("median", "mean"):
        raise ValueError("kind must be 'median' or 'mean'")
    if path is None or cache_dir is None:
        from config import DATA_OUTPUT_DIR, MERGED_DL_PATH
        path = path or MERGED_DL_PATH
        cache_dir = cache_dir or DATA_OUTPUT_DIR
    stats = _raw_train_stats(kind, path, cache_dir).fillna(0.0)
    if cols is not None:
        missing = [c for c in cols if c not in stats.index]
        if missing:
            raise KeyError(f"No training {kind} for columns {missing}")
        stats = stats.loc[list(cols)]
    return stats


def train_observed(path=None, cache_dir=None):
    """Band-month columns observed in at least one training pixel (excludes May)."""
    raw = _raw_train_stats("median", path, cache_dir)
    return set(raw.index[raw.notna()])


def _raw_train_stats(kind, path=None, cache_dir=None):
    """Training statistics with NaN for columns never observed in training; cached."""
    if path is None or cache_dir is None:
        from config import DATA_OUTPUT_DIR, MERGED_DL_PATH
        path = path or MERGED_DL_PATH
        cache_dir = cache_dir or DATA_OUTPUT_DIR
    cache = os.path.join(cache_dir, f"train_{kind}s.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, float_precision="round_trip").iloc[:, 0]
    import pyarrow.parquet as pq
    names = [c for c in pq.ParquetFile(path).schema.names if _band_month(c) is not None]
    data = pq.read_table(path, columns=names).to_pandas()
    raw = (data.median() if kind == "median" else data.mean()).rename(f"train_{kind}")
    raw.to_csv(cache, float_format="%.17g")  # exact float64 round trip; NaN = never observed
    return raw


def fill_missing(df, cols, published, method=None, stats=None, observed=None):
    """Return ``df[cols]`` with missing values filled; ``df`` itself is not modified.

    published : the calling script's original method ("zero" or "self_median"), used when the
                resolved method is "published".
    stats     : optional Series of training statistics (for train_median, train_mean, interp);
                loaded with ``train_stats`` when not given.
    observed  : (interp only) columns observed in training; only these are interpolated, the
                rest get ``stats`` (0 for never-observed months such as May, the value the
                models saw in training). Default: ``train_observed()`` when ``stats`` is not
                given, otherwise every column.
    """
    if published not in ("zero", "self_median"):
        raise ValueError("published must be 'zero' or 'self_median'")
    method = resolve_method(method)
    if method == "published":
        method = published
    cols = list(cols)
    x = df[cols]
    if method == "zero":
        return x.fillna(0)
    if method == "self_median":
        return x.fillna(x.median()).fillna(0)
    need = [c for c in cols if x[c].isna().any()]
    if method in ("train_median", "train_mean"):
        s = stats if stats is not None else train_stats(method.split("_")[1], need)
        out = x.fillna(s.reindex(cols))
    else:  # interp
        if observed is None:
            observed = set(cols) if stats is not None else train_observed()
        s = stats if stats is not None else train_stats("median", need)
        out = _interp(x, s, observed)
    left = [c for c in need if out[c].isna().any()]
    if left:
        raise KeyError(f"No training statistic to fill columns {left}")
    return out


def _band_month(col):
    """'B11_September' -> ('B11', 8); None for non band-month columns."""
    if "_" not in col:
        return None
    band, month = col.rsplit("_", 1)
    return (band, MONTHS.index(month)) if month in MONTHS else None


def _interp(x, stats, observed):
    """Linear interpolation along the months of each band: each missing month takes the value
    on the line between the nearest observed month before it and the nearest after it (same
    pixel, same band). Weights use calendar month positions, so a gap between April and July
    spans three steps. Edge gaps and columns not in ``observed`` fall back to ``stats``.
    Vectorized over rows."""
    out = x.copy()
    bands = {}
    for c in x.columns:
        bm = _band_month(c)
        if bm is not None and c in observed:  # other columns fall through to the stats fallback
            bands.setdefault(bm[0], []).append((bm[1], c))
    for band, pairs in bands.items():
        pairs.sort()
        pos = np.array([p for p, _ in pairs], dtype=float)
        bc = [c for _, c in pairs]
        v = x[bc].to_numpy(dtype=float)
        miss = np.isnan(v)
        if not miss.any():
            continue
        n, t = v.shape
        idx = np.arange(t)
        prev = np.maximum.accumulate(np.where(~miss, idx, -1), axis=1)            # last observed <= j
        nxt = np.minimum.accumulate(np.where(~miss, idx, t)[:, ::-1], axis=1)[:, ::-1]  # first observed >= j
        inside = miss & (prev >= 0) & (nxt < t)
        r, j = np.nonzero(inside)
        p, q = prev[r, j], nxt[r, j]
        w = (pos[j] - pos[p]) / (pos[q] - pos[p])
        v[r, j] = v[r, p] + w * (v[r, q] - v[r, p])
        out[bc] = v
    return out.fillna(stats.reindex(x.columns))
