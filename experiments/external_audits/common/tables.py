"""CSV writing with provenance sidecars, and small summary helpers."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from .runlog import git_commit


def write_csv(df: pd.DataFrame, path: Path, script: str, inputs: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = {
        "script": script,
        "git_commit": git_commit(),
        "run_date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs": list(inputs),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return path


def longest_gap_days(dates: pd.Series, clear: pd.Series, start: str, end: str) -> int:
    """Longest interval in days with no clear acquisition inside [start, end].

    The window edges count as boundaries, so the gap from `start` to the first
    clear date and from the last clear date to `end` are included. With no clear
    date the result is (end - start) in days.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    c = pd.Series(clear).reset_index(drop=True).astype(bool)
    pts = sorted(set(d[c & (d >= s) & (d <= e)]))
    bounds = [s] + pts + [e]
    return int(max((b - a).days for a, b in zip(bounds[:-1], bounds[1:])))
