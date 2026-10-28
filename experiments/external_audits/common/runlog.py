"""Run log written by every audit script: inputs, counts, timings, versions."""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import subprocess
import sys
from typing import Any

from . import paths

_PACKAGES = ("ee", "geopandas", "rasterio", "numpy", "pandas", "matplotlib")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def package_versions() -> dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = "not installed"
    return out


class RunLog:
    """Context manager; writes results_dir(paper)/run_log_<script>.json on exit."""

    def __init__(self, script: str, args: dict[str, Any], paper: str, seed: int = 42):
        self.script = script
        self.paper = paper
        self.data: dict[str, Any] = {
            "script": script,
            "args": {k: (str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v) for k, v in args.items()},
            "seed": seed,
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "packages": package_versions(),
            "counts": {},
        }

    def add(self, key: str, value: Any) -> None:
        self.data[key] = value

    def discard(self) -> None:
        """Do not write this run's log (nothing was done); keeps the earlier record on disk."""
        self._discard = True

    def count(self, key: str, n: int) -> None:
        self.data["counts"][key] = n

    def __enter__(self) -> "RunLog":
        self.data["started"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if getattr(self, "_discard", False):
            return False
        self.data["ended"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        if exc is not None:
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        out = paths.results_dir(self.paper) / f"run_log_{self.script}.json"
        out.write_text(json.dumps(self.data, indent=2, default=str))
        return False
