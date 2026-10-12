"""TimesNet (Wu et al., ICLR 2023), vendored unmodified from THUML Time-Series-Library.

Source: https://github.com/thuml/Time-Series-Library
Commit: 4e938a1767106324dd753b2a44832bf870a0252e (2026-04-18), MIT licence (see LICENSE).
Files:  models/TimesNet.py -> TimesNet.py, layers/Embed.py, layers/Conv_Blocks.py

The files are byte-identical to that commit (SHA-256 listed in ../README.md). Because
TimesNet.py uses absolute imports (`from layers.Embed import ...`), this directory is
put on sys.path by `load_timesnet()` instead of editing the vendored code.
"""
import os
import sys

TSLIB_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"


def load_timesnet():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from TimesNet import Model  # noqa: E402
    return Model
