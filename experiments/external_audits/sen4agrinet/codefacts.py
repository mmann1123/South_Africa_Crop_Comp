"""Pinned source files and the line ranges that carry each code fact the audit
relies on. Each fact has a regular expression that must match somewhere in its
extracted lines, so a moved or changed line fails loudly instead of citing
the wrong text. The files are only read as text, never imported or run."""
from __future__ import annotations

import re
from dataclasses import dataclass

MODELS = ("Orion-AI-Lab/S4A-Models", "ae5a3761395cd259724516fa52a89879850558ab")
LOADERS = ("Orion-AI-Lab/S4A", "7ac66dce57abb6dd05683891bc810c2ddc36a966")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    repo: str
    commit: str
    path: str
    start: int
    end: int
    expect: str  # regex that must match in the extracted text
    note: str


FACTS = [
    Fact("pad_bin_median", *MODELS, "utils/PAD_dataset.py", 293, 299, r"groupby_bins[\s\S]*right=True[\s\S]*include_lowest=False[\s\S]*\.median\(dim='time'\)", "monthly bins (start, end], median over acquisitions"),
    Fact("pad_resample", *MODELS, "utils/PAD_dataset.py", 301, 302, r"resample\(time_bins=self\.group_freq\)\.median", "upsample so empty months are NaN"),
    Fact("pad_interpolate", *MODELS, "utils/PAD_dataset.py", 304, 307, r"interpolate_na\(dim='time_bins', method='linear', fill_value='extrapolate'\)", "linear interpolation, linear extrapolation, no max_gap"),
    Fact("pad_date_range", *MODELS, "utils/PAD_dataset.py", 281, 282, r"pd\.date_range\(start=f'\{year\}-01-01', end=f'\{int\(year\) \+ 1\}-01-01'", "calendar-year month bins"),
    Fact("pad_docstring_claim", *MODELS, "utils/PAD_dataset.py", 31, 38, r"median of all data from previous and\s+following bins", "docstring describes a different fill than the code applies"),
    Fact("median_export_chain", *MODELS, "export_medians_multi.py", 126, 140, r"groupby_bins[\s\S]*resample[\s\S]*interpolate_na[\s\S]*extrapolate", "precomputed-median path used by --saved_medians"),
    Fact("oad_csv_builder_chain", *MODELS, "object-based-csv.py", 69, 83, r"groupby_bins[\s\S]*resample[\s\S]*interpolate_na[\s\S]*extrapolate", "OAD parcel statistics are built from the filled monthly series"),
    Fact("loader_chain", *LOADERS, "patches_dataset.py", 284, 298, r"groupby_bins[\s\S]*resample[\s\S]*interpolate_na[\s\S]*extrapolate", "same chain in the S4A loader repository"),
    Fact("oad_window_slice", *MODELS, "utils/OAD_datamodule.py", 53, 53, r"4 \* self\.num_bands \* self\.metrics_per_band:-\(self\.num_bands \* self\.metrics_per_band \+ 3\)", "month indices 4 to 10, i.e. May to November"),
    Fact("oad_fit_loads_test_csv", *MODELS, "utils/OAD_datamodule.py", 90, 99, r"stage == 'fit'[\s\S]*_test\.csv\.gz[\s\S]*_test\.csv\.gz", "train and eval datasets both read the _test file"),
    Fact("oad_validate_test_predict_load_test_csv", *MODELS, "utils/OAD_datamodule.py", 101, 117, r"validate[\s\S]*_test\.csv\.gz[\s\S]*'test'[\s\S]*_test\.csv\.gz[\s\S]*predict[\s\S]*_test\.csv\.gz", "validate, test and predict read the _test file"),
    Fact("masks_default_off", *MODELS, "pad_experiments.py", 167, 175, r"--return_masks[\s\S]*default=False[\s\S]*--clouds[\s\S]*default=False[\s\S]*--snow[\s\S]*default=False", "no cloud mask unless a flag is passed, and none is passed by the README command"),
]


def extract(text: str, start: int, end: int) -> str:
    lines = text.splitlines()
    if start < 1 or end > len(lines) or start > end:
        raise ValueError(f"line range {start}-{end} outside file of {len(lines)} lines")
    return "\n".join(lines[start - 1 : end])


def check(fact: Fact, text: str) -> bool:
    return re.search(fact.expect, extract(text, fact.start, fact.end)) is not None


def raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
