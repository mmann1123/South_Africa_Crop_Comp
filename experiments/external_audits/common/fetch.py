"""HTTP downloads with a SHA-256 manifest, and HTTP range reads."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import requests

CHUNK = 8 * 1024 * 1024
TIMEOUT = 120


def manifest_entries(manifest: Path) -> list[dict]:
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())


def _write_manifest(manifest: Path, entries: list[dict]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(entries, indent=2))


def download(
    url: str,
    dest: Path,
    manifest: Path,
    key: str,
    expected_bytes: int | None = None,
    force: bool = False,
) -> Path:
    """Stream url to dest, record {key,url,bytes,sha256,fetched_at} in manifest.

    Skips the fetch when dest exists and the manifest entry for key matches its size.
    """
    dest = Path(dest)
    entries = manifest_entries(manifest)
    existing = next((e for e in entries if e["key"] == key), None)
    if not force and dest.exists() and existing and existing["bytes"] == dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    sha = hashlib.sha256()
    n = 0
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                sha.update(chunk)
                n += len(chunk)
    if expected_bytes is not None and n != expected_bytes:
        part.unlink(missing_ok=True)
        raise IOError(f"{url}: got {n} bytes, expected {expected_bytes}")
    part.replace(dest)
    entry = {
        "key": key,
        "url": url,
        "bytes": n,
        "sha256": sha.hexdigest(),
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    entries = [e for e in entries if e["key"] != key] + [entry]
    _write_manifest(manifest, entries)
    return dest


def range_get(url: str, start: int, length: int) -> bytes:
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def head_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
    r.raise_for_status()
    return int(r.headers["Content-Length"])
