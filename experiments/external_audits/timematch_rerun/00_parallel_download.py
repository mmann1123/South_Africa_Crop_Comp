#!/usr/bin/env python
"""Resumable parallel range download of the Zenodo timematch_data.zip (single-stream
Zenodo speed is about 1 MB/s). Writes into one preallocated file; a state file records
finished chunks. Verifies the final byte count. Usage: python 00_parallel_download.py"""
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://zenodo.org/records/5636422/files/timematch_data.zip?download=1"
ROOT = Path("/mnt/bigdrive/external_audits/timematch_rerun")
DEST = ROOT / "timematch_data.zip"
STATE = ROOT / "timematch_data.zip.chunks.json"
SIZE = 73_904_625_097
CHUNK = 64 * 1024 * 1024
WORKERS = 32
lock = threading.Lock()


def fetch(i, done):
    a = i * CHUNK
    b = min(SIZE, a + CHUNK) - 1
    for attempt in range(20):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != b - a + 1:
                raise IOError(f"short read {len(data)}")
            fd = os.open(DEST, os.O_WRONLY)
            os.pwrite(fd, data, a)
            os.close(fd)
            with lock:
                done.add(i)
                STATE.write_text(json.dumps(sorted(done)))
            return
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"chunk {i} failed")


def main():
    n = (SIZE + CHUNK - 1) // CHUNK
    if STATE.exists():
        done = set(json.loads(STATE.read_text()))
    else:  # a prior sequential wget wrote the first cur bytes
        cur = DEST.stat().st_size if DEST.exists() else 0
        done = set(range(cur // CHUNK))
    if not DEST.exists():
        DEST.touch()
    if DEST.stat().st_size < SIZE:
        with open(DEST, "r+b") as f:
            f.truncate(SIZE)
    todo = [i for i in range(n) if i not in done]
    print(f"{len(done)} chunks done, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda i: fetch(i, done), todo))
    assert DEST.stat().st_size == SIZE, DEST.stat().st_size
    print("complete", DEST.stat().st_size)


if __name__ == "__main__":
    main()
