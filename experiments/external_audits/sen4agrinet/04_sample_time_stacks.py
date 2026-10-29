#!/usr/bin/env python
"""Read the acquisition times of a stratified sample of patch NetCDFs from the
Hugging Face dataset orion-ai-lab/S4A (pinned revision) by HTTP range requests:
only the HDF5 metadata and the B02/time vector of each file are read, not the
imagery. The sample is 25 random patches per scenario x split x tile-year for
scenarios 2 and 3 (train and test). Resumable: patches already in the output
are skipped."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import h5py
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import splits  # noqa: E402

PAPER = "sen4agrinet"
HF_REVISION = "4eb44157012b9ec251767721d6bae55cf0c564e2"  # orion-ai-lab/S4A main at the time of the audit
HF_URL = "https://huggingface.co/datasets/orion-ai-lab/S4A/resolve/" + HF_REVISION + "/{path}"
N_PER_GROUP = 25
RETRIES = 5


def read_times(stem: str) -> dict:
    url = HF_URL.format(path=splits.hf_path(stem))
    err = None
    for attempt in range(RETRIES):
        try:
            fs = fsspec.filesystem("http")
            with fs.open(url, block_size=1 << 20, cache_type="readahead") as f:
                h = h5py.File(f, "r")
                tm = h["B02/time"]
                units = tm.attrs["units"]
                units = units.decode() if isinstance(units, bytes) else str(units)
                t = splits.decode_times(units, tm[:])
            return {"stem": stem, "n_obs": len(t), "times": ";".join(x.strftime("%Y-%m-%dT%H:%M:%S") for x in t)}
        except Exception as e:  # noqa: BLE001  (429, resets, truncated reads)
            err = e
            time.sleep(4 * 2**attempt)
    raise RuntimeError(f"{stem}: {err}")


def main(workers: int = 6, force: bool = False) -> None:
    res = paths.results_dir(PAPER)
    out = res / "patch_time_stacks.csv.gz"
    with RunLog("04_sample_time_stacks", {"workers": workers, "force": force, "n_per_group": N_PER_GROUP}, paper=PAPER) as log:
        allp = pd.read_csv(res / "split_patches.csv.gz")
        pool = allp[allp["scenario"].isin([2, 3]) & allp["split"].isin(["train", "test"])]
        sample = splits.sample_per_tile_year(pool, N_PER_GROUP, seed=42)
        sample.to_csv(res / "patch_sample.csv", index=False)
        have = pd.read_csv(out) if out.exists() and not force else pd.DataFrame(columns=["stem", "n_obs", "times"])
        todo = sorted(set(sample["stem"]) - set(have["stem"]))
        log.add("hf_revision", HF_REVISION)
        log.count("sample_rows", len(sample))
        log.count("unique_files_needed", sample["stem"].nunique())
        log.count("to_fetch", len(todo))
        if not todo:
            print("all sampled patches already read")
            if out.exists():
                log.discard()
                return
        rows = have.to_dict("records")
        t0 = time.time()
        failed = []
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(read_times, s): s for s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rows.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    failed.append({"stem": futs[fut], "error": str(e)})
                if i % 50 == 0:
                    pd.DataFrame(rows).to_csv(out, index=False)
                    print(f"{i}/{len(todo)} read, {(time.time() - t0) / 60:.1f} min", flush=True)
        df = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
        write_csv(df, out, "04_sample_time_stacks", [HF_URL.format(path="data/<year>/<tile>/<stem>.nc") + " (B02/time only)"])
        # write_csv names the sidecar from the .gz path; fine
        log.count("read_ok", len(df))
        log.add("failed", failed)
        if failed:
            raise SystemExit(f"{len(failed)} patches failed; rerun to retry them")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
