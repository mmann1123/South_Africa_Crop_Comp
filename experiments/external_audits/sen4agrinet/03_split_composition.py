#!/usr/bin/env python
"""Read the published COCO split files (image lists only) and write which patches
belong to each scenario and split, plus the tile-year composition per split."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.common.tables import write_csv  # noqa: E402
from experiments.external_audits.sen4agrinet import splits  # noqa: E402

PAPER = "sen4agrinet"


def main(force: bool = False) -> None:
    res = paths.results_dir(PAPER)
    out_patches = res / "split_patches.csv.gz"
    with RunLog("03_split_composition", {"force": force}, paper=PAPER) as log:
        if out_patches.exists() and (res / "split_composition.csv").exists() and not force:
            print("split tables exist; use --force to redo")
            log.discard()
            return
        root = paths.paper_root(PAPER) / "dropbox" / "coco_files"
        frames = []
        for exp in (1, 2, 3):
            for split in ("train", "val", "test"):
                f = root / f"exp{exp}_patches5000_strat_coco_{split}.json"
                images = json.loads(f.read_text())["images"]
                df = splits.parse_images([im["file_name"] for im in images])
                df.insert(0, "split", split)
                df.insert(0, "scenario", exp)
                frames.append(df)
                log.count(f"exp{exp}_{split}_patches", len(df))
        allp = pd.concat(frames, ignore_index=True)
        comp = splits.composition(allp)
        allp.to_csv(out_patches, index=False)
        out_patches.with_suffix("").with_suffix(".meta.json").write_text(json.dumps({"script": "03_split_composition", "rows": len(allp), "columns": list(allp.columns)}))
        write_csv(comp, res / "split_composition.csv", "03_split_composition", [str(root)])
        dup = allp.duplicated(["scenario", "stem"], keep=False).sum()
        log.count("patches_in_more_than_one_split_of_a_scenario", int(dup))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
