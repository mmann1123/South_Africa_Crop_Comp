"""Torch-side helpers: load the authors' frozen models and predict with arbitrary
per-parcel date subsets and calendar shifts. Imports the authors' code from the
pinned clone; run with the clone as working directory (class_mapping/ is read
relative to it)."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/bigdrive/external_audits/timematch_rerun")
REPO = ROOT / "repo"
MODELS = ROOT / "models" / "outputs"
DATA_ROOT = ROOT / "timematch_data" / "timematch_data"
PIXEL_MAX = 65535.0
SRC_NAMES = {"DK1": "denmark/32VNH/2017", "FR1": "france/30TXT/2017"}
DIRECTIONS = {"DK1_to_FR1": ("DK1", "FR1", "pseltae_32VNH"), "FR1_to_DK1": ("FR1", "DK1", "pseltae_30TXT")}


def setup_repo() -> None:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def source_classes(source_lab: str) -> list[str]:
    """Class list exactly as train.main builds it: classes of the source country that
    occur at least 200 times among the source tile's parcels."""
    setup_repo()
    from dataset import PixelSetData
    from utils import label_utils

    name = SRC_NAMES[source_lab]
    classes = label_utils.get_classes(name.split("/")[0], combine_spring_and_winter=False)
    data = PixelSetData(str(DATA_ROOT), name, classes)
    labels, counts = np.unique(data.get_labels(), return_counts=True)
    return [classes[i] for i in labels[counts >= 200]]


def load_model(model_dir: Path, fold: int, n_classes: int, device="cuda"):
    import torch

    setup_repo()
    from models.stclassifier import PseLTae

    m = PseLTae(input_dim=10, num_classes=n_classes, with_extra=False)
    sd = torch.load(model_dir / f"fold_{fold}" / "model.pt", map_location="cpu", weights_only=False)["state_dict"]  # compat: checkpoint holds a numpy scalar (best_f1)
    m.load_state_dict(sd)
    return m.to(device).eval()


def read_parcels(tile_name: str, idxs, workers: int = 16) -> list[np.ndarray]:
    """Raw (T, C, S) arrays of the parcels, in order."""
    import zarr

    base = DATA_ROOT / tile_name / "data"

    def rd(i):
        return np.asarray(zarr.load(str(base / f"{int(i)}.zarr")))

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(rd, idxs))


def predict(model, parcels, labels_dummy, positions, keep, shift, group_batches, device="cuda"):
    """Argmax predictions. parcels: list of (T,C,S); positions: (T,) day offsets shared by all parcels;
    keep: (P,T) bool; shift: int days added to every kept position (the authors' `positions + shift`);
    group_batches: list of index lists with identical (kept T, S)."""
    import torch

    preds = np.zeros(len(parcels), dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    with torch.no_grad():
        for b in group_batches:
            x = np.stack([parcels[i][keep[i]] for i in b])  # (B, Tk, C, S)
            p = np.stack([pos[keep[i]] for i in b]) + int(shift)
            xt = torch.from_numpy(np.clip(x, 0, PIXEL_MAX).astype(np.float32) / PIXEL_MAX).to(device, non_blocking=True)
            pt = torch.from_numpy(p).to(device)
            mask = torch.ones(xt.shape[0], xt.shape[1], xt.shape[3], device=device)
            logits = model.forward(xt, mask, pt, None)
            preds[b] = logits.argmax(dim=1).cpu().numpy()
    return preds
