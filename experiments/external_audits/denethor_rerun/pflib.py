"""Batches and model for the DENETHOR PseLTae re-run (lukaskondmann/DENETHOR
55573a8, train_psetae.py / test_psetae.py), on the fixed 128-pixel sets that
01_extract_pf.py stores.

Authors' per-sample transform, reproduced: 20 pixels drawn with replacement
from the field's pixels (here from its 128 stored ones), reflectance * 1e-4,
then (x - (0.1014 + N(0, 0.01))) / (0.1171 + N(0, 0.01)) with one draw of each
jitter per sample, in training and at test. Input layout B x T(365) x C x N(20).

Conditions:
  A0  4 PF-SR bands (the authors' model).
  A1  plus PF-QA layer 1 (percent synthetic, 1 to 100) at the same pixels,
      divided by 100, as a fifth channel (not jittered).
  A2  the fifth channel holds the field's own QA values with the day axis
      permuted by a fixed per-field permutation (seed 0), which keeps the
      channel's values and the field's synthetic share but breaks the link
      between a day and whether it was observed.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/bigdrive/external_audits/denethor_rerun/repo")
CROP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
N_PIXELS = 20
QA_NODATA = -1000


def pseltae_class():
    p = str(REPO / "models")
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module("pseltae").PseLTae


def build_model(input_dim: int):
    """train_psetae.py's PseLTae configuration, with input_dim 4 or 5."""
    PseLTae = pseltae_class()
    return PseLTae(input_dim=input_dim, mlp1=[input_dim, 32, 64], pooling="mean_std", mlp2=[128, 128], with_extra=False,
                   extra_size=0, n_head=16, d_k=8, d_model=256, mlp3=[256, 128], dropout=0.2, T=365, len_max_seq=365,
                   positions=None, mlp4=[128, 64, 32, 9], return_att=False)


def qa_channel(qa: np.ndarray) -> np.ndarray:
    """(T, F, P) int16 percent synthetic -> float32 in [0.01, 1]; nodata -> 1 (unknown treated as synthetic)."""
    q = qa.astype(np.float32)
    q[qa == QA_NODATA] = 100.0
    return np.clip(q, 1, 100) / 100.0


def permuted_qa(q: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per-field fixed permutation of the day axis. q (T, F, P)."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(q)
    for f in range(q.shape[1]):
        out[:, f] = q[rng.permutation(q.shape[0]), f]
    return out


class FieldTensor:
    """Holds one tile-year on the GPU: sr (F, T, 4, P) int16 and optional extra
    channel (F, T, P) float16, labels (F,) int64."""

    def __init__(self, npz: Path, condition: str, device: str = "cuda"):
        z = np.load(npz)
        self.sr = torch.from_numpy(np.ascontiguousarray(z["sr"].transpose(1, 0, 3, 2))).to(device)  # F,T,4,P
        lab = np.array([CROP_IDS.index(int(c)) for c in z["crop_id"]])
        self.y = torch.from_numpy(lab).long().to(device)
        self.fid = z["fid"]
        self.extra = None
        if condition in ("A1", "A2"):
            q = qa_channel(z["qa"])
            if condition == "A2":
                q = permuted_qa(q)
            self.extra = torch.from_numpy(np.ascontiguousarray(q.transpose(1, 0, 2))).half().to(device)  # F,T,P
        self.device = device

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def batch(self, idx: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        b = idx.shape[0]
        P = self.sr.shape[-1]
        pix = torch.randint(0, P, (b, N_PIXELS), generator=gen, device="cpu").to(self.device)
        sr = self.sr[idx]  # b,T,4,P
        g = pix[:, None, None, :].expand(b, sr.shape[1], sr.shape[2], N_PIXELS)
        x = torch.gather(sr, 3, g).float() * 1e-4
        mu = 0.1014 + 0.01 * torch.randn(b, 1, 1, 1, generator=gen).to(self.device)
        sd = 0.1171 + 0.01 * torch.randn(b, 1, 1, 1, generator=gen).to(self.device)
        x = (x - mu) / sd
        if self.extra is not None:
            e = torch.gather(self.extra[idx], 2, pix[:, None, :].expand(b, sr.shape[1], N_PIXELS)).float()
            x = torch.cat([x, e[:, :, None, :]], dim=2)
        return x, self.y[idx]


def authors_split(n: int, validation_split: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """train_psetae.py: indices shuffled with RandomState(0), first 25 % validation."""
    idx = list(range(n))
    np.random.RandomState(0).shuffle(idx)
    split = int(np.floor(validation_split * n))
    return np.array(idx[split:]), np.array(idx[:split])
