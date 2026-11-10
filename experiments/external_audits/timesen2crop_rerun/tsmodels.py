"""LSTM and TempCNN for the 12-month input, the paper's search ranges, and the fill registry."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import tslib as tl


class LSTM(nn.Module):
    def __init__(self, d: int, n_classes: int, hidden: int, layers: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True, dropout=dropout)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))

    def forward(self, x):
        h, _ = self.lstm(self.norm(x))
        return self.out(h[:, -1])


class TempCNN(nn.Module):
    """Pelletier et al. 2019: three conv blocks, then a dense layer."""

    def __init__(self, d: int, n_classes: int, hidden: int, kernel: int, dropout: float = 0.2, t: int = 12):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv1d(a, b, kernel, padding=kernel // 2), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout))

        self.conv = nn.Sequential(block(d, hidden), block(hidden, hidden), block(hidden, hidden))
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(hidden * t, 4 * hidden), nn.BatchNorm1d(4 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(4 * hidden, n_classes))

    def forward(self, x):
        return self.out(self.conv(x.transpose(1, 2)))


def build(name: str, hp: dict, d: int) -> nn.Module:
    if name == "lstm":
        return LSTM(d, 16, hp["hidden"], hp["layers"])
    return TempCNN(d, 16, hp["hidden"], hp["kernel"])


def sample_hp(name: str, rng: np.random.Generator) -> dict:
    hp = {"lr": float(10 ** rng.uniform(-4, -2)), "wd": float(10 ** rng.uniform(-8, -2)), "hidden": int(rng.choice([32, 64, 128]))}
    if name == "lstm":
        hp["layers"] = int(rng.choice([2, 3, 4]))
    else:
        hp["kernel"] = int(rng.choice([3, 5, 7]))
    return hp


def make_input(C: np.ndarray, fill: str, med: np.ndarray) -> np.ndarray:
    """(N, 12, 9) with NaN gaps -> model input. Z zero, M training medians, L linear, Zm zero plus present mask."""
    if fill == "Z":
        return tl.fill_zero(C)
    if fill == "M":
        return tl.fill_median(C, med)
    if fill == "L":
        return tl.fill_linear(C)
    if fill == "Zm":
        return np.concatenate([tl.fill_zero(C), tl.present_mask(C)[..., None].astype(np.float32)], axis=2)
    raise ValueError(fill)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, bs: int = 8192) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + bs]).cuda()).argmax(1).cpu().numpy() for i in range(0, len(x), bs)])


def train(model: nn.Module, x, y, xv, yv, hp: dict, epochs: int, seed: int, bs: int = 1024) -> tuple[float, dict]:
    """x, y on GPU. Cross-entropy, Adam; returns best validation accuracy and its state."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    loss = nn.CrossEntropyLoss()
    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g, device="cuda")
        for i in range(0, len(x), bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss(model(x[j]), y[j]).backward()
            opt.step()
        acc = (predict(model, xv) == yv).mean()
        if acc > best:
            best, state = float(acc), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best, state
