"""Model loading and the authors' PAD test metric, for frozen-checkpoint
evaluation of Sen4AgriNet (S4A-Models ae5a376).

The authors' test_step (identical in PAD_unet.py, PAD_convLSTM.py and
PAD_convSTAR.py) does

    pred = self(inputs).to(torch.long)      # log-softmax cast to long: truncation toward zero
    pred = torch.exp(pred)                  # so every class with log p > -1 becomes 1.0
    pred = torch.clamp(pred, 0, max(label)) # no effect, values are <= 1
    pred[~mask_K] = 0; label[~mask] = 0     # mask = parcel pixel and encoded label != 0
    pred_sparse = pred.argmax(axis=1)       # ties go to the lowest class index, including 0

and test_epoch_end drops row and column 0 of the confusion matrix before
computing per-class F1, so a parcel pixel whose truncated argmax is 0
(background) is removed from the metric rather than counted as an error.
`authors_confusion` reproduces this exactly; `corrected_confusion` takes the
float argmax and keeps background predictions as errors. Both are reported.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUBS = HERE / "stubs"
REPO = Path("/mnt/bigdrive/external_audits/sen4agrinet_rerun/repo")
ARCHS = {"unet": ("model.PAD_unet", "UNet"), "convlstm": ("model.PAD_convLSTM", "ConvLSTM"),
         "convstar": ("model.PAD_convSTAR", "ConvSTAR")}
CKPT_DIR = {"unet": "UNet", "convlstm": "ConvLSTM", "convstar": "ConvSTAR"}


@contextmanager
def _in_repo():
    old = os.getcwd()
    os.chdir(REPO)
    for p in (str(REPO), str(STUBS)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    try:
        yield
    finally:
        os.chdir(old)


def load_config():
    """The authors' utils/settings/config.py (relative mapping paths, so it is
    imported from the repo root). Returns LINEAR_ENCODER and BANDS."""
    with _in_repo():
        from utils.settings import config  # noqa: PLC0415
    return config


def load_model(arch: str, ckpt: Path, device: str = "cuda") -> torch.nn.Module:
    import importlib  # noqa: PLC0415

    cfg = load_config()
    with _in_repo():
        mod = importlib.import_module(ARCHS[arch][0])
        cls = getattr(mod, ARCHS[arch][1])
        kw = {"num_layers": 3} if arch == "unet" else {}
        model = cls(run_path="/tmp/s4a_unused", linear_encoder=cfg.LINEAR_ENCODER, parcel_loss=True, **kw)
    sd = load_checkpoint(ckpt)["state_dict"]
    # the released runs used the weighted loss (--weighted_loss): its class-weight
    # buffer is in the state dict but plays no part in a forward pass
    res = model.load_state_dict({k: v for k, v in sd.items() if k != "lossfunction.weight"}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return model.eval().to(device)


class _Placeholder:
    """Stands in for pickled pytorch-lightning callback and trainer objects."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


def _pickle_module():
    import pickle as _pickle  # noqa: PLC0415
    import types  # noqa: PLC0415

    class Unpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(("pytorch_lightning", "torchmetrics", "lightning")):
                return _Placeholder
            return super().find_class(module, name)

    mod = types.SimpleNamespace(**{k: getattr(_pickle, k) for k in dir(_pickle) if not k.startswith("__")})
    mod.Unpickler = Unpickler
    mod.__name__ = "pickle_placeholder"
    return mod


def load_checkpoint(ckpt: Path) -> dict:
    """torch.load of a Lightning 1.6 checkpoint without Lightning installed:
    callback and loop objects become inert placeholders; tensors load as usual."""
    return torch.load(ckpt, map_location="cpu", weights_only=False, pickle_module=_pickle_module())


def checkpoint_path(root: Path, arch: str, scenario: int) -> Path:
    found = sorted((root / "model_weights" / "PAD" / CKPT_DIR[arch] / f"Scenario {scenario}").glob("*.ckpt"))
    if len(found) != 1:
        raise FileNotFoundError(f"{arch} scenario {scenario}: {found}")
    return found[0]


def forward(model: torch.nn.Module, arch: str, medians: torch.Tensor) -> torch.Tensor:
    """medians (B, T=6, C=4, H, W) float32, normalised. Returns log-probabilities
    (B, K, H, W). U-Net concatenates time along channels as in its test_step;
    the recurrent models take the 5-D tensor."""
    if arch == "unet":
        b, t, c, h, w = medians.shape
        return model(medians.reshape(b, t * c, h, w))
    return model(medians)


def authors_confusion(logp: torch.Tensor, label: torch.Tensor, parcels: torch.Tensor, k: int) -> np.ndarray:
    pred = torch.exp(logp.to(torch.long).to(torch.float32))
    pred = torch.clamp(pred, 0, k - 1)
    mask = parcels & (label != 0)
    lab = label.clone()
    lab[~mask] = 0
    pred = pred.clone()
    pred[~mask[:, None].expand_as(pred)] = 0
    ps = pred.argmax(dim=1)
    return _bincount(lab.flatten(), ps.flatten(), k)


def corrected_confusion(logp: torch.Tensor, label: torch.Tensor, parcels: torch.Tensor, k: int) -> np.ndarray:
    mask = parcels & (label != 0)
    ps = logp.argmax(dim=1)
    return _bincount(label[mask], ps[mask], k)


def _bincount(lab: torch.Tensor, pred: torch.Tensor, k: int) -> np.ndarray:
    idx = (lab.to(torch.int64) * k + pred.to(torch.int64)).cpu()
    return torch.bincount(idx, minlength=k * k).reshape(k, k).numpy().astype(np.int64)


def authors_weighted_f1(cm: np.ndarray) -> float:
    """test_epoch_end arithmetic: drop label and prediction 0, per-class F1,
    weighted by class sample count. NaN if any class has no prediction and no
    sample, exactly as the authors' sum would be."""
    c = cm[1:, 1:].astype(float)
    tp = np.diag(c)
    fp = c.sum(0) - tp
    fn = c.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        ppv = tp / (tp + fp)
        tpr = tp / (tp + fn)
        f1 = 2 * ppv * tpr / (ppv + tpr)
    n = c.sum(1)
    return float(((f1 * n) / n.sum()).sum())


def corrected_weighted_f1(cm: np.ndarray) -> float:
    """Background predictions on parcel pixels count as errors (they stay in
    the row sums); classes with no prediction get F1 0 instead of NaN."""
    c = cm.astype(float)
    tp = np.diag(c)[1:]
    fp = c.sum(0)[1:] - tp
    fn = c.sum(1)[1:] - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    n = c.sum(1)[1:]
    return float((f1 * n).sum() / n.sum())


def corrected_macro_f1(cm: np.ndarray) -> float:
    c = cm.astype(float)
    tp = np.diag(c)[1:]
    fp = c.sum(0)[1:] - tp
    fn = c.sum(1)[1:] - tp
    present = c.sum(1)[1:] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return float(f1[present].mean())
