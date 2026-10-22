"""Training-time month dropout: the standard remedy for missing observations, used as the
control for the sparse gate's robustness to a zero-filled month.

``MonthDropout(model, p, fill)`` wraps a model whose input is (B, T, C) (time steps x bands).
In training mode each (sample, month) is dropped independently with probability ``p``: all C
bands of that month are replaced by ``fill`` (a (T, C) tensor), which should be exactly what a
missing month looks like at inference in that model's pipeline. For the published pipelines a
missing month is raw reflectance 0, so ``fill`` is the scaled zero (``-mean/std`` per feature)
for scaled models and 0 for unscaled ones; ``scaled_zero_fill`` builds it. In eval mode the
wrapper is a no-op.

state_dict / load_state_dict delegate to the wrapped model, so checkpoints are ordinary
checkpoints of the inner model and existing inference code loads them unchanged.
"""
import numpy as np
import torch
import torch.nn as nn


class MonthDropout(nn.Module):
    def __init__(self, model, p, fill):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.net = model
        self.p = float(p)
        self.register_buffer("fill", torch.as_tensor(np.asarray(fill), dtype=torch.float32), persistent=False)

    def forward(self, x, *args, **kwargs):
        if self.training and self.p > 0:
            drop = torch.rand(x.shape[0], x.shape[1], device=x.device) < self.p
            x = torch.where(drop[..., None], self.fill.to(x.dtype)[None], x)
        return self.net(x, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.net.load_state_dict(state_dict, *args, **kwargs)


def scaled_zero_fill(scaler, n_features, to_seq):
    """(T, C) array that a missing month (raw 0) becomes after the model's preprocessing.

    scaler : fitted StandardScaler, or None for models trained on raw reflectance.
    to_seq : function mapping a flat (N, n_features) array to (N, T, C) in the model's layout.
    """
    z = np.zeros((1, n_features), dtype=np.float64)
    if scaler is not None:
        z = np.nan_to_num(scaler.transform(z), nan=0.0, posinf=0.0, neginf=0.0)
    return to_seq(z.astype(np.float32))[0]
