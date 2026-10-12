"""Heads on frozen MiniRocket features.

GatedHead  L-TAE-S's sparse feature gate (models_arch.SparseFeatureGate) transplanted onto
           the 9,996 MiniRocket features, followed by L-TAE-S's downstream layers.
           gate='sparsemax' -> MiniRocket-S; gate='softmax' -> dense control (identical
           network, dense masks). The gate is computed in fp32: sparsemax's cumulative
           sum over ~10k features is not accurate in fp16 (L-TAE-S gates only 128 channels).
L1Head     multinomial logistic regression with scikit-learn's L1 objective,
           C * sum_i w_{y_i} CE_i + ||W||_1 (bias unpenalised, balanced class weights),
           fitted by FISTA with backtracking.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402,F401  (puts experiments/field_reduction on sys.path)
from models_arch import sparsemax  # noqa: E402


class _Sparsemax(torch.autograd.Function):
    """models_arch.sparsemax with its closed-form backward (Martins & Astudillo, 2016):
    grad_z = g - mean_S(g) on the support S, 0 elsewhere. Same values and gradients as
    autograd through sort/cumsum, but only the output is stored, so 16 heads x 2048
    samples x 9,996 features fit in GPU memory."""

    @staticmethod
    def forward(ctx, z):
        out = sparsemax(z, dim=-1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        (out,) = ctx.saved_tensors
        support = out > 0
        g_mean = (g * support).sum(-1, keepdim=True) / support.sum(-1, keepdim=True).clamp(min=1)
        return (g - g_mean) * support


class GatedHead(nn.Module):
    def __init__(self, n_features, num_classes, gate="sparsemax", n_head=16, d_k=8,
                 bottleneck=64, gamma=1.5, dropout=0.3):
        super().__init__()
        assert gate in ("sparsemax", "softmax")
        self.gate, self.n_head, self.gamma = gate, n_head, gamma
        # Gate logits: shared bottleneck, then one projection + BN per head (TabNet-style;
        # a full n_features x n_features map per head would be ~100M parameters).
        self.gate_in = nn.Linear(n_features, bottleneck)
        self.gate_fc = nn.ModuleList([nn.Linear(bottleneck, n_features, bias=False) for _ in range(n_head)])
        self.gate_bn = nn.ModuleList([nn.BatchNorm1d(n_features) for _ in range(n_head)])
        # Per-head projection of the masked features (L-TAE-S: per-head d_k = 8 outputs)
        self.proj = nn.ModuleList([nn.Linear(n_features, d_k) for _ in range(n_head)])
        # Downstream exactly as LTAESparse
        self.norm = nn.LayerNorm(n_head * d_k)
        self.mlp = nn.Sequential(
            nn.Linear(n_head * d_k, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(128, num_classes)

    def masks(self, F):
        """Yield (h, mask_h) per head, with the L-TAE-S prior update, in fp32."""
        with torch.autocast("cuda", enabled=False):
            F = F.float()
            g = self.gate_in(F)
            prior = torch.ones_like(F)
            for h in range(self.n_head):
                logits = self.gate_bn[h](self.gate_fc[h](g)) * prior
                mask = _Sparsemax.apply(logits) if self.gate == "sparsemax" else torch.softmax(logits, dim=-1)
                prior = prior * torch.clamp(self.gamma - mask, min=0)
                yield h, mask

    def forward(self, F):
        Ff = F.float()
        outs = [self.proj[h](Ff * mask) for h, mask in self.masks(F)]
        out = self.mlp(self.norm(torch.cat(outs, dim=-1)))
        return self.classifier(out), {}

    @torch.no_grad()
    def mask_stats(self, loader, device):
        """Mean non-zero features per head per sample, mean effective number
        exp(entropy) per head per sample, and the union of features ever selected."""
        self.eval()
        nnz, eff, n = 0.0, 0.0, 0
        used = None
        for F, _ in loader:
            F = F.to(device)
            for _, m in self.masks(F):
                nz = m > 0
                nnz += nz.sum().item()
                eff += torch.exp(-(m * torch.log(m.clamp_min(1e-30))).sum(-1)).sum().item()
                u = nz.any(0)
                used = u if used is None else used | u
            n += len(F)
        return {"mean_nonzero_per_head": nnz / (n * self.n_head),
                "mean_effective_features_per_head": eff / (n * self.n_head),
                "union_features_selected": int(used.sum().item())}


class L1Head:
    """FISTA for sklearn's L1 multinomial logistic objective (scaled by 1/n)."""

    def __init__(self, C, max_iter=20000, tol=1e-10, device="cpu"):
        # tol is on the relative objective change; 1e-6 stopped early on field data
        # (objective within 0.2% but selected features inflated 2-3x), see README.
        self.C, self.max_iter, self.tol, self.device = C, max_iter, tol, device

    def fit(self, F, y, num_classes, W0=None, b0=None):
        dev = self.device
        X = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)).to(dev)
        yt = torch.from_numpy(y.astype(np.int64)).to(dev)
        n, d = X.shape
        counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        cw = torch.tensor(n / (num_classes * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
        lam = 1.0 / n  # L1 weight after dividing the sklearn objective by n

        def smooth(W, b):
            ce = torch.nn.functional.cross_entropy(X @ W + b, yt, weight=cw, reduction="none")
            return self.C * ce.sum() / n

        def grad(W, b):
            with torch.enable_grad():
                W, b = W.detach().requires_grad_(True), b.detach().requires_grad_(True)
                f = smooth(W, b)
                gW, gb = torch.autograd.grad(f, (W, b))
            return f.detach(), gW, gb

        W = torch.zeros(d, num_classes, device=dev) if W0 is None else torch.from_numpy(W0).clone().to(dev)
        b = torch.zeros(num_classes, device=dev) if b0 is None else torch.from_numpy(b0).clone().to(dev)
        yW, yb, t, L = W.clone(), b.clone(), 1.0, 1.0
        obj_prev = None
        with torch.no_grad():
            for it in range(1, self.max_iter + 1):
                f_y, gW, gb = grad(yW, yb)
                while True:  # backtracking on the Lipschitz estimate
                    zW = yW - gW / L
                    Wn = torch.sign(zW) * torch.clamp(zW.abs() - lam / L, min=0)
                    bn = yb - gb / L
                    dW, db = Wn - yW, bn - yb
                    quad = f_y + (gW * dW).sum() + (gb * db).sum() + L / 2 * ((dW ** 2).sum() + (db ** 2).sum())
                    if smooth(Wn, bn) <= quad + 1e-12:
                        break
                    L *= 2.0
                t_n = (1 + (1 + 4 * t * t) ** 0.5) / 2
                yW = Wn + (t - 1) / t_n * (Wn - W)
                yb = bn + (t - 1) / t_n * (bn - b)
                obj = (smooth(Wn, bn) + lam * Wn.abs().sum()).item()
                W, b, t = Wn, bn, t_n
                L = max(L / 1.5, 1e-8)  # allow the step to grow again
                if obj_prev is not None and abs(obj_prev - obj) <= self.tol * max(1.0, abs(obj)):
                    break
                obj_prev = obj
        self.n_iter_, self.converged_ = it, it < self.max_iter
        self.W, self.b = W.cpu(), b.cpu()
        del X
        self.n_selected_ = int((W.abs().sum(1) > 0).sum().item())
        return self

    def predict_proba(self, F):
        with torch.no_grad():
            return torch.softmax(torch.from_numpy(F) @ self.W + self.b, dim=1).numpy()
