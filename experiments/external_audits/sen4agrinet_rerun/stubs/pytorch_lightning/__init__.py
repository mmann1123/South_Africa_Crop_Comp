"""Minimal stand-in for pytorch_lightning: LightningModule is torch.nn.Module
with no-op logging and hyperparameter saving. Inference only."""
import torch


class LightningModule(torch.nn.Module):
    def save_hyperparameters(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


def seed_everything(seed=None, *args, **kwargs):
    return seed
