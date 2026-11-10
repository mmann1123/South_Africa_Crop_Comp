"""Minimal stand-in for tensorboardX.SummaryWriter (never used at inference)."""


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None
