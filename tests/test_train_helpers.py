"""Regression tests for experiments/sparse_gate_backbones/train.py helpers.

logits_of must run without autograd: a misplaced @torch.no_grad() (2026-10-21) made evaluation
keep every batch's graph alive, so GPU memory grew with the number of samples and validation on
1.2M pixels ran out of memory. Run: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments",
                                "sparse_gate_backbones"))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (train.py runs on the GPU)")
class LogitsOf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import train as tr
        cls.tr = tr
        cls.model = tr.TempCNN(tr.C, 5).to(tr.device)

    def test_no_autograd(self):
        X = np.random.randn(512, self.tr.T, self.tr.C).astype(np.float32)
        out, _ = self.tr.logits_of(self.model, X, 256)
        self.assertFalse(out.requires_grad)

    def test_memory_does_not_grow_with_samples(self):
        torch.cuda.empty_cache()
        peaks = []
        for n in (20_000, 80_000):
            X = np.random.randn(n, self.tr.T, self.tr.C).astype(np.float32)
            torch.cuda.reset_peak_memory_stats()
            self.tr.logits_of(self.model, X, 2048)
            peaks.append(torch.cuda.max_memory_allocated())
        self.assertLess(peaks[1], 1.5 * peaks[0])

    def test_gate_of(self):
        from month_dropout import MonthDropout
        g = self.tr.Gated("tempcnn", "sparsemax", 5)
        self.assertIs(self.tr.gate_of(g), g.gate)
        self.assertIs(self.tr.gate_of(MonthDropout(g, 0.2, np.zeros((10, 6), np.float32))), g.gate)
        self.assertIsNone(self.tr.gate_of(self.model))


if __name__ == "__main__":
    unittest.main()
