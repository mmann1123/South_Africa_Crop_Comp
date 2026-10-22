"""Unit tests for deep_learn/src/month_dropout.py. Run: python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deep_learn", "src"))
from month_dropout import MonthDropout, scaled_zero_fill  # noqa: E402

T, C = 10, 6


class Echo(nn.Module):
    """Returns its input; has one parameter so state_dict is non-empty."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class Pair(nn.Module):
    """Returns (logits, aux) like L-TAE-S."""
    def forward(self, x):
        return x.sum(), {"masks": None}


def time_major(z):
    return z.reshape(-1, T, C)


class MonthDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4000, T, C) + 5.0  # no value equals the fill by chance
        self.fill = torch.arange(T * C, dtype=torch.float32).reshape(T, C) * -1.0

    def test_eval_mode_is_identity(self):
        m = MonthDropout(Echo(), 0.5, self.fill).eval()
        self.assertTrue(torch.equal(m(self.x), self.x))

    def test_whole_months_replaced_with_fill(self):
        m = MonthDropout(Echo(), 0.3, self.fill).train()
        out = m(self.x)
        changed = (out != self.x)                      # (N, T, C)
        per_month = changed.any(-1)
        # a dropped month has every band replaced; a kept month has none changed
        self.assertTrue(torch.equal(per_month, changed.all(-1)))
        n, t = torch.nonzero(per_month, as_tuple=True)
        self.assertTrue(torch.equal(out[n, t], self.fill[t]))

    def test_drop_rate(self):
        m = MonthDropout(Echo(), 0.2, self.fill).train()
        rate = (m(self.x) != self.x).any(-1).float().mean().item()
        self.assertAlmostEqual(rate, 0.2, delta=0.01)

    def test_input_not_modified_in_place(self):
        before = self.x.clone()
        MonthDropout(Echo(), 0.5, self.fill).train()(self.x)
        self.assertTrue(torch.equal(self.x, before))

    def test_p_zero_is_identity_in_training(self):
        m = MonthDropout(Echo(), 0.0, self.fill).train()
        self.assertTrue(torch.equal(m(self.x), self.x))

    def test_bad_p_raises(self):
        with self.assertRaises(ValueError):
            MonthDropout(Echo(), 1.0, self.fill)

    def test_state_dict_is_the_inner_models(self):
        inner = Echo()
        m = MonthDropout(inner, 0.2, self.fill)
        self.assertEqual(list(m.state_dict()), list(inner.state_dict()))  # no "net." prefix, no fill
        fresh = Echo()
        with torch.no_grad():
            fresh.w.fill_(3.0)
        m.load_state_dict(fresh.state_dict())
        self.assertEqual(inner.w.item(), 3.0)

    def test_tuple_outputs_pass_through(self):
        out = MonthDropout(Pair(), 0.2, self.fill).eval()(self.x)
        self.assertIsInstance(out, tuple)

    def test_parameters_are_trained(self):
        inner = Echo()
        m = MonthDropout(inner, 0.2, self.fill)
        self.assertEqual([id(p) for p in m.parameters()], [id(p) for p in inner.parameters()])


class ScaledZeroFillTests(unittest.TestCase):
    def test_scaled_zero_matches_scaler_on_raw_zero(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(0.3, 0.1, size=(500, T * C))
        sc = StandardScaler().fit(raw)
        fill = scaled_zero_fill(sc, T * C, time_major)
        np.testing.assert_allclose(fill, (-sc.mean_ / sc.scale_).reshape(T, C), rtol=1e-6)

    def test_unscaled_is_zero(self):
        self.assertTrue(np.array_equal(scaled_zero_fill(None, T * C, time_major), np.zeros((T, C))))

    def test_layout_follows_to_seq(self):
        # band-major flat layout (CNN-BiLSTM): feature f = band * T + month
        def band_major(z):
            return z.reshape(-1, C, T).transpose(0, 2, 1)
        raw = np.tile(np.repeat(np.arange(C, dtype=float), T), (2, 1))  # feature value = band index
        raw[1] += 1.0
        sc = StandardScaler().fit(raw)  # mean = band + 0.5, scale = 0.5
        fill = scaled_zero_fill(sc, T * C, band_major)
        np.testing.assert_allclose(fill[0], -(np.arange(C) + 0.5) / 0.5)
        np.testing.assert_allclose(fill[:, 2], np.full(T, -(2 + 0.5) / 0.5))


if __name__ == "__main__":
    unittest.main()
