"""Tests for the InfoNCE content-tier contrastive loss."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.model import content_contrastive_loss


class TestInfoNCE(unittest.TestCase):
    def test_empty_mask_returns_zero(self):
        logits = torch.randn(2, 4, 8)
        y = torch.zeros(2, 4, dtype=torch.long)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loss = content_contrastive_loss(logits, y, mask, k=4)
        self.assertEqual(loss.item(), 0.0)

    def test_returns_nonnegative(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 16)
        y = torch.randint(1, 16, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss = content_contrastive_loss(logits, y, mask, k=4)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_perfect_prediction_low_loss(self):
        torch.manual_seed(0)
        v = 1024
        logits = torch.full((1, 2, v), -10.0)
        y = torch.tensor([[3, 7]])
        for t in range(2):
            logits[0, t, y[0, t]] = 100.0
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss = content_contrastive_loss(logits, y, mask, k=4)
        self.assertLess(loss.item(), 0.05)

    def test_chunked_returns_finite(self):
        torch.manual_seed(0)
        full = torch.randn(2, 8, 16)
        y = torch.randint(1, 16, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.bool)
        chunked = [full[:, :4], full[:, 4:]]
        chunked_loss = content_contrastive_loss(chunked, y, mask, k=4)
        self.assertTrue(torch.isfinite(chunked_loss))
        self.assertGreaterEqual(chunked_loss.item(), 0.0)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 16, requires_grad=True)
        y = torch.randint(1, 16, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss = content_contrastive_loss(logits, y, mask, k=4)
        loss.backward()
        self.assertTrue((logits.grad.abs().sum() > 0).item())


if __name__ == "__main__":
    unittest.main()
