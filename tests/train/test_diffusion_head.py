"""Tests for DiffusionContentHead + discrete_diffusion_content_loss."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.model.heads_diffusion import (
    DiffusionContentHead,
    _sinusoidal_time_embed,
)
from preframr.train.model.losses_diffusion import (
    discrete_diffusion_content_loss,
    sample_mask_schedule,
)


def _toy_head(d=16, vocab=12, t_max=8, seed=0):
    torch.manual_seed(seed)
    return DiffusionContentHead(d=d, vocab_size=vocab, t_max=t_max)


class TestSinusoidalTimeEmbed(unittest.TestCase):
    def test_shape(self):
        t = torch.arange(5)
        emb = _sinusoidal_time_embed(t, 16)
        self.assertEqual(tuple(emb.shape), (5, 16))

    def test_deterministic(self):
        t = torch.tensor([1, 3, 7])
        a = _sinusoidal_time_embed(t, 8)
        b = _sinusoidal_time_embed(t, 8)
        self.assertTrue(torch.allclose(a, b))

    def test_different_t_different_emb(self):
        a = _sinusoidal_time_embed(torch.tensor([1]), 16)
        b = _sinusoidal_time_embed(torch.tensor([5]), 16)
        self.assertGreater((a - b).abs().sum().item(), 0.0)


class TestDiffusionContentHeadShape(unittest.TestCase):
    def test_inference_forward_shape(self):
        head = _toy_head()
        h = torch.randn(3, 7, 16)
        log_p = head(h)
        self.assertEqual(tuple(log_p.shape), (3, 7, 12))

    def test_inference_forward_normalised(self):
        head = _toy_head()
        h = torch.randn(2, 4, 16)
        log_p = head(h)
        row_sums = log_p.exp().sum(dim=-1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
            msg=f"inference log_p not normalised; row_sums={row_sums}",
        )

    def test_training_forward_includes_mask_class(self):
        head = _toy_head()
        h = torch.randn(2, 4, 16)
        x_t = torch.randint(0, head.vocab_size + 1, (2, 4))
        t = torch.tensor([3, 5])
        log_p = head.forward_with_mask_class(h, x_t, t)
        self.assertEqual(tuple(log_p.shape), (2, 4, 13))
        row_sums = log_p.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))


class TestDiffusionContentHeadInputs(unittest.TestCase):
    def test_per_batch_t_broadcasts(self):
        head = _toy_head()
        h = torch.randn(2, 4, 16)
        x_t = torch.zeros((2, 4), dtype=torch.long)
        t_per_batch = torch.tensor([1, 5])
        log_p = head.forward_with_mask_class(h, x_t, t_per_batch)
        self.assertEqual(tuple(log_p.shape), (2, 4, 13))

    def test_per_position_t(self):
        head = _toy_head()
        h = torch.randn(2, 4, 16)
        x_t = torch.zeros((2, 4), dtype=torch.long)
        t = torch.randint(1, 9, (2, 4))
        log_p = head.forward_with_mask_class(h, x_t, t)
        self.assertEqual(tuple(log_p.shape), (2, 4, 13))


class TestSampleMaskSchedule(unittest.TestCase):
    def test_returns_correct_shapes(self):
        content_mask = torch.tensor([[True, True, False], [False, True, True]])
        t, m = sample_mask_schedule(content_mask, t_max=8)
        self.assertEqual(tuple(t.shape), (2, 3))
        self.assertEqual(tuple(m.shape), (2, 3))
        self.assertEqual(t.dtype, torch.long)
        self.assertEqual(m.dtype, torch.bool)

    def test_non_content_positions_never_masked(self):
        content_mask = torch.tensor([[True, False, True, False]])
        for _ in range(20):
            _, m = sample_mask_schedule(content_mask, t_max=8)
            self.assertFalse(m[~content_mask].any().item())

    def test_t_range_in_valid_bounds(self):
        content_mask = torch.ones(64, dtype=torch.bool)
        t, _ = sample_mask_schedule(content_mask, t_max=8)
        self.assertTrue((t[content_mask] >= 1).all().item())
        self.assertTrue((t[content_mask] <= 8).all().item())

    def test_empty_content_mask_no_error(self):
        content_mask = torch.zeros(3, dtype=torch.bool)
        t, m = sample_mask_schedule(content_mask, t_max=8)
        self.assertFalse(m.any().item())
        self.assertEqual(int(t.max().item()), 8)

    def test_generator_reproducible(self):
        content_mask = torch.ones(32, dtype=torch.bool)
        g1 = torch.Generator()
        g1.manual_seed(42)
        g2 = torch.Generator()
        g2.manual_seed(42)
        t1, m1 = sample_mask_schedule(content_mask, t_max=8, generator=g1)
        t2, m2 = sample_mask_schedule(content_mask, t_max=8, generator=g2)
        self.assertTrue(torch.equal(t1, t2))
        self.assertTrue(torch.equal(m1, m2))


class TestDiscreteDiffusionContentLoss(unittest.TestCase):
    def _setup(self, vocab=12, n_vocab_full=20):
        head = _toy_head(vocab=vocab)
        h = torch.randn(2, 6, 16)
        gt = torch.randint(0, n_vocab_full, (2, 6))
        content_mask = torch.zeros(2, 6, dtype=torch.bool)
        content_mask[:, ::2] = True
        full_to_local = torch.full((n_vocab_full,), -1, dtype=torch.long)
        in_tier = torch.arange(vocab) + 4
        full_to_local[in_tier] = torch.arange(vocab)
        gt = gt.clone()
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                if content_mask[i, j]:
                    gt[i, j] = in_tier[(i + j) % vocab]
        return head, h, gt, content_mask, full_to_local

    def test_returns_scalar(self):
        head, h, gt, content_mask, ftl = self._setup()
        g = torch.Generator()
        g.manual_seed(0)
        loss = discrete_diffusion_content_loss(
            head, h, gt, content_mask, ftl, t_max=8, generator=g
        )
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_returns_zero_when_no_content(self):
        head, h, gt, _, ftl = self._setup()
        empty = torch.zeros_like(gt, dtype=torch.bool)
        loss = discrete_diffusion_content_loss(head, h, gt, empty, ftl, t_max=8)
        self.assertEqual(loss.item(), 0.0)

    def test_gradient_flows(self):
        head, h, gt, content_mask, ftl = self._setup()
        g = torch.Generator()
        g.manual_seed(0)
        loss = discrete_diffusion_content_loss(
            head, h, gt, content_mask, ftl, t_max=8, generator=g
        )
        if loss.item() == 0.0:
            return
        loss.backward()
        self.assertIsNotNone(head.denoiser.weight.grad)
        self.assertTrue(torch.isfinite(head.denoiser.weight.grad).all())

    def test_loss_reproducible_with_generator(self):
        head, h, gt, content_mask, ftl = self._setup()
        g1 = torch.Generator()
        g1.manual_seed(7)
        g2 = torch.Generator()
        g2.manual_seed(7)
        l1 = discrete_diffusion_content_loss(
            head, h, gt, content_mask, ftl, t_max=8, generator=g1
        )
        l2 = discrete_diffusion_content_loss(
            head, h, gt, content_mask, ftl, t_max=8, generator=g2
        )
        self.assertAlmostEqual(l1.item(), l2.item(), places=5)


if __name__ == "__main__":
    unittest.main()
