"""Tests for per-tier output heads, MoS content head, and router marginalisation."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.model import (
    MoSHead,
    PerTierHeads,
    _LOSS_TIER_ORDER,
    _mos_log_mixture,
    per_tier_unified_log_p,
)


def _toy_partition(n_vocab=100, sizes=(50, 30, 15, 5)):
    assert sum(sizes) == n_vocab
    partition = {}
    offset = 0
    for tier_name, size in zip(_LOSS_TIER_ORDER, sizes):
        in_tier = torch.arange(offset, offset + size)
        ftl = torch.full((n_vocab,), -1, dtype=torch.long)
        ftl[in_tier] = torch.arange(size)
        partition[tier_name] = (in_tier, ftl)
        offset += size
    return partition


class TestMoSHead(unittest.TestCase):
    def test_output_shape(self):
        h = torch.randn(2, 5, 32)
        head = MoSHead(d=32, vocab_size=16, k=4)
        log_p, gate_log_p = head(h)
        self.assertEqual(tuple(log_p.shape), (2, 5, 16))
        self.assertEqual(tuple(gate_log_p.shape), (2, 5, 4))

    def test_log_probs_normalised(self):
        torch.manual_seed(0)
        head = MoSHead(d=8, vocab_size=10, k=3)
        h = torch.randn(2, 4, 8)
        log_p, _ = head(h)
        row_sums = log_p.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_gate_normalised(self):
        torch.manual_seed(0)
        head = MoSHead(d=8, vocab_size=10, k=3)
        h = torch.randn(2, 4, 8)
        _, gate_log_p = head(h)
        gate_sums = gate_log_p.exp().sum(dim=-1)
        self.assertTrue(
            torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)
        )

    def test_gradient_flow(self):
        torch.manual_seed(0)
        head = MoSHead(d=8, vocab_size=10, k=3)
        h = torch.randn(2, 4, 8, requires_grad=True)
        log_p, _ = head(h)
        target = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        loss = torch.nn.functional.nll_loss(log_p.reshape(-1, 10), target.reshape(-1))
        loss.backward()
        self.assertTrue((h.grad.abs().sum() > 0).item())

    def test_bf16_input_preserves_reduced_precision_output(self):
        """Guard against autocast fp32-promotion: MoSHead must keep bf16/fp16 output dtype so the (B,T,K,V_c) component stack stays half-precision (it OOMed at prodlike at fp32)."""
        head = MoSHead(d=16, vocab_size=64, k=4).to(torch.bfloat16)
        h = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        log_p, gate_log_p = head(h)
        self.assertEqual(log_p.dtype, torch.bfloat16)
        self.assertEqual(gate_log_p.dtype, torch.bfloat16)


class TestMosLogMixture(unittest.TestCase):
    def test_equals_logsumexp_of_weighted_components(self):
        torch.manual_seed(0)
        comp_log_p = torch.log_softmax(torch.randn(2, 3, 4, 8), dim=-1)
        gate_log_p = torch.log_softmax(torch.randn(2, 3, 4), dim=-1)
        log_mix = _mos_log_mixture(comp_log_p, gate_log_p)
        ref = torch.log((gate_log_p.exp().unsqueeze(-1) * comp_log_p.exp()).sum(dim=-2))
        self.assertTrue(torch.allclose(log_mix, ref, atol=1e-5))


class TestPerTierHeadsForward(unittest.TestCase):
    def test_output_shapes(self):
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=4)
        h = torch.randn(2, 5, 16)
        out = heads(h)
        self.assertEqual(tuple(out["router"].shape), (2, 5, 4))
        self.assertEqual(tuple(out["structural"].shape), (2, 5, 50))
        self.assertEqual(tuple(out["mid"].shape), (2, 5, 30))
        self.assertEqual(tuple(out["content"].shape), (2, 5, 15))
        self.assertEqual(tuple(out["zero"].shape), (2, 5, 5))
        self.assertEqual(tuple(out["mos_gate_log_p"].shape), (2, 5, 4))

    def test_no_mos_when_k_zero(self):
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=0)
        h = torch.randn(2, 5, 16)
        out = heads(h)
        self.assertIsNone(out["mos_gate_log_p"])
        self.assertEqual(tuple(out["content"].shape), (2, 5, 15))


class TestPerTierUnifiedLogP(unittest.TestCase):
    def test_unified_normalised(self):
        torch.manual_seed(0)
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=4)
        h = torch.randn(2, 5, 16)
        out = heads(h)
        unified = per_tier_unified_log_p(out, partition, n_vocab=100)
        self.assertEqual(tuple(unified.shape), (2, 5, 100))
        rowsums = unified.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(rowsums, torch.ones_like(rowsums), atol=1e-4))

    def test_unified_matches_marginal(self):
        torch.manual_seed(1)
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=4)
        h = torch.randn(1, 1, 16)
        out = heads(h)
        unified = per_tier_unified_log_p(out, partition, n_vocab=100)
        router_p = out["router"].softmax(dim=-1)
        manual_p = torch.zeros(1, 1, 100)
        for tier_id, tier_name in enumerate(_LOSS_TIER_ORDER):
            in_tier, _ = partition[tier_name]
            if tier_name == "content":
                tier_p = out[tier_name].exp()
            else:
                tier_p = out[tier_name].softmax(dim=-1)
            manual_p[..., in_tier] = router_p[..., tier_id : tier_id + 1] * tier_p
        self.assertTrue(torch.allclose(unified.exp(), manual_p, atol=1e-5))

    def test_unified_argmax_gradient_flows(self):
        torch.manual_seed(0)
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=4)
        h = torch.randn(2, 5, 16, requires_grad=True)
        out = heads(h)
        unified = per_tier_unified_log_p(out, partition, n_vocab=100)
        target = torch.randint(0, 100, (2, 5))
        loss = torch.nn.functional.nll_loss(
            unified.reshape(-1, 100), target.reshape(-1)
        )
        loss.backward()
        self.assertTrue((h.grad.abs().sum() > 0).item())

    def test_unified_bf16_input_preserves_reduced_precision_output(self):
        """Guard against autocast fp32-promotion: per_tier_unified_log_p must keep bf16/fp16 dtype so the (B,T,V_full) unified buffer stays half-precision at prodlike."""
        torch.manual_seed(3)
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=4).to(torch.bfloat16)
        h = torch.randn(2, 5, 16, dtype=torch.bfloat16)
        out = heads(h)
        self.assertEqual(out["router"].dtype, torch.bfloat16)
        self.assertEqual(out["content"].dtype, torch.bfloat16)
        self.assertEqual(out["structural"].dtype, torch.bfloat16)
        unified = per_tier_unified_log_p(out, partition, n_vocab=100)
        self.assertEqual(unified.dtype, torch.bfloat16)

    def test_unified_mixed_dtype_router_vs_tier(self):
        torch.manual_seed(2)
        partition = _toy_partition()
        heads = PerTierHeads(d=16, partition=partition, mos_k=4)
        h = torch.randn(2, 5, 16)
        out = heads(h)
        out["router"] = out["router"].to(torch.bfloat16)
        for tier in _LOSS_TIER_ORDER:
            out[tier] = out[tier].to(torch.float32)
        unified = per_tier_unified_log_p(out, partition, n_vocab=100)
        self.assertEqual(tuple(unified.shape), (2, 5, 100))
        self.assertTrue(torch.isfinite(unified).all().item())


class TestPerTierModelSmoke(unittest.TestCase):
    def _model(self):
        import argparse
        import pandas as pd
        from preframr.train.model import Model
        from preframr_tokens.stfconstants import FRAME_REG, MODEL_PDTYPE, SET_OP

        args = argparse.Namespace(
            embed=32,
            heads=4,
            kv_heads=2,
            layers=2,
            intermediate=64,
            max_seq_len=16,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=1e4,
            rope_scale=1.0,
            tie_word_embeddings=True,
            precision="high",
            max_autotune=False,
            accumulate_grad_batches=1,
            learning_rate=1e-4,
            weight_decay=0.01,
            label_smoothing=0.0,
            model="llama3_2",
            per_tier_heads=True,
            per_tier_content_mos_k=2,
            per_tier_mos_entropy_lambda=0.01,
        )
        tokens = pd.DataFrame(
            [
                {"op": SET_OP, "reg": -1, "subreg": -1, "val": 0, "n": 0},
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "n": 1},
                {"op": SET_OP, "reg": 0, "subreg": -1, "val": 5, "n": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        return Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)

    def test_init_builds_per_tier_heads(self):
        model = self._model()
        self.assertTrue(model.per_tier_heads_on)
        self.assertTrue(hasattr(model, "per_tier_heads"))
        self.assertEqual(model.per_tier_heads.mos_k, 2)
        self.assertTrue(hasattr(model, "log_sigma_router"))
        self.assertTrue(hasattr(model, "log_sigma_per_tier"))

    def test_training_step_returns_finite_scalar(self):
        model = self._model()
        model.train()
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        log_calls = []
        model.log = lambda *a, **kw: log_calls.append((a, kw))
        loss = model.training_step((x, y), 0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        names = [a[0] for (a, _) in log_calls]
        self.assertIn("train_router_loss", names)
        self.assertIn("train_content_loss", names)

    def test_eval_forward_returns_unified_logits(self):
        model = self._model()
        model.eval()
        x = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        with torch.no_grad():
            out = model.model(x)
        self.assertEqual(tuple(out.shape), (1, 4, 3))
        rowsums = out.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(rowsums, torch.ones_like(rowsums), atol=1e-4))

    def test_gradient_flows_to_body(self):
        model = self._model()
        model.train()
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        model.log = lambda *a, **kw: None
        loss = model.training_step((x, y), 0)
        loss.backward()
        body_grad_sum = 0.0
        for p in model.model.parameters():
            if p.grad is not None:
                body_grad_sum += p.grad.abs().sum().item()
        self.assertGreater(body_grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()
