"""Tests for the auto-abort Lightning callback."""

from __future__ import annotations

import types
import unittest

import pytest

pytest.importorskip("pytorch_lightning")

from preframr_tokens.audit_primitives import distinct_n as _distinct_n
from preframr.train.generalization_gate import (
    GateThresholds,
    GeneralizationGate,
)


class _FakeTrainer:
    def __init__(self, epoch: int = 0):
        self.current_epoch = epoch
        self.should_stop = False


class _FakeModule:
    def __init__(self):
        self.logged: dict = {}

    def log(self, name, value, **_kwargs):
        self.logged[name] = value


def _outputs(preds, gt):
    p = types.SimpleNamespace(
        flatten=lambda: types.SimpleNamespace(tolist=lambda: preds)
    )
    g = types.SimpleNamespace(flatten=lambda: types.SimpleNamespace(tolist=lambda: gt))
    return {"preds": p, "gt": g}


class TestPrimitives(unittest.TestCase):
    def test_distinct_n_short(self):
        self.assertEqual(_distinct_n([1, 2], n=4), 0)

    def test_distinct_n_unique(self):
        self.assertEqual(_distinct_n([1, 2, 3, 4, 5, 6, 7, 8], n=4), 5)

    def test_distinct_n_repeating(self):
        self.assertEqual(_distinct_n([1, 2, 3, 4] * 8, n=4), 4)


class TestContentStructuralGate(unittest.TestCase):
    def setUp(self):
        self.tier_map = {1: "structural", 2: "structural", 10: "content", 11: "content"}
        self.gate = GeneralizationGate(
            tier_map=self.tier_map,
            thresholds=GateThresholds(content_over_structural_min_epoch=5),
        )

    def _run_one_val(self, trainer, preds, gt):
        mod = _FakeModule()
        self.gate.on_validation_epoch_start(trainer, mod)
        self.gate.on_validation_batch_end(trainer, mod, _outputs(preds, gt), None, 0)
        self.gate.on_validation_epoch_end(trainer, mod)
        return mod

    def test_apush_signature_aborts_above_min_epoch(self):
        trainer = _FakeTrainer(epoch=5)
        gt = [1] * 10 + [10] * 10
        preds = [1] * 10 + [99] * 10
        self._run_one_val(trainer, preds, gt)
        self.assertTrue(trainer.should_stop)
        self.assertIn("content/structural", self.gate.aborted_reason)

    def test_apush_signature_below_min_epoch_no_abort(self):
        trainer = _FakeTrainer(epoch=1)
        gt = [1] * 10 + [10] * 10
        preds = [1] * 10 + [99] * 10
        self._run_one_val(trainer, preds, gt)
        self.assertFalse(trainer.should_stop)

    def test_healthy_ratio_no_abort(self):
        trainer = _FakeTrainer(epoch=10)
        gt = [1] * 10 + [10] * 10
        preds = list(gt)
        mod = self._run_one_val(trainer, preds, gt)
        self.assertFalse(trainer.should_stop)
        self.assertEqual(mod.logged["gate/content_over_structural"], 1.0)


class TestPerOpAccuracy(unittest.TestCase):
    """The per-op-class accuracy instrument: which pattern-compressing token learns."""

    def _gate(self, op_map):
        return GeneralizationGate(
            tier_map={1: "structural", 2: "structural", 10: "content", 11: "content"},
            op_map=op_map,
            thresholds=GateThresholds(content_over_structural_min_epoch=999),
        )

    def test_per_op_acc_logged(self):
        gate = self._gate({1: "BACK_REF", 2: "BACK_REF", 10: "DIFF", 11: "STAMP_REF"})
        trainer = _FakeTrainer(epoch=1)
        mod = _FakeModule()
        gate.on_validation_epoch_start(trainer, mod)
        gt = [1, 1, 1, 1, 10, 10, 11, 11]
        preds = [1, 1, 9, 9, 10, 99, 11, 11]
        gate.on_validation_batch_end(trainer, mod, _outputs(preds, gt), None, 0)
        gate.on_validation_epoch_end(trainer, mod)
        self.assertFalse(trainer.should_stop)
        self.assertAlmostEqual(mod.logged["gate/op_acc/BACK_REF"], 0.5)
        self.assertAlmostEqual(mod.logged["gate/op_acc/DIFF"], 0.5)
        self.assertAlmostEqual(mod.logged["gate/op_acc/STAMP_REF"], 1.0)

    def test_no_op_map_no_per_op_logging(self):
        gate = self._gate(None)
        trainer = _FakeTrainer(epoch=1)
        mod = _FakeModule()
        gate.on_validation_epoch_start(trainer, mod)
        gate.on_validation_batch_end(trainer, mod, _outputs([1, 10], [1, 10]), None, 0)
        gate.on_validation_epoch_end(trainer, mod)
        self.assertFalse(any(k.startswith("gate/op_acc/") for k in mod.logged))


class TestLoopCollapseGate(unittest.TestCase):
    def test_collapse_above_min_epoch_aborts(self):
        prompts = [[0, 1, 2]] * 3

        def gen_fn(_pl, _prompt, _n):
            return [7] * 200

        gate = GeneralizationGate(
            tier_map={},
            audit_prompts=prompts,
            generate_fn=gen_fn,
            generate_every_k_epochs=1,
            thresholds=GateThresholds(loop_collapse_min_epoch=2),
        )
        trainer = _FakeTrainer(epoch=2)
        mod = _FakeModule()
        gate.on_validation_epoch_start(trainer, mod)
        gate.on_validation_epoch_end(trainer, mod)
        self.assertTrue(trainer.should_stop)
        self.assertIn("loop_collapse_rate", gate.aborted_reason)
        self.assertEqual(mod.logged["gate/loop_collapse_rate"], 1.0)


class TestDistinctNGate(unittest.TestCase):
    def test_low_distinct_aborts(self):
        prompts = [[0]] * 2

        def gen_fn(_pl, _prompt, _n):
            return [1, 2] * 100

        gate = GeneralizationGate(
            tier_map={},
            audit_prompts=prompts,
            generate_fn=gen_fn,
            generate_every_k_epochs=1,
            thresholds=GateThresholds(
                loop_collapse_min_epoch=999,
                distinct_n_min_epoch=2,
                distinct_n_min=30,
            ),
        )
        trainer = _FakeTrainer(epoch=3)
        gate.on_validation_epoch_end(trainer, _FakeModule())
        self.assertTrue(trainer.should_stop)
        self.assertIn("distinct_n4", gate.aborted_reason)


if __name__ == "__main__":
    unittest.main()
