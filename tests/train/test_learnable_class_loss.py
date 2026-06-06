"""Learnable per-tier uncertainty loss + LOSS_TIER registry tests."""

import argparse
import unittest

import pandas as pd
import torch

# pylint: disable=unused-import,wrong-import-position
from preframr_tokens.macros import transforms_audio_bit_exact, transforms_bit_exact
from preframr_tokens.macros.transform import collect_op_loss_tiers
from preframr.train.model import (
    Model,
    _LOSS_TIER_ORDER,
    _LOSS_TIER_TO_ID,
    _N_LOSS_TIERS,
)
from preframr_tokens.stfconstants import FRAME_REG, MODEL_PDTYPE, SET_OP


def _tiny_args(**overrides):
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
        token_class_loss=False,
        token_class_weight_structural=0.5,
        token_class_weight_mid=1.0,
        token_class_weight_content=2.0,
        token_class_weight_zero=4.0,
        learnable_class_loss=False,
        macro_flags="",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _tiny_tokens():
    return pd.DataFrame(
        [
            {"op": SET_OP, "reg": -1, "subreg": -1, "val": 0, "n": 0},
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "n": 1},
            {"op": SET_OP, "reg": 0, "subreg": -1, "val": 5, "n": 2},
        ],
        dtype=MODEL_PDTYPE,
    )


class TestBuildOpMap(unittest.TestCase):
    """build_op_map labels each vocab id by its op-class (the pattern-compressing token kind)."""

    def test_op_map_labels_by_op(self):
        from preframr.train.model import build_op_map
        from preframr_tokens.stfconstants import PATTERN_REPLAY_OP, DIFF_OP

        tokens = pd.DataFrame(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0, "n": 0},
                {"op": int(DIFF_OP), "reg": 0, "subreg": -1, "val": 5, "n": 1},
                {
                    "op": int(PATTERN_REPLAY_OP),
                    "reg": -1,
                    "subreg": 0,
                    "val": 2,
                    "n": 2,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        op_map = build_op_map(_tiny_args(), len(tokens), tokens, tkmodel=None)
        self.assertEqual(op_map[0], "SET")
        self.assertEqual(op_map[1], "DIFF")
        self.assertEqual(op_map[2], "PATTERN_REPLAY")

    def test_empty_tokens_defaults_set(self):
        from preframr.train.model import build_op_map

        op_map = build_op_map(_tiny_args(), 4, None, tkmodel=None)
        self.assertEqual(set(op_map.values()), {"SET"})


class TestLossTierRegistry(unittest.TestCase):
    def test_loss_tier_order_canonical(self):
        self.assertEqual(_LOSS_TIER_ORDER, ("structural", "mid", "content", "zero"))
        self.assertEqual(_N_LOSS_TIERS, 4)
        self.assertEqual(_LOSS_TIER_TO_ID["structural"], 0)
        self.assertEqual(_LOSS_TIER_TO_ID["zero"], 3)

    def test_op_tier_map_has_expected_classes(self):
        op_tier = collect_op_loss_tiers()
        from preframr_tokens.stfconstants import (
            HARD_RESTART_OP,
            PATTERN_REPLAY_OP,
            SET_OP,
        )

        self.assertEqual(op_tier[int(HARD_RESTART_OP)], "structural")
        self.assertEqual(op_tier[int(PATTERN_REPLAY_OP)], "structural")
        self.assertEqual(op_tier[int(SET_OP)], "content")

    def test_all_registered_transforms_declare_valid_loss_tier(self):
        for op, tier in collect_op_loss_tiers().items():
            self.assertIn(tier, _LOSS_TIER_ORDER, msg=f"op={op} tier={tier}")


class TestLearnableLossLayer(unittest.TestCase):
    def test_default_off_no_log_sigma(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        self.assertFalse(model.learnable_class_loss)
        self.assertFalse(hasattr(model, "log_sigma_per_tier"))

    def test_enabled_registers_log_sigma_parameter(self):
        args = _tiny_args(learnable_class_loss=True)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        self.assertTrue(model.learnable_class_loss)
        self.assertTrue(isinstance(model.log_sigma_per_tier, torch.nn.Parameter))
        self.assertEqual(model.log_sigma_per_tier.shape, torch.Size([_N_LOSS_TIERS]))
        self.assertTrue(model.log_sigma_per_tier.requires_grad)

    def test_training_step_returns_finite_scalar(self):
        args = _tiny_args(learnable_class_loss=True)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        model.log = lambda *a, **k: None
        loss = model.training_step((x, y), 0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_log_sigma_gradient_flows(self):
        args = _tiny_args(learnable_class_loss=True)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        model.log = lambda *a, **k: None
        loss = model.training_step((x, y), 0)
        loss.backward()
        self.assertIsNotNone(model.log_sigma_per_tier.grad)
        self.assertTrue(torch.isfinite(model.log_sigma_per_tier.grad).all())

    def test_log_sigma_is_in_optimizer_param_groups(self):
        args = _tiny_args(learnable_class_loss=True)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        opt = model.configure_optimizers()
        params_in_opt = []
        for pg in opt.param_groups:
            params_in_opt.extend(id(p) for p in pg["params"])
        self.assertIn(
            id(model.log_sigma_per_tier),
            params_in_opt,
            msg=(
                "log_sigma_per_tier is registered as nn.Parameter but NOT in "
                "the optimizer's param groups. This means sigmas will never "
                "update during training -- the Kendall-Gal loss degenerates "
                "to fixed uniform weighting. Build the optimizer AFTER all "
                "Parameters are registered."
            ),
        )


class TestVocabTierIdBuffer(unittest.TestCase):
    def test_always_registered_even_when_disabled(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        self.assertTrue(hasattr(model, "vocab_tier_id"))
        self.assertEqual(model.vocab_tier_id.shape, torch.Size([3]))
        self.assertEqual(model.vocab_tier_id.dtype, torch.long)


class TestMaskStructuralTierLoss(unittest.TestCase):
    """--mask-structural-tier-loss zeroes the loss contribution from structural-tier targets only."""

    def test_default_off(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        self.assertFalse(model.mask_structural_tier_loss)

    def test_all_structural_targets_yield_zero_loss(self):
        from preframr.train.model.tier_map import _STRUCTURAL_TIER_ID

        args = _tiny_args(mask_structural_tier_loss=True)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        model.vocab_tier_id.fill_(_STRUCTURAL_TIER_ID)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        model.log = lambda *a, **k: None
        loss = model.training_step((x, y), 0)
        self.assertEqual(loss.item(), 0.0)

    def test_partial_masking_changes_loss_value(self):
        from preframr.train.model.tier_map import (
            _CONTENT_TIER_ID,
            _STRUCTURAL_TIER_ID,
        )

        args_off = _tiny_args()
        args_on = _tiny_args(mask_structural_tier_loss=True)
        tokens = _tiny_tokens()
        torch.manual_seed(0)
        model_off = Model(
            args_off, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None
        )
        torch.manual_seed(0)
        model_on = Model(args_on, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        for m in (model_off, model_on):
            m.vocab_tier_id[0] = _STRUCTURAL_TIER_ID
            m.vocab_tier_id[1] = _STRUCTURAL_TIER_ID
            m.vocab_tier_id[2] = _CONTENT_TIER_ID
            m.log = lambda *a, **k: None
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        loss_off = model_off.training_step((x, y), 0).item()
        loss_on = model_on.training_step((x, y), 0).item()
        self.assertNotAlmostEqual(loss_off, loss_on, places=5)
        self.assertTrue(torch.isfinite(torch.tensor(loss_on)))


if __name__ == "__main__":
    unittest.main()
