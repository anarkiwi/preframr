"""Smoke test for ``Model.__init__`` + a single training_step on a
tiny synthetic vocab. Covers the LightningModule glue + forward path
without standing up a Trainer."""

import argparse
import unittest

import pandas as pd
import torch

from preframr.train.model import Model, build_tier_map
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


class TestModelInit(unittest.TestCase):
    def test_init_builds_underlying_model(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        self.assertEqual(model.n_vocab, 3)
        self.assertEqual(model.metadata, None)
        self.assertEqual(model.vocab_frame_weight.shape, torch.Size([3]))

    def test_forward_on_random_input(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        model.eval()
        x = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        with torch.no_grad():
            out = model.model(x)
        self.assertEqual(out.shape, torch.Size([1, 4, 3]))

    def test_training_step_returns_scalar_loss(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        loss = model.training_step((x, y), 0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_build_tier_map_returns_named_tiers(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        tier_map = build_tier_map(args, n_vocab=3, tokens=tokens, tkmodel=None)
        self.assertEqual(set(tier_map.keys()), {0, 1, 2})
        for tier in tier_map.values():
            self.assertIn(tier, {"structural", "mid", "content", "zero"})

    def test_build_tier_map_empty_tokens_default_content(self):
        args = _tiny_args()
        tier_map = build_tier_map(args, n_vocab=5, tokens=None, tkmodel=None)
        self.assertEqual(tier_map, {i: "content" for i in range(5)})

    def test_validation_step_logs_acc_and_loss(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        log_calls = []

        def _log(*args_, **kw_):
            log_calls.append((args_, kw_))

        model.log = _log
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        out = model.validation_step((x, y), 0)
        self.assertIn("loss", out)
        self.assertIn("preds", out)
        self.assertIn("gt", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["preds"].shape, y.shape)
        names = [a[0] for (a, _) in log_calls]
        self.assertIn("val_loss", names)
        self.assertIn("val_acc", names)

    def test_configure_optimizers(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        opt = model.configure_optimizers()
        self.assertIsNotNone(opt)


if __name__ == "__main__":
    unittest.main()
