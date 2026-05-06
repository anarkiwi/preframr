"""Smoke test for ``Model.__init__`` + a single training_step on a
tiny synthetic vocab. Covers the LightningModule glue + forward path
without standing up a Trainer."""

import argparse
import unittest

import pandas as pd
import torch

from preframr.model import Model
from preframr.stfconstants import FRAME_REG, MODEL_PDTYPE, SET_OP


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
        l1_lambda=0.0,
        focal_alpha=1.0,
        focal_gamma=0.0,
        label_smoothing=0.0,
        model="llama3_2",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _tiny_tokens():
    # Three tokens: PAD-ish, FRAME_REG val=1, SET reg=0 val=5.
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
        # Vocab frame weight buffer present + correct length.
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
        # Cover the training_step body: per-token CE -> focal -> pad
        # masking -> frame-weight reduction.
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        loss = model.training_step((x, y), 0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_training_step_focal_loss_branch(self):
        # focal_gamma > 0 takes the focal-scaling branch.
        args = _tiny_args(focal_gamma=2.0, focal_alpha=0.5)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        loss = model.training_step((x, y), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_training_step_l1_branch(self):
        # l1_lambda > 0 adds the L1 norm to the loss.
        args = _tiny_args(l1_lambda=0.001)
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        loss = model.training_step((x, y), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_validation_step_logs_acc_and_loss(self):
        # validation_step does its own CE + pad_mask + accuracy. The
        # underlying ``self.log`` call is a no-op in this test (no
        # Trainer attached) but the math runs to completion.
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        # Stub out self.log so it doesn't try to talk to a Trainer.
        log_calls = []

        def _log(*args_, **kw_):
            log_calls.append((args_, kw_))

        model.log = _log
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        out = model.validation_step((x, y), 0)
        self.assertTrue(torch.isfinite(out))
        # val_loss + val_acc both logged.
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
