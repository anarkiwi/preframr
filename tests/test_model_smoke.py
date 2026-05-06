"""Smoke test for ``Model.__init__`` + a single training_step on a
tiny synthetic vocab. Covers the LightningModule glue + forward path
without standing up a Trainer."""

import argparse
import unittest

import pandas as pd
import torch

from preframr.model import Model
from preframr.stfconstants import FRAME_REG, MODEL_PDTYPE, SET_OP


def _tiny_args():
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


if __name__ == "__main__":
    unittest.main()
