"""Tests for ``preframr.model`` loss-function machinery.

Exercises:
  * ``_build_vocab_frame_weight`` -- frame weighting per vocab id.
  * Focal-loss scaling math (verified directly against the formula).
  * Frame-weighted CE collapses to plain CE when all weights are 1.
"""

import unittest

import pandas as pd
import torch

from preframr.macros import _pack_back_ref
from preframr.model import _build_vocab_frame_weight
from preframr.stfconstants import (
    BACK_REF_OP,
    DELAY_REG,
    DIFF_OP,
    DO_LOOP_OP,
    FRAME_REG,
    LOOP_OP_REG,
    SET_OP,
)


class FakeArgs:
    def __init__(self, **kw):
        self.tkvocab = 0
        for k, v in kw.items():
            setattr(self, k, v)


def _tokens_df(rows):
    df = pd.DataFrame(rows)
    if "n" not in df.columns:
        df["n"] = df.index
    return df


class TestVocabFrameWeight(unittest.TestCase):
    def test_within_frame_writes_default_to_one(self):
        # Three plain SET base tokens. No tkmodel -> identity vocab.
        tokens = _tokens_df(
            [
                {"reg": 4, "subreg": -1, "val": 8, "op": SET_OP},
                {"reg": 5, "subreg": -1, "val": 10, "op": SET_OP},
                {"reg": 6, "subreg": -1, "val": 200, "op": SET_OP},
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 3, tokens, None)
        self.assertTrue(torch.equal(w, torch.ones(3)))

    def test_back_ref_weight_equals_length(self):
        tokens = _tokens_df(
            [
                {"reg": 4, "subreg": -1, "val": 8, "op": SET_OP},
                {
                    "reg": LOOP_OP_REG,
                    "subreg": -1,
                    "val": _pack_back_ref(distance=10, length=8),
                    "op": BACK_REF_OP,
                },
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 2, tokens, None)
        self.assertEqual(float(w[0]), 1.0)
        self.assertEqual(float(w[1]), 8.0)

    def test_do_loop_begin_weight_equals_iteration_count(self):
        tokens = _tokens_df(
            [
                {"reg": LOOP_OP_REG, "subreg": 0, "val": 4, "op": DO_LOOP_OP},
                {"reg": LOOP_OP_REG, "subreg": 1, "val": 0, "op": DO_LOOP_OP},
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 2, tokens, None)
        self.assertEqual(float(w[0]), 4.0)  # BEGIN N=4
        self.assertEqual(float(w[1]), 1.0)  # END floored to 1

    def test_delay_reg_weight_equals_val(self):
        tokens = _tokens_df(
            [
                {"reg": DELAY_REG, "subreg": -1, "val": 5, "op": SET_OP},
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 1, tokens, None)
        self.assertEqual(float(w[0]), 5.0)

    def test_frame_reg_weight_one(self):
        tokens = _tokens_df(
            [
                {"reg": FRAME_REG, "subreg": -1, "val": 0, "op": SET_OP},
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 1, tokens, None)
        self.assertEqual(float(w[0]), 1.0)


class TestFocalLossMath(unittest.TestCase):
    """Verify the focal scaling implemented in training_step matches the
    formula. We run the math directly against the tensor formulation; the
    full Model.training_step requires building a transformer and is
    overkill for this check."""

    def _focal(self, ce, alpha, gamma):
        # mirrors the body of training_step (no smoothing)
        p = (-ce).exp().clamp(max=1.0)
        return alpha * (1.0 - p).pow(gamma) * ce

    def test_gamma_zero_is_alpha_times_ce(self):
        ce = torch.tensor([0.1, 1.0, 5.0])
        out = self._focal(ce, alpha=0.25, gamma=0.0)
        self.assertTrue(torch.allclose(out, 0.25 * ce))

    def test_higher_gamma_downweights_confident_predictions(self):
        # CE=0.05 means the model was very confident and right.
        # CE=2.0 means the model was wrong-ish.
        # focal should reduce the small-CE term more aggressively than the
        # large-CE term.
        ce = torch.tensor([0.05, 2.0])
        out = self._focal(ce, alpha=1.0, gamma=2.0)
        ratio = out / ce
        self.assertLess(float(ratio[0]), float(ratio[1]))


class TestFrameWeightedReduction(unittest.TestCase):
    """The weighted-mean reduction in training_step degenerates to plain
    mean when every target gets weight 1, and biases toward heavily-weighted
    targets otherwise."""

    def test_all_ones_equals_mean(self):
        per_tok = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = torch.ones_like(per_tok)
        loss = (per_tok * weights).sum() / weights.sum().clamp(min=1.0)
        self.assertAlmostEqual(float(loss), float(per_tok.mean()))

    def test_heavy_weight_dominates(self):
        per_tok = torch.tensor([0.1, 0.1, 5.0])
        # First two tokens are cheap (CE=0.1 each), third is a misprediction.
        # If the third is a multi-frame macro token (weight=10), the loss
        # should be much closer to its CE than to the mean.
        weights = torch.tensor([1.0, 1.0, 10.0])
        loss = (per_tok * weights).sum() / weights.sum().clamp(min=1.0)
        self.assertGreater(float(loss), 4.0)  # close to the 5.0 outlier
        # Same per_tok with uniform weights would average to ~1.73.
        self.assertGreater(float(loss), float(per_tok.mean()))


if __name__ == "__main__":
    unittest.main()
