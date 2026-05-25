"""Tests for ``preframr.model`` loss-function machinery."""

import unittest

import pandas as pd
import torch

from preframr.train.model import _build_vocab_frame_weight, chunked_cross_entropy
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    DELAY_REG,
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
        from preframr_tokens.stfconstants import (
            BACK_REF_SUBREG_DIST_HI,
            BACK_REF_SUBREG_DIST_LO,
            BACK_REF_SUBREG_LEN,
        )

        tokens = _tokens_df(
            [
                {"reg": 4, "subreg": -1, "val": 8, "op": SET_OP},
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_DIST_HI,
                    "val": 0,
                    "op": BACK_REF_OP,
                },
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_DIST_LO,
                    "val": 10,
                    "op": BACK_REF_OP,
                },
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_LEN,
                    "val": 8,
                    "op": BACK_REF_OP,
                },
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 4, tokens, None)
        self.assertEqual(float(w[0]), 1.0)
        self.assertEqual(float(w[1]), 1.0)
        self.assertEqual(float(w[2]), 1.0)
        self.assertEqual(float(w[3]), 8.0)

    def test_do_loop_begin_weight_equals_iteration_count(self):
        tokens = _tokens_df(
            [
                {"reg": LOOP_OP_REG, "subreg": 0, "val": 4, "op": DO_LOOP_OP},
                {"reg": LOOP_OP_REG, "subreg": 1, "val": 0, "op": DO_LOOP_OP},
            ]
        )
        w = _build_vocab_frame_weight(FakeArgs(), 2, tokens, None)
        self.assertEqual(float(w[0]), 4.0)
        self.assertEqual(float(w[1]), 1.0)

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
        weights = torch.tensor([1.0, 1.0, 10.0])
        loss = (per_tok * weights).sum() / weights.sum().clamp(min=1.0)
        self.assertGreater(float(loss), 4.0)
        self.assertGreater(float(loss), float(per_tok.mean()))


class TestChunkedCrossEntropy(unittest.TestCase):
    """``chunked_cross_entropy`` is a memory-efficient drop-in for
    ``F.cross_entropy(logits.swapaxes(1, 2), target, reduction='none', ...)``.
    Values and gradients must match the eager call to within fp tolerance.
    """

    def _reference(self, logits, target, label_smoothing):
        return torch.nn.functional.cross_entropy(
            logits.swapaxes(1, 2),
            target,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def test_values_match_reference_no_chunk(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 7, 17, dtype=torch.float32)
        target = torch.randint(0, 17, (2, 7))
        got = chunked_cross_entropy(logits, target, chunk_bytes=1 << 30)
        ref = self._reference(logits, target, 0.0)
        torch.testing.assert_close(got, ref)

    def test_values_match_reference_chunked(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 13, 32, dtype=torch.float32)
        target = torch.randint(0, 32, (4, 13))
        got = chunked_cross_entropy(logits, target, chunk_bytes=256)
        ref = self._reference(logits, target, 0.0)
        torch.testing.assert_close(got, ref)

    def test_values_match_reference_with_label_smoothing(self):
        torch.manual_seed(1)
        logits = torch.randn(3, 11, 23, dtype=torch.float32)
        target = torch.randint(0, 23, (3, 11))
        got = chunked_cross_entropy(
            logits, target, label_smoothing=0.1, chunk_bytes=256
        )
        ref = self._reference(logits, target, 0.1)
        torch.testing.assert_close(got, ref)

    def test_gradients_match_reference_chunked(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 9, 19, dtype=torch.float32, requires_grad=True)
        target = torch.randint(0, 19, (2, 9))
        got = chunked_cross_entropy(logits, target, chunk_bytes=128)
        got.sum().backward()
        got_grad = logits.grad.detach().clone()
        logits.grad = None
        ref = self._reference(logits, target, 0.0)
        ref.sum().backward()
        torch.testing.assert_close(got_grad, logits.grad)


if __name__ == "__main__":
    unittest.main()
