"""Unit tests for ``preframr.structural_loss``."""

import math
import unittest

import pandas as pd
import torch

from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    FRAME_REG,
    PAD_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_OVERLAY_SUBREG_TARGET_REG,
    PATTERN_OVERLAY_SUBREG_NEW_VAL,
    SET_OP,
)
from preframr.train.structural_loss import StructuralLoss


def _build_tokens():
    """Compact atomic vocab with the structural classes the loss cares
    about. Mirrors the fixture in tests/test_constrained_decode.py.
    """
    rows = [
        {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0},
        {"op": SET_OP, "reg": 0, "subreg": -1, "val": 7},
        {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11},
        {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_DIST_HI, "val": 0},
        {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_DIST_LO, "val": 1},
        {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_LEN, "val": 1},
        {
            "op": PATTERN_OVERLAY_OP,
            "reg": -125,
            "subreg": PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
            "val": 0,
        },
        {
            "op": PATTERN_OVERLAY_OP,
            "reg": -125,
            "subreg": PATTERN_OVERLAY_SUBREG_TARGET_REG,
            "val": 4,
        },
        {
            "op": PATTERN_OVERLAY_OP,
            "reg": -125,
            "subreg": PATTERN_OVERLAY_SUBREG_NEW_VAL,
            "val": 99,
        },
    ]
    return pd.DataFrame(rows)


class _Args:
    """Minimal stand-in; ``StructuralLoss`` only forwards ``args`` to
    ``RegTokenizer`` for the optional Unigram path. Atomic-mode tests
    don't need any fields populated."""


def _logits_one_hot(target_id, n_vocab, peak=20.0):
    """Logits putting essentially all mass on ``target_id``."""
    logits = torch.full((n_vocab,), -1.0)
    logits[target_id] = peak
    return logits


def _stack(rows):
    """Promote a list of (V,) row tensors to (1, T, V)."""
    return torch.stack(rows).unsqueeze(0)


class TestStructuralLossAtomic(unittest.TestCase):
    def setUp(self):
        self.tokens = _build_tokens()
        self.n_vocab = len(self.tokens)
        self.loss_fn = StructuralLoss(_Args(), self.n_vocab, self.tokens, tkmodel=None)
        self.PAD = 0
        self.SET_R0 = 1
        self.FRAME = 2
        self.BR_DIST_HI = 3
        self.BR_DIST_LO = 4
        self.BR_LEN = 5
        self.OV_FOFF = 6
        self.OV_TARG = 7
        self.OV_NEW = 8

    def test_aux_zero_when_all_mass_on_valid_target(self):
        """Single-step: logits concentrated on a top-level-legal target
        produce aux ~ 0 (within fp tolerance of the high-peak softmax)."""
        logits = _stack([_logits_one_hot(self.FRAME, self.n_vocab, peak=30.0)])
        targets = torch.tensor([[self.FRAME]], dtype=torch.long)
        pad_mask = torch.ones_like(targets, dtype=torch.float32)
        aux = self.loss_fn.compute(logits, targets, pad_mask).item()
        self.assertLess(aux, 1e-6)

    def test_aux_large_when_mass_on_orphan(self):
        """All mass on PATTERN_OVERLAY (orphan at top) -> aux ~ peak.
        With logits=[peak on orphan, -1 elsewhere], the valid-mass is
        the small tail and -log(valid_mass) is large."""
        logits = _stack([_logits_one_hot(self.OV_FOFF, self.n_vocab, peak=20.0)])
        targets = torch.tensor([[self.FRAME]], dtype=torch.long)
        pad_mask = torch.ones_like(targets, dtype=torch.float32)
        aux = self.loss_fn.compute(logits, targets, pad_mask).item()
        self.assertGreater(aux, 10.0)

    def test_aux_pair_intermediate_orphan(self):
        """BR-len at top level (no open BR pair) is also an orphan."""
        logits = _stack([_logits_one_hot(self.BR_LEN, self.n_vocab, peak=20.0)])
        targets = torch.tensor([[self.FRAME]], dtype=torch.long)
        pad_mask = torch.ones_like(targets, dtype=torch.float32)
        aux = self.loss_fn.compute(logits, targets, pad_mask).item()
        self.assertGreater(aux, 10.0)

    def test_aux_state_advances_through_pair(self):
        """Four-step sequence: FRAME bumps frame_count to 1, then
        BR-dist-hi (val=0), BR-dist-lo (val=1), BR-len."""
        seq = [self.FRAME, self.BR_DIST_HI, self.BR_DIST_LO, self.BR_LEN]
        logits = _stack([_logits_one_hot(t, self.n_vocab, peak=30.0) for t in seq])
        targets = torch.tensor([seq], dtype=torch.long)
        pad_mask = torch.ones_like(targets, dtype=torch.float32)
        aux = self.loss_fn.compute(logits, targets, pad_mask).item()
        self.assertLess(aux, 1e-6)

    def test_aux_pad_positions_excluded(self):
        """Pad positions must contribute neither numerator nor
        denominator: scaling pad-position aux to a huge value should
        not change the mean-over-non-pad result."""
        logits = torch.stack(
            [
                _logits_one_hot(self.OV_FOFF, self.n_vocab, peak=20.0),
                _logits_one_hot(self.FRAME, self.n_vocab, peak=30.0),
            ]
        ).unsqueeze(0)
        targets = torch.tensor([[self.PAD, self.FRAME]], dtype=torch.long)
        pad_mask = torch.tensor([[0.0, 1.0]])
        aux = self.loss_fn.compute(logits, targets, pad_mask).item()
        self.assertLess(aux, 1e-6)

    def test_aux_finite_and_differentiable(self):
        """Smoke check: the loss is differentiable wrt logits."""
        n = self.n_vocab
        logits = torch.randn(1, 4, n, requires_grad=True)
        targets = torch.tensor(
            [[self.SET_R0, self.FRAME, self.SET_R0, self.FRAME]], dtype=torch.long
        )
        pad_mask = torch.ones_like(targets, dtype=torch.float32)
        aux = self.loss_fn.compute(logits, targets, pad_mask)
        self.assertTrue(math.isfinite(aux.item()))
        aux.backward()
        self.assertEqual(logits.grad.shape, logits.shape)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
