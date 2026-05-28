"""--onset-loss-weight: the FREQ V0-onset classifier and the per-vocab weight builder's
no-op guards (default W=1.0, empty tokens)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.model.tier_map import _build_vocab_onset_weight, _is_freq_onset_atom
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_OP,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
)


def test_is_freq_onset_atom():
    assert _is_freq_onset_atom(FREQ_TRAJ_OP, 0, FT_SUBREG_V0_HI)
    assert _is_freq_onset_atom(FREQ_TRAJ_OP, 7, FT_SUBREG_V0_LO)
    assert _is_freq_onset_atom(FREQ_TRAJ_OP, 14, FT_SUBREG_V0_HI)
    assert not _is_freq_onset_atom(0, 0, FT_SUBREG_V0_HI)
    assert not _is_freq_onset_atom(FREQ_TRAJ_OP, 2, FT_SUBREG_V0_HI)
    assert not _is_freq_onset_atom(FREQ_TRAJ_OP, 0, 6)


def test_onset_weight_default_is_noop():
    w = _build_vocab_onset_weight(
        SimpleNamespace(onset_loss_weight=1.0), 5, ["a", "b"], None
    )
    assert torch.equal(w, torch.ones(5))


def test_onset_weight_empty_tokens_is_ones():
    w = _build_vocab_onset_weight(SimpleNamespace(onset_loss_weight=8.0), 5, None, None)
    assert torch.equal(w, torch.ones(5))
