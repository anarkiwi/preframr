"""--onset-loss-weight: the per-vocab weight builder's no-op guards (default W=1.0,
empty tokens). The ``is_freq_onset_atom`` predicate lives in preframr_tokens and is
tested there (tests/test_is_freq_onset_atom.py)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.model.tier_map import _build_vocab_onset_weight


def test_onset_weight_default_is_noop():
    w = _build_vocab_onset_weight(
        SimpleNamespace(onset_loss_weight=1.0), 5, ["a", "b"], None
    )
    assert torch.equal(w, torch.ones(5))


def test_onset_weight_empty_tokens_is_ones():
    w = _build_vocab_onset_weight(SimpleNamespace(onset_loss_weight=8.0), 5, None, None)
    assert torch.equal(w, torch.ones(5))
