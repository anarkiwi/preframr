"""Tests for train.py audit-prompt builder."""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.trainer import _build_audit_prompts


class _FakeLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def _batch(rows):
    x = torch.tensor(rows, dtype=torch.long)
    y = torch.zeros_like(x)
    return (x, y)


class TestBuildAuditPrompts(unittest.TestCase):
    def test_none_loader_returns_empty(self):
        self.assertEqual(_build_audit_prompts(None, 4, 8), [])

    def test_zero_prompts_returns_empty(self):
        loader = _FakeLoader([_batch([[1, 2, 3, 4]])])
        self.assertEqual(_build_audit_prompts(loader, 0, 8), [])

    def test_truncates_to_prompt_len(self):
        loader = _FakeLoader([_batch([[1, 2, 3, 4, 5, 6]])])
        out = _build_audit_prompts(loader, n_prompts=1, prompt_len=3)
        self.assertEqual(out, [[1, 2, 3]])

    def test_collects_across_batches(self):
        loader = _FakeLoader(
            [
                _batch([[1, 2, 3]]),
                _batch([[4, 5, 6], [7, 8, 9]]),
            ]
        )
        out = _build_audit_prompts(loader, n_prompts=3, prompt_len=2)
        self.assertEqual(out, [[1, 2], [4, 5], [7, 8]])

    def test_stops_at_n_prompts(self):
        loader = _FakeLoader([_batch([[i, i + 1] for i in range(20)])])
        out = _build_audit_prompts(loader, n_prompts=4, prompt_len=1)
        self.assertEqual(len(out), 4)

    def test_handles_list_of_loaders(self):
        loaders = [
            _FakeLoader([_batch([[1, 2]])]),
            _FakeLoader([_batch([[3, 4]])]),
        ]
        out = _build_audit_prompts(loaders, n_prompts=2, prompt_len=2)
        self.assertEqual(out, [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
