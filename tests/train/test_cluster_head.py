"""Tests for ClusterContentHead (hierarchical cluster-conditional content head)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from preframr.train.model.heads_cluster import (
    ClusterContentHead,
    load_cluster_assignments,
)


def _toy_head(d=16, vocab=12, c=4, seed=0):
    torch.manual_seed(seed)
    cluster_id = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=torch.long)
    assert cluster_id.numel() == vocab
    head = ClusterContentHead(d=d, vocab_size=vocab, c=c, cluster_id_local=cluster_id)
    return head, cluster_id


class TestClusterContentHeadShape(unittest.TestCase):
    def test_forward_shape(self):
        head, _ = _toy_head()
        h = torch.randn(3, 7, 16)
        joint_log_p, cluster_log_p = head(h)
        self.assertEqual(tuple(joint_log_p.shape), (3, 7, 12))
        self.assertEqual(tuple(cluster_log_p.shape), (3, 7, 4))

    def test_joint_log_probs_normalised(self):
        head, _ = _toy_head()
        h = torch.randn(2, 4, 16)
        joint_log_p, _ = head(h)
        row_sums = joint_log_p.exp().sum(dim=-1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
            msg=f"joint not normalised; row_sums={row_sums}",
        )

    def test_cluster_log_probs_normalised(self):
        head, _ = _toy_head()
        h = torch.randn(2, 4, 16)
        _, cluster_log_p = head(h)
        row_sums = cluster_log_p.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_within_cluster_log_probs_normalised_per_cluster(self):
        """For each cluster c, sum of joint p over v in c should equal cluster p(c)."""
        head, cluster_id = _toy_head()
        h = torch.randn(2, 4, 16)
        joint_log_p, cluster_log_p = head(h)
        for c in range(head.c):
            mask = cluster_id == c
            if not mask.any():
                continue
            joint_p_in_c = joint_log_p[..., mask].exp().sum(dim=-1)
            cluster_p_c = cluster_log_p[..., c].exp()
            self.assertTrue(
                torch.allclose(joint_p_in_c, cluster_p_c, atol=1e-5),
                msg=f"cluster {c}: joint-sum {joint_p_in_c} != cluster_p {cluster_p_c}",
            )


class TestClusterContentHeadFactorisation(unittest.TestCase):
    """Cross-check: NLL on the joint equals cluster CE + within CE on the explicit decomposition."""

    def test_joint_nll_matches_explicit_decomposition(self):
        head, cluster_id = _toy_head()
        h = torch.randn(5, 16)
        joint_log_p, cluster_log_p = head(h)
        gt = torch.tensor([0, 3, 5, 9, 11], dtype=torch.long)
        joint_nll = -joint_log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
        gt_clusters = cluster_id[gt]
        cluster_nll = -cluster_log_p.gather(-1, gt_clusters.unsqueeze(-1)).squeeze(-1)
        within_log_p = joint_log_p - cluster_log_p.index_select(-1, cluster_id)
        within_nll = -within_log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
        decomposed = cluster_nll + within_nll
        self.assertTrue(
            torch.allclose(joint_nll, decomposed, atol=1e-5),
            msg=f"joint {joint_nll} != decomposed {decomposed}",
        )


class TestClusterContentHeadValidation(unittest.TestCase):
    def test_rejects_wrong_cluster_id_shape(self):
        with self.assertRaises(ValueError):
            ClusterContentHead(
                d=8,
                vocab_size=10,
                c=4,
                cluster_id_local=torch.zeros(9, dtype=torch.long),
            )

    def test_rejects_wrong_cluster_id_dtype(self):
        with self.assertRaises(TypeError):
            ClusterContentHead(
                d=8,
                vocab_size=10,
                c=4,
                cluster_id_local=torch.zeros(10, dtype=torch.int32),
            )

    def test_rejects_out_of_range_cluster_id(self):
        bad = torch.zeros(10, dtype=torch.long)
        bad[3] = 4
        with self.assertRaises(ValueError):
            ClusterContentHead(d=8, vocab_size=10, c=4, cluster_id_local=bad)


class TestLoadClusterAssignments(unittest.TestCase):
    def test_translates_full_to_local(self):
        in_tier = torch.tensor([10, 20, 30, 40], dtype=torch.long)
        full_assignments = {"10": 0, "20": 1, "30": 0, "40": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ca.json"
            path.write_text(json.dumps({"cluster_assignments": full_assignments}))
            cluster_id_local = load_cluster_assignments(path, in_tier, c=2)
        self.assertEqual(tuple(cluster_id_local.tolist()), (0, 1, 0, 1))
        self.assertEqual(cluster_id_local.dtype, torch.long)

    def test_raises_on_missing_vid_within_index_range(self):
        in_tier = torch.tensor([5, 15, 25], dtype=torch.long)
        full_assignments = {"10": 0, "20": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ca.json"
            path.write_text(json.dumps({"cluster_assignments": full_assignments}))
            with self.assertRaises(ValueError):
                load_cluster_assignments(path, in_tier, c=2)

    def test_above_index_range_defaults_to_cluster_zero(self):
        in_tier = torch.tensor([10, 20, 100, 200], dtype=torch.long)
        full_assignments = {"10": 1, "20": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ca.json"
            path.write_text(json.dumps({"cluster_assignments": full_assignments}))
            cluster_id_local = load_cluster_assignments(path, in_tier, c=2)
        self.assertEqual(tuple(cluster_id_local.tolist()), (1, 1, 0, 0))

    def test_custom_default_cluster(self):
        in_tier = torch.tensor([10, 100], dtype=torch.long)
        full_assignments = {"10": 0}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ca.json"
            path.write_text(json.dumps({"cluster_assignments": full_assignments}))
            cluster_id_local = load_cluster_assignments(
                path, in_tier, c=4, default_cluster=3
            )
        self.assertEqual(tuple(cluster_id_local.tolist()), (0, 3))

    def test_raises_on_out_of_range_cluster(self):
        in_tier = torch.tensor([10, 20], dtype=torch.long)
        full_assignments = {"10": 0, "20": 5}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ca.json"
            path.write_text(json.dumps({"cluster_assignments": full_assignments}))
            with self.assertRaises(ValueError):
                load_cluster_assignments(path, in_tier, c=2)


if __name__ == "__main__":
    unittest.main()
