"""Cluster-conditional content head. Hierarchical: predict acoustic-equivalence cluster first, then token within cluster. Joint log-prob factorises as log p(v|h) = log p(c(v)|h) + log p(v|c(v),h). Plug-in shape matches MoSHead — forward returns (joint_log_p, cluster_log_p) so PerTierHeads + per_tier_unified_log_p + the lightning loss path work unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class ClusterContentHead(torch.nn.Module):
    """Hierarchical content head: cluster predictor + within-cluster token predictor with per-cluster mask. Vectorised per-cluster logsumexp via scatter_reduce — no Python loop."""

    def __init__(self, d: int, vocab_size: int, c: int, cluster_id_local: torch.Tensor):
        super().__init__()
        if cluster_id_local.shape != (vocab_size,):
            raise ValueError(
                f"cluster_id_local shape {tuple(cluster_id_local.shape)} != "
                f"({vocab_size},)"
            )
        if cluster_id_local.dtype != torch.long:
            raise TypeError(
                f"cluster_id_local dtype {cluster_id_local.dtype} != torch.long"
            )
        if int(cluster_id_local.max()) >= c or int(cluster_id_local.min()) < 0:
            raise ValueError(
                f"cluster_id_local out of [0,{c-1}]: min={int(cluster_id_local.min())} max={int(cluster_id_local.max())}"
            )
        self.c = c
        self.vocab_size = vocab_size
        self.cluster_proj = torch.nn.Linear(d, c)
        self.token_proj = torch.nn.Linear(d, vocab_size)
        self.register_buffer("cluster_id_local", cluster_id_local, persistent=True)

    def forward(self, h):
        cluster_logits = self.cluster_proj(h)
        token_logits = self.token_proj(h)
        with torch.amp.autocast(h.device.type, enabled=False):
            if h.is_cuda and token_logits.dtype == torch.float32:
                cluster_logits = cluster_logits.to(torch.bfloat16)
                token_logits = token_logits.to(torch.bfloat16)
            cluster_log_p = torch.log_softmax(cluster_logits, dim=-1)
            within_log_p = self._within_log_p(token_logits)
            token_cluster_lp = cluster_log_p.index_select(-1, self.cluster_id_local)
            joint_log_p = token_cluster_lp + within_log_p
            if joint_log_p.dtype != cluster_logits.dtype:
                joint_log_p = joint_log_p.to(cluster_logits.dtype)
                cluster_log_p = cluster_log_p.to(cluster_logits.dtype)
            return joint_log_p, cluster_log_p

    def _within_log_p(self, token_logits):
        """Per-cluster log_softmax via vectorised scatter_reduce. Returns (..., V) where within_log_p[..., v] = log p(v | h, c(v)) restricted to v's cluster."""
        prefix_shape = token_logits.shape[:-1]
        index = self.cluster_id_local.expand(*prefix_shape, -1)
        cluster_max = token_logits.new_full((*prefix_shape, self.c), float("-inf"))
        cluster_max = cluster_max.scatter_reduce(
            -1, index, token_logits, reduce="amax", include_self=False
        )
        token_cluster_max = cluster_max.gather(-1, index)
        exp_shifted = (token_logits - token_cluster_max).exp()
        cluster_sum = exp_shifted.new_zeros((*prefix_shape, self.c))
        cluster_sum = cluster_sum.scatter_add(-1, index, exp_shifted)
        cluster_lse = cluster_max + cluster_sum.clamp_min(1e-30).log()
        token_cluster_lse = cluster_lse.gather(-1, index)
        return token_logits - token_cluster_lse


def load_cluster_assignments(
    path: Path | str,
    in_tier: torch.Tensor,
    c: int,
    default_cluster: int = 0,
) -> torch.Tensor:
    """Read cluster_assignments.json (produced by an offline content-clustering pass), translate full-vocab ids to local-content ids via ``in_tier``, return cluster_id_local: (|V_content|,) long. ``in_tier`` is the content partition's full-vocab-id tensor. Vocab ids absent from the JSON (e.g. tokenizer-merged BPE ids outside the base tokens.csv) default to ``default_cluster``; tokens.csv-base ids must be present or this raises."""
    raw = json.loads(Path(path).read_text())
    full_assignments = raw["cluster_assignments"]
    indexed_max_vid = max((int(k) for k in full_assignments), default=-1)
    n_local = int(in_tier.numel())
    cluster_id_local = torch.full((n_local,), default_cluster, dtype=torch.long)
    in_tier_list = in_tier.tolist()
    for local_idx, full_vid in enumerate(in_tier_list):
        key = str(int(full_vid))
        if key in full_assignments:
            cluster_id_local[local_idx] = int(full_assignments[key])
        elif int(full_vid) <= indexed_max_vid:
            raise ValueError(
                f"vocab id {full_vid} (local {local_idx}) is within the index's "
                f"vocab range [0,{indexed_max_vid}] but missing from "
                f"cluster_assignments.json at {path}; tokenizer mismatch?"
            )
    if (cluster_id_local < 0).any() or (cluster_id_local >= c).any():
        raise ValueError(
            f"cluster ids out of [0,{c-1}] after translation: "
            f"min={int(cluster_id_local.min())} max={int(cluster_id_local.max())}"
        )
    return cluster_id_local
