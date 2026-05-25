"""Per-tier output heads + router. `MoSHead` (mixture-of-softmaxes) for the content tier when K > 0; plain Linear for other tiers. `per_tier_unified_log_p` combines the per-tier head outputs + router posterior into a unified (B,T,V_full) log-prob tensor via marginal-factorization disjoint scatter (the tier partition is disjoint, so each vocab id receives a contribution from exactly one tier)."""

import torch

from preframr.train.model.heads_cluster import (
    ClusterContentHead,
    load_cluster_assignments,
)
from preframr.train.model.heads_diffusion import DiffusionContentHead
from preframr.train.model.tier_map import _LOSS_TIER_ORDER


def _mos_log_mixture(component_log_p, gate_log_p):
    """Mix K softmax components by log gate weights. component_log_p: (..., K, V). gate_log_p: (..., K). Returns log-mixture: (..., V). Output dtype matches component_log_p so reduced-precision callers don't lose the bf16/fp16 cast that the upstream MoSHead applied."""
    mix = torch.logsumexp(gate_log_p.unsqueeze(-1) + component_log_p, dim=-2)
    if mix.dtype != component_log_p.dtype:
        mix = mix.to(component_log_p.dtype)
    return mix


class MoSHead(torch.nn.Module):
    def __init__(self, d, vocab_size, k):
        super().__init__()
        self.k = k
        self.components = torch.nn.ModuleList(
            [torch.nn.Linear(d, vocab_size) for _ in range(k)]
        )
        self.gate = torch.nn.Linear(d, k)

    def forward(self, h):
        """Stack K log_softmax components then mix by gate. Linear projections run under the caller's autocast (so fp32 weights auto-cast to bf16 for a bf16 input); the log_softmax + logsumexp pipeline runs with autocast disabled because those ops are on torch's fp32-promotion list and would otherwise force the (B,T,K,V_c) buffer to fp32 (~6 GiB at prodlike). On CUDA, fp32 logits are cast to bf16 before log_softmax so the head stays half-precision regardless of caller autocast state."""
        comp_logits = [c(h) for c in self.components]
        gate_logits = self.gate(h)
        with torch.amp.autocast(h.device.type, enabled=False):
            if h.is_cuda and comp_logits[0].dtype == torch.float32:
                comp_logits = [x.to(torch.bfloat16) for x in comp_logits]
                gate_logits = gate_logits.to(torch.bfloat16)
            comps = [torch.log_softmax(x, dim=-1) for x in comp_logits]
            comp_log_p = torch.stack(comps, dim=-2)
            gate_log_p = torch.log_softmax(gate_logits, dim=-1)
            return _mos_log_mixture(comp_log_p, gate_log_p), gate_log_p


class PerTierHeads(torch.nn.Module):
    def __init__(
        self,
        d,
        partition,
        mos_k,
        cluster_head_args=None,
        diffusion_head_args=None,
    ):
        super().__init__()
        self.tier_names = list(_LOSS_TIER_ORDER)
        self.mos_k = mos_k
        self.cluster_head_on = cluster_head_args is not None
        self.diffusion_head_on = diffusion_head_args is not None
        active_content_modes = sum(
            1 for x in (self.cluster_head_on, self.diffusion_head_on, mos_k > 0) if x
        )
        if active_content_modes > 1:
            raise ValueError(
                "at most one of {cluster_head_args, diffusion_head_args, "
                "mos_k > 0} may be active on the content tier"
            )
        self.heads = torch.nn.ModuleDict()
        for tier_name in self.tier_names:
            in_tier, _ = partition[tier_name]
            sub_v = max(int(in_tier.numel()), 1)
            if tier_name == "content" and self.cluster_head_on:
                c = int(cluster_head_args["c"])
                cluster_id_local = load_cluster_assignments(
                    cluster_head_args["index_path"], in_tier, c
                )
                self.heads[tier_name] = ClusterContentHead(
                    d, sub_v, c=c, cluster_id_local=cluster_id_local
                )
            elif tier_name == "content" and self.diffusion_head_on:
                self.heads[tier_name] = DiffusionContentHead(
                    d,
                    sub_v,
                    t_max=int(diffusion_head_args.get("t_max", 8)),
                    d_time=int(diffusion_head_args.get("d_time", 128)),
                )
            elif tier_name == "content" and mos_k > 0:
                self.heads[tier_name] = MoSHead(d, sub_v, k=mos_k)
            else:
                self.heads[tier_name] = torch.nn.Linear(d, sub_v)
        self.router = torch.nn.Linear(d, len(self.tier_names))

    def forward(self, h):
        out = {"router": self.router(h)}
        out["mos_gate_log_p"] = None
        out["cluster_log_p"] = None
        for tier_name in self.tier_names:
            head = self.heads[tier_name]
            if isinstance(head, MoSHead):
                log_p, gate_log_p = head(h)
                out[tier_name] = log_p
                out["mos_gate_log_p"] = gate_log_p
            elif isinstance(head, ClusterContentHead):
                joint_log_p, cluster_log_p = head(h)
                out[tier_name] = joint_log_p
                out["cluster_log_p"] = cluster_log_p
            elif isinstance(head, DiffusionContentHead):
                out[tier_name] = head(h)
            else:
                out[tier_name] = head(h)
        return out


def per_tier_unified_log_p(head_outputs, partition, n_vocab):
    """Combine per-tier head outputs + router posterior into a unified (B,T,V) log-prob tensor via marginal factorization. Empty tiers are masked from the router. Disables autocast and forces bf16 on CUDA so the inner log_softmax / scatter pipeline stays half-precision (fp32 would push the prodlike (B,T,V) buffer through ~1 GiB plus matching add intermediates)."""
    router_logits = head_outputs["router"]
    device = router_logits.device
    if router_logits.dtype in (torch.bfloat16, torch.float16):
        dtype = torch.bfloat16
    elif router_logits.is_cuda:
        dtype = torch.bfloat16
    else:
        dtype = router_logits.dtype
    if router_logits.dtype != dtype:
        router_logits = router_logits.to(dtype)
    active_mask = torch.tensor(
        [partition[t][0].numel() > 0 for t in _LOSS_TIER_ORDER],
        dtype=torch.bool,
        device=device,
    )
    with torch.amp.autocast(device.type, enabled=False):
        router_log_p = torch.log_softmax(
            router_logits.masked_fill(~active_mask, float("-inf")), dim=-1
        )
        bt_shape = router_logits.shape[:-1]
        result = torch.full(
            (*bt_shape, n_vocab), float("-inf"), device=device, dtype=dtype
        )
        any_active = False
        for tier_id, tier_name in enumerate(_LOSS_TIER_ORDER):
            in_tier, _ = partition[tier_name]
            if in_tier.numel() == 0:
                continue
            any_active = True
            tier_out = head_outputs[tier_name]
            if tier_out.dtype != dtype:
                tier_out = tier_out.to(dtype)
            if tier_name == "content":
                tier_log_p = tier_out
            else:
                tier_log_p = torch.log_softmax(tier_out, dim=-1)
            contribution = router_log_p[..., tier_id : tier_id + 1] + tier_log_p
            idx = in_tier.to(device)
            scatter_idx = idx.view(*([1] * (contribution.dim() - 1)), -1).expand_as(
                contribution
            )
            result = result.scatter(-1, scatter_idx, contribution)
    if not any_active:
        return torch.zeros((*bt_shape, n_vocab), device=device, dtype=dtype)
    return result
