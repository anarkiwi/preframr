"""Discrete-diffusion content head (D3PM absorbing-state variant). Plug-in shape: forward(h) returns the one-step fully-masked unified posterior so PerTierHeads + per_tier_unified_log_p + the lightning loss path stay unchanged at inference. Training uses forward(h, x_t, t) directly via the loss path in losses_diffusion.py."""

from __future__ import annotations

import math

import torch


def _sinusoidal_time_embed(t: torch.Tensor, d: int) -> torch.Tensor:
    """Standard sinusoidal time embedding. ``t`` shape (...,). Returns (..., d)."""
    half = d // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < d:
        pad = torch.zeros(*emb.shape[:-1], d - emb.shape[-1], device=t.device)
        emb = torch.cat([emb, pad], dim=-1)
    return emb


class DiffusionContentHead(torch.nn.Module):
    """D3PM absorbing-state content head. vocab_size is |V_content|; the head internally extends to vocab_size + 1 for the [MASK] absorbing token at index vocab_size."""

    def __init__(
        self,
        d: int,
        vocab_size: int,
        t_max: int = 8,
        d_time: int = 128,
    ):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.t_max = int(t_max)
        self.d_time = int(d_time)
        self.token_embed = torch.nn.Embedding(vocab_size + 1, d)
        self.time_proj = torch.nn.Linear(self.d_time, d)
        self.denoiser = torch.nn.Linear(d, vocab_size + 1)

    def forward(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-step denoising prediction. Returns log_p(x_0 | h, x_t, t) of shape (..., vocab_size) -- the [MASK] class is excluded from the output so the result slots into per_tier_unified_log_p like any other content-tier log-prob. If ``x_t`` and ``t`` are None: fully-masked inference posterior at ``t = t_max``."""
        prefix_shape = h.shape[:-1]
        if x_t is None:
            x_t = h.new_full(prefix_shape, self.mask_id, dtype=torch.long)
        if t is None:
            t = h.new_full((1,), self.t_max, dtype=torch.long)
        if t.numel() == 1:
            t_b = t.expand(*prefix_shape)
        elif t.shape == prefix_shape:
            t_b = t
        elif t.shape == prefix_shape[:1]:
            t_b = t.view(-1, *([1] * (len(prefix_shape) - 1))).expand(*prefix_shape)
        else:
            raise ValueError(
                f"t shape {tuple(t.shape)} incompatible with h prefix {tuple(prefix_shape)}"
            )
        tok = self.token_embed(x_t)
        t_emb_raw = _sinusoidal_time_embed(t_b, self.d_time).to(h.dtype)
        t_emb = self.time_proj(t_emb_raw)
        fused = h + tok + t_emb
        with torch.amp.autocast(h.device.type, enabled=False):
            logits = self.denoiser(fused)
            if h.is_cuda and logits.dtype == torch.float32:
                logits = logits.to(torch.bfloat16)
            log_p_full = torch.log_softmax(logits, dim=-1)
        log_p = log_p_full[..., : self.vocab_size]
        log_p = log_p - log_p.exp().sum(dim=-1, keepdim=True).clamp_min(1e-20).log()
        return log_p

    def forward_with_mask_class(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Training-side forward: returns log_p over vocab_size + 1 (including [MASK]), without renormalisation. The loss caller selects positions and computes NLL against the unmasked-class subset."""
        prefix_shape = h.shape[:-1]
        if t.shape == prefix_shape[:1]:
            t_b = t.view(-1, *([1] * (len(prefix_shape) - 1))).expand(*prefix_shape)
        elif t.shape == prefix_shape:
            t_b = t
        else:
            raise ValueError(
                f"t shape {tuple(t.shape)} incompatible with h prefix {tuple(prefix_shape)}"
            )
        tok = self.token_embed(x_t)
        t_emb_raw = _sinusoidal_time_embed(t_b, self.d_time).to(h.dtype)
        t_emb = self.time_proj(t_emb_raw)
        fused = h + tok + t_emb
        with torch.amp.autocast(h.device.type, enabled=False):
            logits = self.denoiser(fused)
            if h.is_cuda and logits.dtype == torch.float32:
                logits = logits.to(torch.bfloat16)
            log_p = torch.log_softmax(logits, dim=-1)
        return log_p
