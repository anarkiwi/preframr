"""D3PM absorbing-state training loss for DiffusionContentHead. Forward corruption masks each content position with prob alpha_t = cos(pi/2 * t/T); reverse loss is NLL against the masked tokens' ground truth restricted to the unmasked vocab classes."""

from __future__ import annotations

import math

import torch

from preframr.train.model.heads_diffusion import DiffusionContentHead


def sample_mask_schedule(
    content_mask: torch.Tensor,
    t_max: int,
    schedule: str = "cosine",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each content position (where ``content_mask`` is True), sample t ~ Uniform(1, t_max) and decide whether to mask via Bernoulli(alpha_t). Returns (t_per_position: long, masked_per_position: bool), both same shape as ``content_mask``. Non-content positions receive t=t_max (sentinel) and masked=False."""
    if schedule != "cosine":
        raise ValueError(f"unsupported schedule {schedule!r}; only 'cosine' so far")
    device = content_mask.device
    shape = content_mask.shape
    t = torch.full(shape, t_max, dtype=torch.long, device=device)
    masked = torch.zeros(shape, dtype=torch.bool, device=device)
    n_content = int(content_mask.sum().item())
    if n_content == 0:
        return t, masked
    if generator is not None:
        t_sample = torch.randint(
            1, t_max + 1, (n_content,), generator=generator, device=device
        )
        u = torch.rand(n_content, generator=generator, device=device)
    else:
        t_sample = torch.randint(1, t_max + 1, (n_content,), device=device)
        u = torch.rand(n_content, device=device)
    alpha = torch.cos(math.pi / 2 * t_sample.float() / t_max)
    is_masked = u < alpha
    t[content_mask] = t_sample
    masked[content_mask] = is_masked
    return t, masked


def discrete_diffusion_content_loss(
    head: DiffusionContentHead,
    h: torch.Tensor,
    gt: torch.Tensor,
    content_mask: torch.Tensor,
    full_to_local: torch.Tensor,
    t_max: int = 8,
    schedule: str = "cosine",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """D3PM absorbing-state NLL on content positions. ``gt`` is the full-vocab ground-truth ids (...,); ``content_mask`` selects content positions (...,) bool; ``full_to_local`` maps full-vocab id -> local content-vocab id (-1 if not content). Returns scalar loss; mean over masked content positions. Returns 0 if no positions get masked this batch."""
    t_pos, masked_pos = sample_mask_schedule(
        content_mask, t_max=t_max, schedule=schedule, generator=generator
    )
    if not masked_pos.any():
        return h.new_zeros(())
    gt_local = full_to_local[gt]
    x_t = gt_local.clone()
    x_t[masked_pos] = head.mask_id
    x_t[~content_mask] = head.mask_id
    log_p = head.forward_with_mask_class(h, x_t, t_pos)
    log_p_masked = log_p[masked_pos]
    gt_masked = gt_local[masked_pos]
    nll = -log_p_masked.gather(-1, gt_masked.unsqueeze(-1)).squeeze(-1)
    return nll.mean()
