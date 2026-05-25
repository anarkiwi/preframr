"""Per-token training losses: chunked CE (memory-bounded), InfoNCE on content positions, per-vocab-id audio-frame weighting. All loss terms are pure torch; tier-aware terms read tier-id buffers built in `tier_map.py`."""

import torch
import torch.utils.checkpoint as _torch_checkpoint


def _infonce_per_tensor(logits, y, content_mask, k):
    """InfoNCE on content positions within a single (B,T,V) logits tensor."""
    import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

    pos = content_mask.nonzero()
    if pos.numel() == 0:
        return logits.new_zeros(()), 0
    pos_logits = logits[pos[:, 0], pos[:, 1]]
    gt = y[pos[:, 0], pos[:, 1]]
    n = gt.numel()
    v = logits.size(-1)
    distractors = torch.randint(v, (n, k), device=logits.device)
    gt_logit = pos_logits.gather(1, gt.unsqueeze(1))
    distractor_logits = pos_logits.gather(1, distractors)
    combined = torch.cat([gt_logit, distractor_logits], dim=1)
    target = torch.zeros(n, dtype=torch.long, device=logits.device)
    return F.cross_entropy(combined, target, reduction="sum"), n


def content_contrastive_loss(preds, y, content_mask, k=32):
    """InfoNCE auxiliary loss on content-tier positions; chunked + full preds supported."""
    if isinstance(preds, list):
        total_loss = preds[0].new_zeros(())
        total_n = 0
        offset = 0
        for chunk in preds:
            t_chunk = chunk.size(1)
            chunk_mask = content_mask[:, offset : offset + t_chunk]
            chunk_y = y[:, offset : offset + t_chunk]
            chunk_loss, chunk_n = _infonce_per_tensor(chunk, chunk_y, chunk_mask, k)
            total_loss = total_loss + chunk_loss
            total_n += chunk_n
            offset += t_chunk
        return total_loss / max(total_n, 1)
    chunk_loss, chunk_n = _infonce_per_tensor(preds, y, content_mask, k)
    return chunk_loss / max(chunk_n, 1)


def _cross_entropy_chunk(logits, target, label_smoothing):
    return torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction="none",
        label_smoothing=label_smoothing,
    )


def chunked_cross_entropy(
    logits,
    target,
    *,
    label_smoothing=0.0,
    chunk_bytes=512 * 1024 * 1024,
):
    """Memory-efficient per-token cross-entropy for large vocabularies."""
    if isinstance(logits, list):
        return _chunked_list_cross_entropy(
            logits, target, label_smoothing=label_smoothing
        )
    B, S, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_target = target.reshape(-1)
    n_rows = flat_logits.shape[0]
    bytes_per_elem = flat_logits.element_size()
    rows_per_chunk = max(1, int(chunk_bytes) // (V * bytes_per_elem))
    if rows_per_chunk >= n_rows:
        return _cross_entropy_chunk(flat_logits, flat_target, label_smoothing).view(
            B, S
        )
    use_ckpt = flat_logits.requires_grad and torch.is_grad_enabled()
    parts = []
    for i in range(0, n_rows, rows_per_chunk):
        j = min(i + rows_per_chunk, n_rows)
        chunk_l = flat_logits[i:j]
        chunk_t = flat_target[i:j]
        if use_ckpt:
            parts.append(
                _torch_checkpoint.checkpoint(
                    _cross_entropy_chunk,
                    chunk_l,
                    chunk_t,
                    label_smoothing,
                    use_reentrant=False,
                )
            )
        else:
            parts.append(_cross_entropy_chunk(chunk_l, chunk_t, label_smoothing))
    return torch.cat(parts, dim=0).view(B, S)


def _chunked_list_cross_entropy(logit_chunks, target, *, label_smoothing):
    """CE over a seq-dim-split list of logit chunks."""
    n = len(logit_chunks)
    target_chunks = list(target.chunk(n, dim=1))
    use_ckpt = torch.is_grad_enabled() and logit_chunks[0].requires_grad
    parts = []
    for logit_chunk, tgt_chunk in zip(logit_chunks, target_chunks):
        if use_ckpt:
            parts.append(
                _torch_checkpoint.checkpoint(
                    _cross_entropy_logit_chunk,
                    logit_chunk,
                    tgt_chunk,
                    label_smoothing,
                    use_reentrant=False,
                )
            )
        else:
            parts.append(
                _cross_entropy_logit_chunk(logit_chunk, tgt_chunk, label_smoothing)
            )
    return torch.cat(parts, dim=1)


def _cross_entropy_logit_chunk(logit_chunk, tgt_chunk, label_smoothing):
    """fp32-upcasted CE on one (B, S/N, V) chunk -> (B, S/N) per-token."""
    B, S_c, V = logit_chunk.shape
    flat_logits = logit_chunk.reshape(-1, V).float()
    flat_target = tgt_chunk.reshape(-1)
    per_tok = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_target,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    return per_tok.view(B, S_c)


def _build_vocab_frame_weight(args, n_vocab, tokens, tkmodel):
    """Per-vocab-id audio-frame weight used to scale per-token CE loss. Thin torch-side adapter; the weight switch lives in `preframr_tokens.token_weighting.vocab_frame_weights`."""
    if tokens is None or len(tokens) == 0:
        return torch.ones(n_vocab, dtype=torch.float32)
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        RegTokenizer,
    )
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        vocab_frame_weights,
    )

    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    return torch.from_numpy(vocab_frame_weights(rt, tokens, n_vocab))
