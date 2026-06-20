"""Per-token training loss: chunked cross-entropy (memory-bounded over the vocab dim). Pure torch; the BACC vocab is tiny so chunking is rarely triggered, but the helper is kept for large-vocab safety + parity with the validation path."""

import torch
import torch.utils.checkpoint as _torch_checkpoint


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
