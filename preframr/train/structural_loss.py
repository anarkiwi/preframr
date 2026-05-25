"""Auxiliary training-time loss penalising probability mass on
structurally-invalid continuations.
"""

import torch

from preframr_tokens import (
    StreamState,
    precompute_subtoken_arrays,
    precompute_vocab_arrays,
)
from preframr_tokens import RegTokenizer
from preframr_tokens.stfconstants import PAD_ID

_TRAIN_IRQ_SENTINEL = 1 << 30


class StructuralLoss:
    """Per-batch auxiliary loss helper. Build once; ``compute`` per step."""

    def __init__(self, args, n_vocab, tokens, tkmodel):
        self.args = args
        self.n_vocab = n_vocab
        self.tokens = tokens
        self.tkmodel = tkmodel
        self.vocab_arrays = None

    def _ensure_built(self):
        if self.vocab_arrays is not None:
            return
        if self.tkmodel is not None:
            tk_str = (
                self.tkmodel if isinstance(self.tkmodel, str) else self.tkmodel.to_str()
            )
            rt = RegTokenizer(self.args, tokens=self.tokens)
            rt.load(tk_str, self.tokens)
            self.vocab_arrays = precompute_subtoken_arrays(self.tokens, rt)
        else:
            self.vocab_arrays = precompute_vocab_arrays(self.tokens)

    def _new_state(self):
        return StreamState(
            self.vocab_arrays,
            init_frame_count=0,
            irq=_TRAIN_IRQ_SENTINEL,
            init_budget=_TRAIN_IRQ_SENTINEL,
            disable_resource_masks=True,
        )

    def compute(self, logits, targets, pad_mask):
        """Return scalar mean aux loss over non-pad positions."""
        self._ensure_built()
        bsz, tlen, _ = logits.shape
        full_lse = logits.logsumexp(-1)
        masked_lse_rows = []
        targets_cpu = targets.detach().to("cpu")
        for b in range(bsz):
            state = self._new_state()
            row_lse = []
            for t in range(tlen):
                masked = state.mask_logits(logits[b, t])
                row_lse.append(masked.logsumexp(-1))
                target_id = int(targets_cpu[b, t].item())
                if target_id == PAD_ID:
                    continue
                state.update(target_id)
            masked_lse_rows.append(torch.stack(row_lse))
        masked_lse = torch.stack(masked_lse_rows)
        aux = (full_lse - masked_lse) * pad_mask
        denom = pad_mask.sum().clamp(min=1.0)
        return aux.sum() / denom
