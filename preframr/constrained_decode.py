"""Sampling-time logit guard for predict.py.

Masks structurally-invalid macro tokens at decode time so the LM cannot
emit a stream the safety net (``validate_back_refs`` /
``validate_gate_replays``) or ``expand_loops`` will reject post-hoc.

Scope (v1):
  * BACK_REF / PATTERN_REPLAY distance bounds (must reach within the
    running output frame buffer).
  * PATTERN_OVERLAY structural pairing with PATTERN_REPLAY (orphan
    overlays at top level are masked; inside an open PATTERN_REPLAY's
    overlay block only PATTERN_OVERLAY rows are legal).
  * GATE_REPLAY and PLAY_INSTRUMENT masked entirely. Both reference a
    palette grown by the encoder's ``DecodeState.observe_frame``;
    mirroring that growth in StreamState would require porting the
    full instrument-capture / gate-bundle bookkeeping. v1 trades the
    macro replays' expressiveness for guaranteed-resolvable streams;
    the underlying writes can still be emitted as literal SETs.
  * Per-frame diff budget: each row whose ``reg`` is a real SID reg
    (0..MAX_REG) consumes ``MIN_DIFF`` cycles. Mask those rows when
    the running frame budget can't absorb another ``MIN_DIFF``.
    FRAME_REG / DELAY_REG reset the budget. Without this, the model
    over-packs frames -- the FRAME_REG ``delay`` in ``_reset_diffs``
    goes negative and ``pyresidfp.clock`` aborts.

Follow-ups: DO_LOOP / END_REPEAT / END_FLIP nesting (END decoders are
no-ops so they don't crash; expand_loops handles orphan DO_LOOP_END
gracefully -- so these aren't blockers today). Palette tracking for
GATE_REPLAY / PLAY_INSTRUMENT once the encoder logic can be lifted
out.
"""

import numpy as np
import torch

from preframr.stfconstants import (
    BACK_REF_OP,
    DELAY_REG,
    FRAME_REG,
    GATE_REPLAY_OP,
    MAX_REG,
    MIN_DIFF,
    PAD_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    PLAY_INSTRUMENT_OP,
    SET_OP,
)


def _frame_marker_count(token_ids, is_frame_marker):
    """Number of frame-marker tokens in ``token_ids`` (a 1-D iterable of
    vocab indices)."""
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.size == 0:
        return 0
    return int(is_frame_marker[arr].sum().item())


def precompute_vocab_arrays(tokens_df, device):
    """Per-vocab-id tensors for the per-step mask. All length n_vocab."""
    n = len(tokens_df)
    op = tokens_df["op"].fillna(SET_OP).astype(np.int64).to_numpy()
    reg = tokens_df["reg"].astype(np.int64).to_numpy()
    subreg = tokens_df["subreg"].fillna(-1).astype(np.int64).to_numpy()
    val = tokens_df["val"].astype(np.int64).to_numpy()

    is_frame_marker = np.isin(reg, [FRAME_REG, DELAY_REG])
    is_delay_reg = reg == DELAY_REG
    is_pad = reg == PAD_REG
    # Real SID regs (0..MAX_REG) consume MIN_DIFF cycles of the per-frame
    # IRQ budget; loop-op / voice-rotation / marker rows are zero-cost.
    is_real_reg = (reg >= 0) & (reg <= MAX_REG)
    is_back_ref = op == BACK_REF_OP
    is_pattern_replay = op == PATTERN_REPLAY_OP
    is_pattern_overlay = op == PATTERN_OVERLAY_OP
    is_palette_macro = (op == GATE_REPLAY_OP) | (op == PLAY_INSTRUMENT_OP)

    distance = np.zeros(n, dtype=np.int64)
    has_distance = is_back_ref | is_pattern_replay
    distance[has_distance] = val[has_distance] >> 8

    overlay_count = np.zeros(n, dtype=np.int64)
    overlay_count[is_pattern_replay] = np.maximum(subreg[is_pattern_replay], 0)

    return {
        "n_vocab": n,
        "is_frame_marker": torch.from_numpy(is_frame_marker).to(device),
        "is_delay_reg": torch.from_numpy(is_delay_reg.astype(np.bool_)).to(device),
        "is_pad": torch.from_numpy(is_pad.astype(np.bool_)).to(device),
        "is_real_reg": torch.from_numpy(is_real_reg.astype(np.bool_)).to(device),
        "is_back_ref_or_pattern_replay": torch.from_numpy(
            (is_back_ref | is_pattern_replay).astype(np.bool_)
        ).to(device),
        "is_pattern_replay": torch.from_numpy(is_pattern_replay.astype(np.bool_)).to(
            device
        ),
        "is_pattern_overlay": torch.from_numpy(is_pattern_overlay.astype(np.bool_)).to(
            device
        ),
        "is_palette_macro": torch.from_numpy(is_palette_macro.astype(np.bool_)).to(
            device
        ),
        "distance": torch.from_numpy(distance).to(device),
        "overlay_count_cpu": overlay_count,
    }


class StreamState:
    """Per-step structural-validity tracker.

    Tracks the running output-frame counter and the open
    PATTERN_REPLAY's overlay-row debt. ``mask_logits`` returns a copy of
    the per-vocab logits with structurally-invalid positions set to
    ``-inf``; ``update`` advances state with the just-sampled token id.
    """

    def __init__(
        self, vocab_arrays, init_frame_count, irq, init_budget=None, logger=None
    ):
        self.arrays = vocab_arrays
        self.frame_count = int(init_frame_count)
        # When > 0, the next ``pending_overlays`` tokens must be
        # PATTERN_OVERLAY_OP rows (consumed by the open PATTERN_REPLAY).
        self.pending_overlays = 0
        # Per-frame budget tracking. ``irq`` is the IRQ window in cycles
        # (e.g. 19656 for PAL); each real-reg row consumes ``MIN_DIFF``.
        # ``init_budget`` is the budget remaining at the end of the
        # prompt; defaults to ``irq`` (full window) if caller can't
        # compute the prompt's last-frame consumption.
        self.irq = int(irq)
        self.frame_budget = int(init_budget) if init_budget is not None else int(irq)
        self.logger = logger
        self._stuck_warned = False

    def mask_logits(self, logits):
        """Set logits of structurally-invalid tokens to -inf.

        ``logits`` may be (vocab,) or (..., vocab); we mask the last dim.
        """
        a = self.arrays
        # Build invalid mask in the logits' dtype/device.
        invalid = torch.zeros(a["n_vocab"], dtype=torch.bool, device=logits.device)

        # Pad token (vocab idx 0) is always invalid in generation -- it
        # exists only to reserve PAD_ID's vocab slot, never to be emitted.
        invalid |= a["is_pad"]
        if self.pending_overlays > 0:
            # Inside a PATTERN_REPLAY's overlay block: only PATTERN_OVERLAY
            # rows are legal next.
            invalid |= ~a["is_pattern_overlay"]
        else:
            # Top-level: PATTERN_OVERLAY would be an orphan.
            invalid |= a["is_pattern_overlay"]
            # BACK_REF / PATTERN_REPLAY whose distance reaches before
            # frame 0 of the safety-net buffer.
            too_far = a["distance"] > self.frame_count
            invalid |= a["is_back_ref_or_pattern_replay"] & too_far
            # GATE_REPLAY / PLAY_INSTRUMENT reference an encoder-side
            # palette we don't yet mirror. Mask them entirely; the
            # underlying writes can still be issued as literal SETs.
            invalid |= a["is_palette_macro"]
            # DELAY_REG is a frame marker for variable-IRQ idle frames.
            # Training has FRAME_REG:DELAY_REG = 40:1 but constrained-
            # decode runs see the model fall onto DELAY_REG val=98 (a
            # ~2s pause) when budget is exhausted, blowing up generated
            # audio length. Mask DELAY_REG entirely so FRAME_REG carries
            # all marker traffic. Trade-off: lose representation of
            # idle-frame compression; acceptable in this regime.
            invalid |= a["is_delay_reg"]
            # Per-frame diff budget: if the next real-reg row would
            # overflow the IRQ window, mask all real-reg tokens. The
            # model's only legal moves are then FRAME_REG or zero-cost
            # ops (loop-op / voice-rotation).
            if self.frame_budget < MIN_DIFF:
                invalid |= a["is_real_reg"]

        if invalid.all():
            # All-masked safety valve: force a frame marker so the stream
            # advances and frame_count grows. Picking the first frame
            # marker token deterministically; the model's logits beyond
            # this fallback are irrelevant since we're already off-path.
            if self.logger is not None and not self._stuck_warned:
                self.logger.warning(
                    "constrained_decode: all tokens masked at frame=%u, "
                    "pending_overlays=%u; falling back to FRAME_REG",
                    self.frame_count,
                    self.pending_overlays,
                )
                self._stuck_warned = True
            invalid = invalid.clone()
            frame_idxs = torch.nonzero(a["is_frame_marker"], as_tuple=False)
            if frame_idxs.numel():
                invalid[int(frame_idxs[0].item())] = False

        masked = logits.clone()
        masked = masked.masked_fill(invalid, float("-inf"))
        return masked

    def update(self, token_id):
        """Advance state with the just-sampled token."""
        token_id = int(token_id)
        a = self.arrays
        if bool(a["is_frame_marker"][token_id].item()):
            self.frame_count += 1
            # FRAME_REG and DELAY_REG both reset the per-frame budget --
            # they're the markers between which ``_reset_diffs`` sums
            # the inner real-reg diffs.
            self.frame_budget = self.irq
        elif bool(a["is_real_reg"][token_id].item()):
            # Real-reg row: charge MIN_DIFF against the running budget.
            self.frame_budget -= MIN_DIFF
        if self.pending_overlays > 0:
            # Consume one overlay slot. We don't validate that the token
            # IS a PATTERN_OVERLAY here -- mask_logits already enforced
            # that, so any token reaching update() inside the overlay
            # block is one.
            self.pending_overlays -= 1
        elif bool(a["is_pattern_replay"][token_id].item()):
            self.pending_overlays = int(a["overlay_count_cpu"][token_id])
