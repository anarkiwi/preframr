"""Macro op infrastructure.

Encode-side `MacroPass` instances rewrite the parsed token DataFrame to use
typed macro ops; decode-side `MacroDecoder` instances are dispatched from
``RegLogParser._expand_ops`` to expand those tokens back to register writes.

Existing ops (``SET_OP``, ``DIFF_OP``, ``REPEAT_OP``, ``FLIP_OP``) are hosted
here as decoders. New macro ops plug in by adding a ``MacroPass`` to
``PASSES`` and a ``MacroDecoder`` to ``DECODERS`` without touching
``_expand_ops``.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Palettes:
    """Authoritative palette container threaded through the macro pipeline.

    ``GateMacroPass`` and ``InstrumentProgramPass`` publish into this; later
    passes (``SubregPass``, ``DedupSetPass``, ``_expand_ops``) seed their
    ``DecodeState`` from it. Replaces the former ``df.attrs["instrument_
    palette"]`` / ``df.attrs["gate_palette"]`` pattern -- pandas
    ``__finalize__`` deep-copied attrs on every dataframe op, and at
    palette sizes seen on rich songs that drove ~30% of parse wallclock.

    ``gate_palette`` is keyed by ``(voice, dir)`` and stores per-(v,d)
    lists of ``(ctrl, AD, SR)`` byte triples. ``instrument_palette`` is
    a flat list of programs, each a tuple of ``(rel_frame, reg_offset,
    val)`` triples.
    """

    instrument_palette: Optional[List[Tuple]] = None
    gate_palette: Optional[Dict[Tuple[int, int], List[Tuple[int, int, int]]]] = None


class _FastRow:
    """Lightweight row stand-in for ``decoder.expand``.

    ``itertuples`` allocates a ``namedtuple`` per row and accesses each
    cell through pandas' indexing layer (~300ns overhead per cell). For
    multi-million-row macro passes that adds up to most of parse time.
    The hot loops extract column arrays via ``.to_numpy()`` once, then
    construct one ``_FastRow`` per dispatched row from raw ints --
    skipping pandas entirely after the extraction.
    """

    __slots__ = ("reg", "val", "op", "subreg", "diff", "description", "Index")

    def __init__(self, reg, val, op, subreg, diff, description, Index):
        self.reg = reg
        self.val = val
        self.op = op
        self.subreg = subreg
        self.diff = diff
        self.description = description
        self.Index = Index


def _fastrow_from_arrs(arrs, i):
    """Build a ``_FastRow`` from the per-column arrays produced by
    ``_df_arrays_and_frames`` at row index ``i``. Centralises the
    repeated ``_FastRow(reg=int(regs[i]), val=int(vals[i]), ...)``
    pattern that appeared in 8+ pass walkers; one place to fix if the
    columns ever change.
    """
    return _FastRow(
        reg=int(arrs["reg"][i]),
        val=int(arrs["val"][i]),
        op=int(arrs["op"][i]),
        subreg=int(arrs["subreg"][i]),
        diff=int(arrs["diff"][i]),
        description=int(arrs["description"][i]),
        Index=int(arrs["Index"][i]),
    )


def _deserialize_gate_palette(attrs_value):
    """Restore a gate_palette dict from a ``df.attrs`` payload.

    ``GateMacroPass`` stores ``{(voice, dir): [list(bundle_tuple), ...]}``
    so the structure is JSON-serialisable. Decoders need
    ``{(voice, dir): [tuple(bundle), ...]}`` because GateReplayDecoder
    indexes the inner list with ``palette[idx]`` and bundle-comparison
    relies on tuple equality. This helper does the inverse conversion;
    pass through ``None``/``{}`` unchanged.
    """
    if not attrs_value:
        return None
    return {k: [tuple(b) for b in v] for k, v in attrs_value.items()}


def _frame_arrays(f_df):
    """Extract per-row arrays for a frame group. Used by hot pass loops
    that previously walked via ``itertuples`` (slow: per-row namedtuple +
    per-cell pandas indexing). Caller iterates by integer index.

    For passes that walk many frames, prefer ``_df_arrays_and_frames``
    below: extract whole-df columns once and slice via integer ranges,
    avoiding the per-group pandas column-access overhead.
    """
    cols = {
        "reg": f_df["reg"].to_numpy(),
        "val": f_df["val"].to_numpy(),
        "op": f_df["op"].to_numpy(),
        "diff": f_df["diff"].to_numpy(),
        "Index": f_df.index.to_numpy(),
    }
    if "subreg" in f_df.columns:
        cols["subreg"] = f_df["subreg"].to_numpy()
    else:
        cols["subreg"] = np.full(len(f_df), -1, dtype=np.int64)
    if "description" in f_df.columns:
        cols["description"] = f_df["description"].to_numpy()
    else:
        cols["description"] = np.zeros(len(f_df), dtype=np.int64)
    return cols


def _df_arrays_and_frames(df):
    """Extract whole-df column arrays once, plus frame-start positions.

    Returns ``(arrs, frame_starts)`` where ``arrs`` is the same dict
    shape as ``_frame_arrays`` and ``frame_starts`` is a numpy array of
    row positions where each logical frame begins (each FRAME_REG /
    DELAY_REG row starts a frame). The arrays use original df row
    positions, so callers slice ``arrs["reg"][start:end]`` etc. to walk
    a frame, and use ``arrs["Index"]`` for the original index values
    (used as ``__pos`` keys in ``_splice_rows``).
    """
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    ops = df["op"].to_numpy()
    diffs = df["diff"].to_numpy()
    subregs = (
        df["subreg"].to_numpy()
        if "subreg" in df.columns
        else np.full(len(df), -1, dtype=np.int64)
    )
    descs = (
        df["description"].to_numpy()
        if "description" in df.columns
        else np.zeros(len(df), dtype=np.int64)
    )
    indices = df.index.to_numpy()
    is_marker = (regs == FRAME_REG) | (regs == DELAY_REG)
    frame_starts = np.where(is_marker)[0]
    arrs = {
        "reg": regs,
        "val": vals,
        "op": ops,
        "diff": diffs,
        "subreg": subregs,
        "description": descs,
        "Index": indices,
    }
    return arrs, frame_starts


from preframr.stfconstants import (
    BACK_REF_OP,
    DELAY_REG,
    DIFF_OP,
    DO_LOOP_OP,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    END_FLIP_OP,
    END_REPEAT_OP,
    FC_LO_REG,
    FILTER_REG,
    FILTER_SWEEP_OP,
    FLIP2_OP,
    FLIP_OP,
    FRAME_REG,
    GATE_REPLAY_OP,
    INTERVAL_OP,
    LOOP_OP_REG,
    MIN_DIFF,
    MODE_VOL_REG,
    PLAY_INSTRUMENT_OP,
    PWM_OP,
    REPEAT_OP,
    SET_OP,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICES,
    VOICE_REG_SIZE,
)

# Frame markers (in encoder coordinates each row is one logical frame slot).
_FRAME_MARKER_REGS = {FRAME_REG, DELAY_REG}

# BACK_REF payload packing: (distance << 8) | length.
# Distance up to 2**24 = 16M frames (vastly more than any song); length 1..255.
_BACK_REF_LEN_MASK = 0xFF


def _pack_back_ref(distance, length):
    assert 1 <= length <= 255, length
    assert distance >= 1, distance
    return (int(distance) << 8) | int(length)


def _unpack_back_ref(val):
    val = int(val)
    return val >> 8, val & _BACK_REF_LEN_MASK


# PATTERN_OVERLAY target_reg sentinels (signalling body-wide modes).
# Real reg values are 0..24, so any value > 0xFE is free as a sentinel.
# Body-wide modes are emitted with subreg=-1 (no specific frame_offset);
# the decoder applies them across the whole replayed body.
OVERLAY_BODY_FREQ_DELTA = 0xFE  # subsumes BACK_REF_TRANSPOSED: signed delta
# (in val's low 16 bits) added to every freq
# SET in the body


# Per-voice register *bases* (relative to voice slot) whose byte value
# carries two semantically-independent nibbles. SubregPass runs before
# _add_voice_reg, so we expand to the absolute per-voice instances
# (4, 5, 6 for voice 0; 11, 12, 13 for voice 1; 18, 19, 20 for voice 2).
_PER_VOICE_SUBREG_BASES = (4, 5, 6)
SUBREG_REGS = tuple(
    base + v * VOICE_REG_SIZE for v in range(VOICES) for base in _PER_VOICE_SUBREG_BASES
) + (FILTER_REG, MODE_VOL_REG)

PWM_REGS_BY_VOICE = tuple(2 + v * VOICE_REG_SIZE for v in range(VOICES))
FREQ_REGS_BY_VOICE = tuple(0 + v * VOICE_REG_SIZE for v in range(VOICES))
CTRL_REGS_BY_VOICE = tuple(4 + v * VOICE_REG_SIZE for v in range(VOICES))
AD_REGS_BY_VOICE = tuple(5 + v * VOICE_REG_SIZE for v in range(VOICES))
SR_REGS_BY_VOICE = tuple(6 + v * VOICE_REG_SIZE for v in range(VOICES))
# (ctrl, AD, SR) reg triple for each voice -- the byte set whose end-of-frame
# values form a GateMacroPass bundle.
GATE_REGS_BY_VOICE = tuple(
    (CTRL_REGS_BY_VOICE[v], AD_REGS_BY_VOICE[v], SR_REGS_BY_VOICE[v])
    for v in range(VOICES)
)
_GATE_REG_TO_VOICE = {r: v for v in range(VOICES) for r in GATE_REGS_BY_VOICE[v]}
# Flat set of all gate-bundle regs across all voices. Instrument captures
# exclude these because gate-bundle writes are handled by GateMacroPass
# (literal SETs grow ``gate_palette``; replays go through GATE_REPLAY_OP).
# Keeping them out of captured instrument programs decouples
# ``instrument_palette`` evolution from gate-state evolution, so
# Phase 1's captured program tuples and the post-pass parse's
# capture from PlayInstrumentDecoder writes match deterministically.
_BUNDLE_REGS_FLAT = frozenset(
    reg for v in range(VOICES) for reg in GATE_REGS_BY_VOICE[v]
)


# ---------------------------------------------------------------------------
# Decode side
# ---------------------------------------------------------------------------
class DecodeState:
    """Per-stream state shared by all ``MacroDecoder`` invocations."""

    def __init__(
        self,
        frame_diff,
        last_diff=None,
        strict=False,
        gate_palette_cap=None,
        instrument_window=8,
        instrument_palette_cap=None,
        frozen_instrument_palette=None,
        frozen_gate_palette=None,
    ):
        self.frame_diff = frame_diff
        self.last_val = defaultdict(int)
        self.last_repeat = defaultdict(int)
        self.last_flip = defaultdict(int)
        self.last_diff = dict(last_diff) if last_diff else {}
        self.strict = strict
        # When non-None, ``observe_frame`` and ``GateMacroPass`` stop adding
        # to a (voice, direction) palette once it has this many entries.
        # Over-cap transitions stay literal -- vocab pressure traded against
        # long-tail compression.
        self.gate_palette_cap = gate_palette_cap
        # PWM/FILTER_SWEEP-style bursts: each pending entry is one frame's
        # delta. tick_frame consumes one per frame.
        self.pending_diffs = defaultdict(list)
        # INTERVAL: each entry is a dict {tgt, src, remaining}. tick_frame
        # mirrors the per-frame change in src's last_val into tgt.
        self.interval_links = []
        # Snapshot of last_val at the END of the previous frame's tick_frame.
        # Used by INTERVAL to compute "what diff did src receive this frame".
        self.prev_frame_val = {}
        # Subreg deferred coalescing (set by SetDecoder for subreg=0/1 rows;
        # cleared by SUBREG_FLUSH_OP / different-reg ops / same-nibble
        # repeats / frame boundaries). When non-None, holds the reg whose
        # nibble updates are pending, plus which nibbles have been touched
        # since the last flush.
        self.pending_subreg_reg = None
        self.pending_subreg_nibbles = set()
        # Per-voice running ctrl byte for gate-bit transition detection by
        # ``observe_frame``. Initialised to 0 (SID-reset state).
        self.last_ctrl = {v: 0 for v in range(VOICES)}
        # Gate-replay palette: (voice, direction) -> ordered list of bundle
        # tuples ``(ctrl_byte, ad_byte, sr_byte)``. Slot index = list index.
        # ``GateReplayDecoder`` looks up by slot; ``observe_frame`` appends
        # newly seen bundles when not frozen. ``frozen_gate_palette``
        # mirrors the instrument-palette pattern: the authoritative
        # palette is published by ``GateMacroPass`` to ``df.attrs`` and
        # downstream walkers initialise from it, so palette indices stay
        # aligned across walks even when intermediate passes (FuzzyLoop)
        # reorder or re-emit the literal SETs that originally grew it.
        if frozen_gate_palette is not None:
            self.gate_palette = {k: list(v) for k, v in frozen_gate_palette.items()}
            self.gate_palette_frozen = True
        else:
            self.gate_palette = {}
            self.gate_palette_frozen = False
        # Frame index at which each palette slot was first defined.
        # Populated by ``observe_frame`` when ``frame_idx`` is provided.
        self.gate_palette_def_frames = {}
        # Multi-frame instrument programs. ``instrument_palette[idx]`` is a
        # tuple of ``(rel_frame, reg_offset, val)`` triples. Reg offsets are
        # voice-relative (0..VOICE_REG_SIZE-1); the macro's ``reg`` field
        # carries the voice's ctrl reg, so absolute regs are reconstructed
        # at decode time by adding ``voice * VOICE_REG_SIZE``.
        # ``frozen_instrument_palette``: when non-None, the encoder has
        # already determined the authoritative palette and downstream
        # walkers initialise from it. ``_close_instr_capture`` then never
        # appends -- captures still resolve via ``last_closed_instr_captures``
        # for any caller that needs them, but palette indices are fixed.
        # This breaks the cross-pass observation divergence that previously
        # made InstrumentProgramPass unreliable: every walker sees the same
        # palette regardless of its local dispatch choices.
        if frozen_instrument_palette is not None:
            self.instrument_palette = list(frozen_instrument_palette)
            self.instrument_palette_frozen = True
        else:
            self.instrument_palette = []
            self.instrument_palette_frozen = False
        self.instrument_palette_def_frames = {}
        # Per-voice queue of pending future-frame writes scheduled by
        # ``PlayInstrumentDecoder``. Each list entry is one frame's worth
        # of ``(absolute_reg, val)`` writes, in order. ``tick_frame`` pops
        # one entry per voice per call -- this is what gives the macro its
        # multi-frame reach without any change to the per-row dispatcher.
        self.pending_program_writes = defaultdict(list)
        # Instrument-program capture state used by ``observe_frame``: per
        # voice ``{start_frame, length_cap, writes}``. A capture opens on
        # an off->on gate transition for the voice and closes on the next
        # gate event for that voice OR after ``length_cap`` logical frames,
        # whichever comes first. Closed captures whose program isn't yet in
        # ``instrument_palette`` get appended (subject to
        # ``instrument_palette_cap``).
        self.open_instr_captures = {}
        self.instrument_window = int(instrument_window)
        self.instrument_palette_cap = instrument_palette_cap
        # ``observe_frame`` populates this with ``{voice: (program, start_frame)}``
        # for any instrument captures it closed in the most recent call.
        # Encoder passes (``InstrumentProgramPass``) read it to decide
        # replay-vs-literal alongside their parallel row-index tracking.
        self.last_closed_instr_captures = {}
        # ``observe_frame`` populates this with one entry per gate
        # transition processed in the most recent call. Each entry is
        # ``(voice, direction, bundle, was_added, slot)``: ``was_added``
        # is True if observe_frame appended ``bundle`` to ``gate_palette
        # [(voice, direction)]`` (first occurrence), False if the bundle
        # was already present (replay) or skipped due to cap. ``slot`` is
        # the resulting palette index (or ``None`` if neither added nor
        # found, e.g. over cap and not previously seen). ``GateMacroPass``
        # reads this to decide replay-vs-literal so encoder and downstream
        # palettes evolve identically by construction.
        self.last_gate_transitions = []

    def diff_for(self, reg):
        return self.last_diff.get(reg, MIN_DIFF)

    def flush_pending_subreg(self):
        """Emit and clear any pending subreg state. Returns at most one
        write tuple ``(reg, val, diff)``; empty list if nothing pending."""
        if self.pending_subreg_reg is None:
            return []
        reg = self.pending_subreg_reg
        write = (reg, self.last_val[reg], self.diff_for(reg))
        self.pending_subreg_reg = None
        self.pending_subreg_nibbles = set()
        return [write]

    def maybe_flush_for(self, incoming_reg, incoming_subreg):
        """Decide whether the incoming row should flush pending subreg
        state. Returns the flush writes (possibly empty)."""
        if self.pending_subreg_reg is None:
            return []
        if self.pending_subreg_reg != incoming_reg:
            # Different reg -- the old pending is logically "between" this
            # row and any prior subreg, so flush it now.
            return self.flush_pending_subreg()
        # Same reg.
        if incoming_subreg in (0, 1):
            if incoming_subreg in self.pending_subreg_nibbles:
                # Same reg, same nibble already pending -> the about-to-be
                # set nibble would overwrite the pending one. Flush first.
                return self.flush_pending_subreg()
            # Same reg, different nibble -> coalesce into one byte write.
            return []
        # Same reg, non-subreg op (full-byte SET, DIFF, REPEAT, FLIP, etc.):
        # the pending subreg state needs to be observable before the new op
        # mutates the byte further.
        return self.flush_pending_subreg()

    def tick_frame(self):
        """Apply pending REPEAT/FLIP/PWM/INTERVAL/subreg ops at a frame boundary."""
        # Subreg flush first so the consolidated byte appears at the frame
        # boundary, before any tick-driven writes.
        writes = self.flush_pending_subreg()
        for reg, val in self.last_repeat.items():
            self.last_val[reg] += val
            writes.append((reg, self.last_val[reg], self.diff_for(reg)))
        for reg, val in list(self.last_flip.items()):
            self.last_val[reg] += val
            self.last_flip[reg] = -val
            writes.append((reg, self.last_val[reg], self.diff_for(reg)))
        for reg in list(self.pending_diffs.keys()):
            if not self.pending_diffs[reg]:
                del self.pending_diffs[reg]
                continue
            delta = self.pending_diffs[reg].pop(0)
            if not self.pending_diffs[reg]:
                del self.pending_diffs[reg]
            self.last_val[reg] += delta
            writes.append((reg, self.last_val[reg], self.diff_for(reg)))
        # INTERVAL: target voice mirrors the per-frame change in source's
        # value. Source's diff for *this* frame = current last_val[src] -
        # snapshot from end of previous frame.
        for link in list(self.interval_links):
            cur_src = self.last_val.get(link["src"], 0)
            prev_src = self.prev_frame_val.get(link["src"], 0)
            delta = cur_src - prev_src
            if delta != 0:
                self.last_val[link["tgt"]] += delta
                writes.append(
                    (
                        link["tgt"],
                        self.last_val[link["tgt"]],
                        self.diff_for(link["tgt"]),
                    )
                )
            link["remaining"] -= 1
            if link["remaining"] <= 0:
                self.interval_links.remove(link)
        # Multi-frame instrument programs: pop one frame's worth per voice.
        # An empty entry means this program has nothing for this frame
        # (sparse program); the entry is still consumed so subsequent
        # frames advance correctly. We do NOT skip writes whose val
        # already matches state.last_val: doing so would make
        # ``observe_frame``'s capture see fewer writes than Phase 1
        # recorded into the captured program tuple, so the post-pass
        # parse's ``instrument_palette`` and ``gate_palette`` would grow
        # at different positions than the encoder predicted, breaking
        # PLAY_INSTRUMENT_OP and GATE_REPLAY_OP slot references. The
        # cost is up to ~1 redundant SID write per replay (no token
        # cost), which is negligible compared to InstrumentProgramPass'
        # token reductions.
        for v in list(self.pending_program_writes.keys()):
            queue = self.pending_program_writes[v]
            if not queue:
                del self.pending_program_writes[v]
                continue
            frame_writes = queue.pop(0)
            for reg, val in frame_writes:
                self.last_val[reg] = val
                writes.append((reg, val, self.diff_for(reg)))
            if not queue:
                del self.pending_program_writes[v]
        # Snapshot for next frame's INTERVAL comparison.
        for reg, v in list(self.last_val.items()):
            self.prev_frame_val[reg] = v
        return writes

    def observe_frame(self, writes, frame_idx=None, track_instruments=True):
        """After a frame's writes are committed, update gate and instrument
        palettes by passive observation.

        ``track_instruments=False`` suppresses the instrument-program
        capture logic — used by ``InstrumentProgramPass`` which manages
        its own palette via first-occurrence detection on the pre-pass
        row stream. Downstream simulators leave the default so their
        ``state.instrument_palette`` builds up by walking the post-pass
        stream's emitted writes.

        - Detects per-voice gate-bit transitions and appends new
          ``(ctrl, AD, SR)`` bundles to ``gate_palette[(v, d)]``.
        - Maintains per-voice multi-frame instrument captures opened on
          off->on transitions; closes on the next gate event or after
          ``instrument_window`` logical frames; appends new programs to
          ``instrument_palette``.

        ``writes`` is the list of 3- or 4-tuples emitted this frame.
        ``frame_idx`` records the frame at which a slot was first defined
        (logical-slot coords, matching ``materialize_*_outside``).
        """
        # Reset the per-call closure ledger so encoder passes can read
        # what closed during this call without false carryover.
        self.last_closed_instr_captures = {}
        self.last_gate_transitions = []
        # Pre-compute per-voice writes for both the open-capture append
        # step and the new-capture initialiser.
        voice_writes = {v: [] for v in range(VOICES)}
        voices_with_ctrl = set()
        for w in writes:
            reg = int(w[0])
            if reg < 0:
                continue
            if reg in CTRL_REGS_BY_VOICE:
                voices_with_ctrl.add(_GATE_REG_TO_VOICE[reg])
            v = reg // VOICE_REG_SIZE
            if 0 <= v < VOICES and reg < (v + 1) * VOICE_REG_SIZE:
                voice_writes[v].append((reg, int(w[1])))

        # 1. Append this frame's writes to any pre-existing open captures
        #    and close any that have hit their length cap. Skipped when
        #    the caller manages its own captures. Gate-bundle regs are
        #    excluded -- they're handled by GateMacroPass / gate_palette.
        if track_instruments and frame_idx is not None and self.open_instr_captures:
            for v in list(self.open_instr_captures.keys()):
                cap = self.open_instr_captures[v]
                rel_frame = int(frame_idx) - cap["start_frame"]
                base = v * VOICE_REG_SIZE
                for reg, val in voice_writes[v]:
                    if reg in _BUNDLE_REGS_FLAT:
                        continue
                    cap["writes"].append((rel_frame, reg - base, val))
                if rel_frame + 1 >= cap["length_cap"]:
                    self._close_instr_capture(v)

        # 2. Process gate transitions per voice.
        for v in sorted(voices_with_ctrl):
            ctrl_reg, ad_reg, sr_reg = GATE_REGS_BY_VOICE[v]
            new_ctrl = int(self.last_val[ctrl_reg])
            old_gate = self.last_ctrl.get(v, 0) & 1
            new_gate = new_ctrl & 1
            if old_gate != new_gate:
                d = new_gate
                bundle = (
                    new_ctrl,
                    int(self.last_val[ad_reg]),
                    int(self.last_val[sr_reg]),
                )
                palette = self.gate_palette.setdefault((v, d), [])
                was_added = False
                slot = None
                if bundle in palette:
                    slot = palette.index(bundle)
                elif self.gate_palette_frozen:
                    # Frozen palette: the authoritative encoder published
                    # it via ``df.attrs``; downstream walkers must not
                    # mutate. A bundle observed here that isn't in the
                    # palette means the walker dispatched a write the
                    # encoder didn't account for (e.g., FuzzyLoopPass
                    # body replay's intermediate state). Leave slot=None
                    # so callers know there's no replay slot for it.
                    pass
                else:
                    if (
                        self.gate_palette_cap is None
                        or len(palette) < self.gate_palette_cap
                    ):
                        palette.append(bundle)
                        slot = len(palette) - 1
                        was_added = True
                        if frame_idx is not None:
                            self.gate_palette_def_frames[(v, d, slot)] = int(frame_idx)
                    # else: over cap, leave palette alone.
                self.last_gate_transitions.append((v, d, bundle, was_added, slot))

                # Instrument capture: close any open capture for this voice
                # (its program is what got us up to but not including this
                # transition's writes), then on gate-on open a fresh one
                # seeded with this frame's voice writes (rel_frame=0).
                if track_instruments:
                    if v in self.open_instr_captures:
                        self._close_instr_capture(v)
                    if new_gate == 1 and frame_idx is not None:
                        base = v * VOICE_REG_SIZE
                        self.open_instr_captures[v] = {
                            "start_frame": int(frame_idx),
                            "length_cap": self.instrument_window,
                            "writes": [
                                (0, reg - base, val)
                                for reg, val in voice_writes[v]
                                if reg not in _BUNDLE_REGS_FLAT
                            ],
                        }
            self.last_ctrl[v] = new_ctrl

    def _close_instr_capture(self, voice):
        """Finalise the open instrument capture for ``voice``, append its
        program to ``instrument_palette`` if new (subject to
        ``instrument_palette_cap``), and record the closure on
        ``last_closed_instr_captures`` so encoder passes can act on it.
        Returns the program tuple, or ``None`` if the capture had no writes.
        """
        cap = self.open_instr_captures.pop(voice, None)
        if cap is None or not cap["writes"]:
            return None
        # Canonicalise program tuple by sorting (rel_frame, reg_offset, val).
        # ``_norm_pr_order`` reorders rows within frames between the encoder
        # pass (which runs before ``_norm_pr_order``) and the downstream
        # ``_expand_ops`` walk (after), so the within-frame write order in
        # ``cap["writes"]`` may differ. Sorting makes the tuple a deterministic
        # function of the bytes only, so encoder and downstream agree.
        program = tuple(sorted(cap["writes"]))
        # Always announce the closure with start_frame so encoder passes
        # can decide replay-vs-literal independently of the palette.
        self.last_closed_instr_captures[voice] = (program, cap["start_frame"])
        if self.instrument_palette_frozen:
            # Palette is authoritative (set by the encoder); downstream
            # walks must not mutate it -- their captures may differ from
            # the encoder's captures because of local dispatch choices
            # (DedupSet drops, Subreg splits) but that doesn't change
            # which slot a PLAY_INSTRUMENT_OP in the row stream
            # references. Returning the program lets ``last_closed_instr_captures``
            # stay populated for callers that consume it.
            return program
        if program in self.instrument_palette:
            return program
        if (
            self.instrument_palette_cap is None
            or len(self.instrument_palette) < self.instrument_palette_cap
        ):
            self.instrument_palette.append(program)
            self.instrument_palette_def_frames[len(self.instrument_palette) - 1] = cap[
                "start_frame"
            ]
        return program


class MacroDecoder:
    """Base class for op decoders dispatched from ``_expand_ops``."""

    op_code = -1

    def expand(self, row, state):
        """Update ``state`` and return a list of writes (or None for no write).

        Each write is a 4-tuple ``(reg, val, diff, description)``.
        """
        raise NotImplementedError


class SetDecoder(MacroDecoder):
    op_code = SET_OP

    def expand(self, row, state):
        # First, flush any pending subreg if this row would otherwise corrupt
        # the deferred state (different reg, same-nibble repeat, or full-byte
        # SET on the same reg).
        pre = state.maybe_flush_for(row.reg, row.subreg)
        if row.subreg == 0:
            # Low-nibble update -- defer; the actual SID write is emitted
            # later by maybe_flush / SUBREG_FLUSH / tick_frame.
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0xF0) | int(row.val)
            state.last_diff[row.reg] = row.diff
            state.pending_subreg_reg = row.reg
            state.pending_subreg_nibbles.add(0)
            return pre or None
        if row.subreg == 1:
            # High-nibble update -- defer.
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0x0F) | (
                int(row.val) << 4
            )
            state.last_diff[row.reg] = row.diff
            state.pending_subreg_reg = row.reg
            state.pending_subreg_nibbles.add(1)
            return pre or None
        # Full-byte SET: apply state and emit immediately.
        state.last_val[row.reg] = row.val
        own = (row.reg, state.last_val[row.reg], row.diff, row.description)
        return pre + [own]


class DiffDecoder(MacroDecoder):
    op_code = DIFF_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, row.subreg)
        assert row.subreg == -1
        state.last_val[row.reg] += row.val
        own = (row.reg, state.last_val[row.reg], row.diff, row.description)
        return pre + [own]


class RepeatDecoder(MacroDecoder):
    op_code = REPEAT_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, row.subreg)
        assert row.subreg == -1
        if row.val == 0:
            state.last_val[row.reg] += state.last_repeat[row.reg]
            del state.last_repeat[row.reg]
            own = (row.reg, state.last_val[row.reg], row.diff, row.description)
            return pre + [own]
        if state.strict:
            assert row.reg not in state.last_repeat, (row.reg, state.last_repeat)
        state.last_repeat[row.reg] = row.val
        return pre or None


class FlipDecoder(MacroDecoder):
    op_code = FLIP_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, row.subreg)
        assert row.subreg == -1
        if row.val == 0:
            state.last_val[row.reg] += state.last_flip[row.reg]
            del state.last_flip[row.reg]
            own = (row.reg, state.last_val[row.reg], row.diff, row.description)
            return pre + [own]
        if state.strict:
            assert row.reg not in state.last_flip, (row.reg, state.last_flip)
        state.last_flip[row.reg] = row.val
        return pre or None


class _PendingDiffBurstDecoder(MacroDecoder):
    """Common decoder for burst ops that schedule N consecutive DIFFs.

    Subclasses set ``op_code``. Encodes:
      reg    -> target register
      val    -> per-frame delta
      subreg -> burst length (frames)
    """

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, -1)
        length = int(row.subreg)
        assert length > 0, (self.op_code, row)
        state.last_diff[row.reg] = row.diff
        # Queue N deltas; tick_frame consumes one per frame, including the
        # frame this row appears in (matches how REPEAT_OP/FLIP_OP behave).
        for _ in range(length):
            state.pending_diffs[row.reg].append(row.val)
        return pre or None


class PwmDecoder(_PendingDiffBurstDecoder):
    op_code = PWM_OP


class FilterSweepDecoder(_PendingDiffBurstDecoder):
    op_code = FILTER_SWEEP_OP


class Flip2Decoder(MacroDecoder):
    """Asymmetric ±a/±b alternation across N frames."""

    op_code = FLIP2_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, -1)
        # val packs (a << 8) | (b & 0xff), interpreted as signed 8-bit each.
        # subreg = burst length.
        length = int(row.subreg)
        assert length >= 2, row
        a = (int(row.val) >> 8) & 0xFF
        b = int(row.val) & 0xFF
        if a >= 128:
            a -= 256
        if b >= 128:
            b -= 256
        state.last_diff[row.reg] = row.diff
        # Queue length deltas: a, b, a, b, ... -- one per frame including this.
        for k in range(length):
            state.pending_diffs[row.reg].append(a if k % 2 == 0 else b)
        return pre or None


class TransposeDecoder(MacroDecoder):
    """Single-frame: apply same delta to multiple voices' freq regs."""

    op_code = TRANSPOSE_OP

    def expand(self, row, state):
        # val = delta (signed); subreg = voice mask (bit v set => voice v).
        delta = int(row.val)
        if delta >= 0x8000:
            delta -= 0x10000
        mask = int(row.subreg)
        # Flush pending subreg state on every freq reg this op will touch.
        pre = []
        for v in range(VOICES):
            if mask & (1 << v):
                pre.extend(state.maybe_flush_for(FREQ_REGS_BY_VOICE[v], -1))
        writes = []
        for v in range(VOICES):
            if mask & (1 << v):
                reg = FREQ_REGS_BY_VOICE[v]
                state.last_val[reg] += delta
                state.last_diff[reg] = row.diff
                writes.append((reg, state.last_val[reg], row.diff, row.description))
        return (pre + writes) if (pre or writes) else None


class IntervalDecoder(MacroDecoder):
    """Bind one voice's freq DIFF to another's for N frames."""

    op_code = INTERVAL_OP

    def expand(self, row, state):
        # val packs (target_voice << 4) | source_voice; subreg = length.
        length = int(row.subreg)
        assert length > 0
        tgt_v = (int(row.val) >> 4) & 0xF
        src_v = int(row.val) & 0xF
        tgt_reg = FREQ_REGS_BY_VOICE[tgt_v]
        src_reg = FREQ_REGS_BY_VOICE[src_v]
        pre = state.maybe_flush_for(tgt_reg, -1)
        state.last_diff[tgt_reg] = row.diff
        state.interval_links.append(
            {"tgt": tgt_reg, "src": src_reg, "remaining": length}
        )
        return pre or None


class _EndOpDecoder(MacroDecoder):
    """Explicit terminator for REPEAT/FLIP runs.

    Used purely as an LM-predictability marker; produces no writes because
    the actual terminating write is emitted by the corresponding RepeatDecoder
    or FlipDecoder when it sees val=0 (which, when terminators are enabled,
    is what immediately follows this token in the stream).
    """

    def expand(self, row, state):
        return None


class EndRepeatDecoder(_EndOpDecoder):
    op_code = END_REPEAT_OP


class EndFlipDecoder(_EndOpDecoder):
    op_code = END_FLIP_OP


class GateReplayDecoder(MacroDecoder):
    """Replay a previously seen ``(ctrl, AD, SR)`` bundle for a voice.

    Encoded as ``op=GATE_REPLAY_OP``, ``reg`` = the voice's ctrl reg (so
    voice rotation rotates correctly), ``subreg`` = direction (0 = on->off,
    1 = off->on), ``val`` = palette slot index. The palette is built up
    passively in ``DecodeState.observe_frame`` whenever a literal bundle's
    end-of-frame state introduces a new ``(ctrl, AD, SR)`` triple, so
    encoder and decoder palettes stay in sync.

    On dispatch, only emit writes for regs whose running value differs
    from the bundle's target -- avoids re-introducing the redundant SET
    behavior ``DedupSetPass`` removes.
    """

    op_code = GATE_REPLAY_OP

    def expand(self, row, state):
        v = int(row.reg) // VOICE_REG_SIZE
        d = int(row.subreg) & 1
        idx = int(row.val)
        bundle = state.gate_palette.get((v, d))
        assert bundle is not None and idx < len(bundle), (
            f"GATE_REPLAY references undefined palette slot "
            f"(voice={v}, dir={d}, idx={idx})"
        )
        target_ctrl, target_ad, target_sr = bundle[idx]
        ctrl_reg, ad_reg, sr_reg = GATE_REGS_BY_VOICE[v]
        writes = []
        for reg, target in (
            (ctrl_reg, target_ctrl),
            (ad_reg, target_ad),
            (sr_reg, target_sr),
        ):
            if int(state.last_val[reg]) == int(target):
                continue
            writes.extend(state.maybe_flush_for(reg, -1))
            state.last_val[reg] = int(target)
            writes.append((reg, int(target), state.diff_for(reg), row.description))
        return writes if writes else None


class PlayInstrumentDecoder(MacroDecoder):
    """Replay a previously seen multi-frame ``(reg, val)`` program for a
    voice.

    Encoded as ``op=PLAY_INSTRUMENT_OP``, ``reg`` = the voice's ctrl reg
    (so voice rotation rotates correctly), ``val`` = palette slot index,
    ``subreg`` = program length in frames. The palette is built up
    passively in ``DecodeState.observe_frame`` by the encoder pass and
    mirrored here as a sequence of per-frame write lists.

    On dispatch, the program's frame-0 writes hit this frame's
    ``tick_frame`` and frames 1..L-1 hit subsequent ``tick_frame`` calls,
    so the macro reaches L frames forward into the stream. Other voices'
    rows in those frames pass through normally.
    """

    op_code = PLAY_INSTRUMENT_OP

    def expand(self, row, state):
        v = int(row.reg) // VOICE_REG_SIZE
        idx = int(row.val)
        assert 0 <= idx < len(state.instrument_palette), (
            f"PLAY_INSTRUMENT references undefined slot "
            f"(voice={v}, idx={idx}, palette_size={len(state.instrument_palette)})"
        )
        program = state.instrument_palette[idx]
        if not program:
            return None
        # Group program writes by their relative frame index, resolving
        # voice-relative reg offsets to absolute regs.
        voice_base = v * VOICE_REG_SIZE
        max_rel = max(rel for (rel, _, _) in program)
        per_frame = [[] for _ in range(max_rel + 1)]
        for rel_frame, reg_offset, val in program:
            per_frame[rel_frame].append((voice_base + int(reg_offset), int(val)))
        # Flush any pending subreg state on the regs we'll write so the
        # decoder doesn't coalesce a deferred nibble into our program.
        pre = []
        touched = {voice_base + int(ro) for (_, ro, _) in program}
        for reg in touched:
            pre.extend(state.maybe_flush_for(reg, -1))
        # Schedule the per-frame writes onto this voice's queue. A prior
        # program on the same voice should have closed at gate-on (see
        # encoder pass), so overwriting is the right policy.
        state.pending_program_writes[v] = per_frame
        return pre or None


class SubregFlushDecoder(MacroDecoder):
    """Force-flush deferred subreg state. Inserted by SubregPass between two
    consecutive subreg rows that are on the same reg, touch different
    nibbles, AND came from different baseline SETs (so they would otherwise
    coalesce and lose the intermediate write)."""

    op_code = SUBREG_FLUSH_OP

    def expand(self, row, state):
        return state.flush_pending_subreg() or None


DECODERS = {
    d.op_code: d
    for d in (
        SetDecoder(),
        DiffDecoder(),
        RepeatDecoder(),
        FlipDecoder(),
        PwmDecoder(),
        TransposeDecoder(),
        Flip2Decoder(),
        IntervalDecoder(),
        EndRepeatDecoder(),
        EndFlipDecoder(),
        FilterSweepDecoder(),
        SubregFlushDecoder(),
        GateReplayDecoder(),
        PlayInstrumentDecoder(),
    )
}


# ---------------------------------------------------------------------------
# Encode side
# ---------------------------------------------------------------------------
class MacroPass:
    """Base class for encode-side passes operating on a token DataFrame."""

    def apply(self, df, args=None, palettes=None):
        raise NotImplementedError


def _frame_index(df):
    """Cumulative frame index for each row (boundary at FRAME_REG/DELAY_REG)."""
    return df["reg"].isin({FRAME_REG, DELAY_REG}).astype(int).cumsum()


def _ensure_subreg(df):
    if "subreg" not in df.columns:
        df = df.copy()
        df["subreg"] = -1
    return df


def _splice_rows(df, drop_idx, new_rows):
    """Drop rows by index and splice ``new_rows`` (each carrying ``__pos``)
    into their original positions, preserving the rest of the row order.

    Critically: preserve the original column dtypes. ``pd.concat`` with a
    plain-int-built ``new_df`` would promote ``UInt16`` etc. to ``Int64``,
    which changes downstream behavior (e.g. ``Series.diff()`` on a
    nullable-int column treats the leading row's NaN differently from on a
    regular Int64 column, perturbing ``_norm_df``'s frame-boundary v-reset).
    """
    if not new_rows:
        return df
    # Capture df.attrs before any concat -- ``pd.concat`` over multiple
    # dfs drops attrs, so we re-attach after.
    orig_attrs = dict(df.attrs)
    df = _ensure_subreg(df)
    irq_value = (
        int(df["irq"].iloc[0])
        if "irq" in df.columns and len(df) and df["irq"].notna().any()
        else -1
    )
    orig_dtypes = df.dtypes.to_dict()
    df = df.drop(index=drop_idx)
    df["__pos"] = df.index.astype("int64")
    new_df = pd.DataFrame(new_rows)
    for col in df.columns:
        if col not in new_df.columns:
            if col == "description":
                new_df[col] = 0
            elif col == "irq":
                new_df[col] = irq_value
            else:
                new_df[col] = -1
    new_df = new_df[df.columns]
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.sort_values("__pos", kind="stable").reset_index(drop=True)
    combined = combined.drop(columns=["__pos"])
    # Restore original dtypes for columns that had them.
    for col, dt in orig_dtypes.items():
        if col == "__pos":
            continue
        try:
            combined[col] = combined[col].astype(dt)
        except (TypeError, ValueError):
            pass
    if orig_attrs:
        combined.attrs.update(orig_attrs)
    return combined


class PwmPass(MacroPass):
    """Collapse runs of consecutive identical PWM DIFFs into a PWM_OP burst.

    Operates on the per-voice pulse-width register (reg 2/9/16). Runs of
    length >= 2 are absorbed; the burst row encodes ``(reg, step, length)``
    in ``(reg, val, subreg)``. Voice rotation rotates the burst's reg field.
    """

    target_regs = PWM_REGS_BY_VOICE
    min_run = 2

    def apply(self, df, args=None, palettes=None):
        # Sparse early-out: skip the per-reg loop entirely if no DIFF
        # rows touch any PWM target reg. Cheap mask-and-any() scan
        # avoids the f_idx/reset_index/copy overhead on small slices
        # that have no PWM activity (the common case for short
        # self-contained blocks).
        if (
            "op" not in df.columns
            or not (df["op"].eq(DIFF_OP) & df["reg"].isin(self.target_regs)).any()
        ):
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        drop_idx = []
        new_rows = []
        for reg in self.target_regs:
            mask = (df["reg"] == reg) & (df["op"] == DIFF_OP)
            sub = df[mask]
            if sub.empty:
                continue
            indices = sub.index.tolist()
            frames = sub["mf"].tolist()
            vals = sub["val"].tolist()
            diffs = sub["diff"].tolist()
            i = 0
            n = len(indices)
            while i < n:
                step = vals[i]
                j = i
                while (
                    j + 1 < n and frames[j + 1] == frames[j] + 1 and vals[j + 1] == step
                ):
                    j += 1
                run_len = j - i + 1
                if run_len >= self.min_run:
                    drop_idx.extend(indices[i : j + 1])
                    new_rows.append(
                        {
                            "reg": int(reg),
                            "val": int(step),
                            "diff": int(diffs[i]),
                            "op": int(PWM_OP),
                            "subreg": int(run_len),
                            "__pos": int(indices[i]),
                        }
                    )
                i = j + 1

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


class TransposePass(MacroPass):
    """Within one frame, collapse same-delta freq DIFFs across >=2 voices.

    Replaces the matching DIFF rows with one TRANSPOSE_OP row carrying
    ``(delta, voice_mask)`` in ``(val, subreg)``. The reg field is the
    smallest voice's freq reg (so voice rotation still rotates correctly,
    though the mask is recomputed at decode time per voice index).
    """

    target_regs = FREQ_REGS_BY_VOICE

    def apply(self, df, args=None, palettes=None):
        # Sparse early-out: TransposePass needs at least 2 voices'
        # freq-DIFFs in the same frame. Cheap any() check on freq-DIFFs
        # avoids the reset_index/copy/numpy-conversion overhead on
        # blocks with no freq-DIFF activity.
        if (
            "op" not in df.columns
            or not (df["op"].eq(DIFF_OP) & df["reg"].isin(self.target_regs)).any()
        ):
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)

        # Pre-filter to the only rows TransposePass acts on: freq-DIFFs
        # on a target reg. Iterating ``df.groupby("mf")`` over every
        # frame (including the vast majority that have no qualifying
        # rows) was the dominant cost; restricting to the filtered
        # subset cuts work to actual-match frames.
        target_set = set(self.target_regs)
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        target_mask = np.isin(regs, list(target_set)) & (ops == DIFF_OP)
        if not target_mask.any():
            return df
        sub = df[target_mask].copy()
        sub["mf"] = f_idx[target_mask]

        drop_idx = []
        new_rows = []
        # Group by (frame, val). Each group with >= 2 voices on the same
        # diff value collapses to one TRANSPOSE_OP.
        target_reg_to_voice = {r: v for v, r in enumerate(self.target_regs)}
        for (_, val), grp in sub.groupby(["mf", "val"], sort=False):
            if len(grp) < 2:
                continue
            grp_regs = grp["reg"].to_numpy()
            grp_idx = grp.index.to_numpy()
            voice_mask = 0
            for r in grp_regs:
                voice_mask |= 1 << target_reg_to_voice[int(r)]
            drop_idx.extend(int(i) for i in grp_idx)
            first_reg = int(grp_regs.min())
            new_rows.append(
                {
                    "reg": first_reg,
                    "val": int(val) & 0xFFFF,
                    "diff": int(grp["diff"].iloc[0]),
                    "op": int(TRANSPOSE_OP),
                    "subreg": int(voice_mask),
                    "__pos": int(grp_idx.min()),
                }
            )
        return _splice_rows(df, drop_idx, new_rows)


class Flip2Pass(MacroPass):
    """Asymmetric ±a/±b alternation across consecutive frames per (reg, voice).

    Skips symmetric flips (handled by FLIP_OP) and zero-valued steps. Burst
    encodes ``(a, b, length)`` in ``(val_packed, subreg)``.
    """

    min_run = 3

    def apply(self, df, args=None, palettes=None):
        # Sparse early-out: Flip2Pass needs DIFF rows. Slices with no
        # DIFFs (literal-only blocks) skip immediately.
        if "op" not in df.columns or not df["op"].eq(DIFF_OP).any():
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        drop_idx = []
        new_rows = []
        diff_rows = df[(df["op"] == DIFF_OP) & (df["reg"] >= 0)]
        for reg, sub in diff_rows.groupby("reg"):
            indices = sub.index.tolist()
            frames = sub["mf"].tolist()
            vals = sub["val"].tolist()
            diffs = sub["diff"].tolist()
            i = 0
            n = len(indices)
            while i < n - 1:
                a = vals[i]
                b = vals[i + 1]
                if (
                    a == 0
                    or b == 0
                    or a == b
                    or abs(a) == abs(b)
                    or frames[i + 1] != frames[i] + 1
                ):
                    i += 1
                    continue
                j = i + 2
                while (
                    j < n
                    and frames[j] == frames[j - 1] + 1
                    and vals[j] == (a if (j - i) % 2 == 0 else b)
                ):
                    j += 1
                run_len = j - i
                if run_len >= self.min_run:
                    drop_idx.extend(indices[i:j])
                    packed = ((a & 0xFF) << 8) | (b & 0xFF)
                    new_rows.append(
                        {
                            "reg": int(reg),
                            "val": int(packed),
                            "diff": int(diffs[i]),
                            "op": int(FLIP2_OP),
                            "subreg": int(run_len),
                            "__pos": int(indices[i]),
                        }
                    )
                    i = j
                else:
                    i += 1

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


class IntervalPass(MacroPass):
    """One voice's freq DIFFs match another's, frame-by-frame, for N frames.

    Replaces the dependent voice's DIFF rows with INTERVAL_OP at the start
    of the run; the decoder mirrors source DIFFs to the target voice.
    """

    min_run = 2
    target_regs = FREQ_REGS_BY_VOICE

    def apply(self, df, args=None, palettes=None):
        # Sparse early-out: IntervalPass needs freq-DIFFs from at least
        # two voices in the same frame. If there are no freq-DIFFs at
        # all, exit before any work.
        if (
            "op" not in df.columns
            or not (df["op"].eq(DIFF_OP) & df["reg"].isin(self.target_regs)).any()
        ):
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        # Per voice, build per-frame freq-DIFF map
        n_frames = int(f_idx.max() + 1) if len(f_idx) else 0
        per_voice = {v: {} for v in range(VOICES)}  # voice -> {frame: (idx, val, diff)}
        for v, reg in enumerate(self.target_regs):
            mask = (df["reg"] == reg) & (df["op"] == DIFF_OP)
            sub = df[mask]
            sub_mf = sub["mf"].to_numpy()
            sub_val = sub["val"].to_numpy()
            sub_diff = sub["diff"].to_numpy()
            sub_idx = sub.index.to_numpy()
            for k in range(len(sub)):
                per_voice[v][int(sub_mf[k])] = (
                    int(sub_idx[k]),
                    int(sub_val[k]),
                    int(sub_diff[k]),
                )

        drop_idx = []
        new_rows = []
        used_target_frames = {v: set() for v in range(VOICES)}
        # Greedy: scan each (target, source) pair for runs of length >= min_run.
        # Restrict to tgt > src so each pair is considered exactly once -- the
        # higher-indexed voice tracks the lower-indexed one, leaving the
        # lower's DIFFs intact as the source.
        for src_v in range(VOICES):
            for tgt_v in range(src_v + 1, VOICES):
                f = 0
                while f < n_frames:
                    if (
                        f not in per_voice[tgt_v]
                        or f not in per_voice[src_v]
                        or per_voice[tgt_v][f][1] != per_voice[src_v][f][1]
                        or f in used_target_frames[tgt_v]
                    ):
                        f += 1
                        continue
                    g = f
                    while (
                        g + 1 < n_frames
                        and (g + 1) in per_voice[tgt_v]
                        and (g + 1) in per_voice[src_v]
                        and per_voice[tgt_v][g + 1][1] == per_voice[src_v][g + 1][1]
                        and (g + 1) not in used_target_frames[tgt_v]
                    ):
                        g += 1
                    run_len = g - f + 1
                    if run_len >= self.min_run:
                        for h in range(f, g + 1):
                            drop_idx.append(per_voice[tgt_v][h][0])
                            used_target_frames[tgt_v].add(h)
                        first_idx, _v, first_diff = per_voice[tgt_v][f]
                        new_rows.append(
                            {
                                "reg": int(self.target_regs[tgt_v]),
                                "val": int(((tgt_v & 0xF) << 4) | (src_v & 0xF)),
                                "diff": int(first_diff),
                                "op": int(INTERVAL_OP),
                                "subreg": int(run_len),
                                "__pos": int(first_idx),
                            }
                        )
                        f = g + 1
                    else:
                        f += 1

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


class FilterSweepPass(MacroPass):
    """Mirror of PwmPass for the filter cutoff register (FC_LO_REG)."""

    target_regs = (FC_LO_REG,)
    min_run = 2

    def apply(self, df, args=None, palettes=None):
        # Sparse early-out: FilterSweepPass only acts on FC_LO_REG
        # writes. Slices without any filter-cutoff activity skip.
        if not df["reg"].eq(FC_LO_REG).any():
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        drop_idx = []
        new_rows = []
        mask = (df["reg"] == FC_LO_REG) & (df["op"] == DIFF_OP)
        sub = df[mask]
        if not sub.empty:
            indices = sub.index.tolist()
            frames = sub["mf"].tolist()
            vals = sub["val"].tolist()
            diffs = sub["diff"].tolist()
            i = 0
            n = len(indices)
            while i < n:
                step = vals[i]
                j = i
                while (
                    j + 1 < n and frames[j + 1] == frames[j] + 1 and vals[j + 1] == step
                ):
                    j += 1
                run_len = j - i + 1
                if run_len >= self.min_run:
                    drop_idx.extend(indices[i : j + 1])
                    new_rows.append(
                        {
                            "reg": int(FC_LO_REG),
                            "val": int(step),
                            "diff": int(diffs[i]),
                            "op": int(FILTER_SWEEP_OP),
                            "subreg": int(run_len),
                            "__pos": int(indices[i]),
                        }
                    )
                i = j + 1

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


class SubregPass(MacroPass):
    """Always-split byte-to-nibble for subreg-eligible registers, with
    SUBREG_FLUSH inserted to preserve byte-equality across intra-frame
    multi-write sequences.

    For each SET on a reg in ``SUBREG_REGS``, compare against the last value
    seen on that reg and emit nibble rows:

      - lo only changed -> one ``subreg=0`` row
      - hi only changed -> one ``subreg=1`` row
      - both nibbles changed -> two rows: ``subreg=0`` then ``subreg=1``
      - no change -> leave SET as-is (subreg=-1, redundant-but-harmless)

    The decoder defers ``subreg=0/1`` updates and emits one consolidated
    SID write at frame boundaries (or earlier when a different reg / a
    same-nibble repeat / a non-subreg op forces a flush). To preserve
    byte-equality with the baseline in the case where two adjacent baseline
    SETs each change only one (different) nibble (which would otherwise
    coalesce into one write), the encoder inserts a ``SUBREG_FLUSH_OP`` row
    between them.

    Subsumes the previous ``GateTogglePass`` and ``FilterModeVolPass``;
    eliminates the ``subreg=-1`` byte-vocab entries on subreg-eligible regs
    in exchange for slightly longer streams on both-nibble events.
    """

    target_regs = SUBREG_REGS

    def apply(self, df, args=None, palettes=None):
        # Sparse early-out: SubregPass only splits ctrl/AD/SR SETs.
        # Slices with none skip the DecodeState construction below.
        if not df["reg"].isin(self.target_regs).any():
            return df
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        if "description" not in df.columns:
            df["description"] = 0
        # SubregPass's nibble-split decisions need the *actual* prior register
        # value, which on a stream containing GATE_REPLAY_OP can differ from
        # "value of the last SET row". Drive a real DecodeState through every
        # row's decoder so GATE_REPLAY's writes show up in last_val, and use
        # state.last_val[reg] as the SET's pre-image.
        state = _build_decode_state(df, args=args, palettes=palettes)

        last_emitted_reg = None  # most recently emitted subreg row's reg
        last_emitted_nib = None  # ... and nibble (0 or 1)
        drop_idx = []
        new_rows = []

        def dispatch(row):
            if state is None:
                return None
            decoder = DECODERS.get(int(row.op))
            if decoder is None:
                return None
            return decoder.expand(row, state)

        arrs, frame_starts = _df_arrays_and_frames(df)
        regs = arrs["reg"]
        vals = arrs["val"]
        ops = arrs["op"]
        subregs = arrs["subreg"]
        diffs = arrs["diff"]
        descs = arrs["description"]
        indices = arrs["Index"]
        n_total = len(df)
        n_frames = len(frame_starts)

        for fi in range(n_frames):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            cur_frame = fi
            f_writes = []
            for i in range(start, end):
                reg = int(regs[i])
                if reg < 0:
                    if state is not None and reg == DELAY_REG:
                        for _ in range(int(vals[i]) - 1):
                            state.tick_frame()
                            state.observe_frame([], frame_idx=cur_frame)
                    last_emitted_reg = None
                    last_emitted_nib = None
                    continue
                op = int(ops[i])
                subreg = int(subregs[i])
                row_idx = int(indices[i])
                row_diff = int(diffs[i])
                row_desc = int(descs[i])
                row_val = int(vals[i])
                if reg not in self.target_regs or op != SET_OP:
                    # Non-target row: dispatch to update state, then reset
                    # the SUBREG_FLUSH bookkeeping (decoder flushes naturally
                    # when the next subreg row arrives on a different reg).
                    fast_row = _FastRow(
                        reg=reg,
                        val=row_val,
                        op=op,
                        subreg=subreg,
                        diff=row_diff,
                        description=row_desc,
                        Index=row_idx,
                    )
                    writes = dispatch(fast_row)
                    if writes:
                        f_writes.extend(writes)
                    last_emitted_reg = None
                    last_emitted_nib = None
                    continue
                if subreg != -1:
                    # Already-split row: dispatch and remember nibble.
                    fast_row = _FastRow(
                        reg=reg,
                        val=row_val,
                        op=op,
                        subreg=subreg,
                        diff=row_diff,
                        description=row_desc,
                        Index=row_idx,
                    )
                    writes = dispatch(fast_row)
                    if writes:
                        f_writes.extend(writes)
                    last_emitted_reg = reg
                    last_emitted_nib = subreg
                    continue
                # Full-byte SET on a SUBREG_REG: split based on *actual* prior
                # value. state may be None (no FRAME_REG); fall back to 0.
                cur = row_val
                prev = int(state.last_val[reg]) if state is not None else 0
                cur_lo = cur & 0x0F
                cur_hi = (cur & 0xF0) >> 4
                prev_lo = prev & 0x0F
                prev_hi = (prev & 0xF0) >> 4
                lo_changed = cur_lo != prev_lo
                hi_changed = cur_hi != prev_hi

                # Conditional split:
                #   - one nibble changed -> emit one subreg row (same row
                #     count as a full-byte SET, smaller-alphabet vocab)
                #   - both nibbles changed -> keep full-byte SET (avoid
                #     the 2x split + the SUBREG_FLUSH that follows when
                #     the next write on this reg starts on the opposite
                #     nibble; on ctrl-heavy songs the always-split design
                #     blew up to 36.9% of total tokens via FLUSH alone)
                #   - neither changed -> redundant SET, leave untouched
                emitted_subregs = []
                if lo_changed and not hi_changed:
                    emitted_subregs.append((0, cur_lo))
                elif hi_changed and not lo_changed:
                    emitted_subregs.append((1, cur_hi))
                # both-nibble change: emitted_subregs stays empty so the
                # original SET row passes through untouched below.

                if not emitted_subregs:
                    # Either redundant SET or both-nibble change kept as
                    # full-byte SET. In both cases the existing row stays
                    # in df; advance state so subsequent rows see the
                    # post-write byte value.
                    if state is not None:
                        state.last_val[reg] = cur
                        f_writes.append((reg, cur, state.diff_for(reg), row_desc))
                    last_emitted_reg = None
                    last_emitted_nib = None
                    continue

                drop_idx.append(row_idx)

                if (
                    last_emitted_reg == reg
                    and emitted_subregs[0][0] != last_emitted_nib
                ):
                    new_rows.append(
                        {
                            "reg": reg,
                            "val": 0,
                            "diff": row_diff,
                            "op": int(SUBREG_FLUSH_OP),
                            "subreg": -1,
                            "__pos": row_idx,
                        }
                    )

                for subr, sval in emitted_subregs:
                    new_rows.append(
                        {
                            "reg": reg,
                            "val": int(sval),
                            "diff": row_diff,
                            "op": int(SET_OP),
                            "subreg": subr,
                            "__pos": row_idx,
                        }
                    )

                if state is not None:
                    # Apply the byte to state.last_val so the next SubregPass
                    # decision (and any GATE_REPLAY palette comparison) sees
                    # the post-SET register value.
                    state.last_val[reg] = cur
                    f_writes.append((reg, cur, state.diff_for(reg), row_desc))
                last_emitted_reg = reg
                last_emitted_nib = emitted_subregs[-1][0]
            if state is not None:
                f_writes.extend(state.tick_frame())
                state.observe_frame(f_writes, frame_idx=cur_frame)

        return _splice_rows(df, drop_idx, new_rows)


class EndTerminatorPass(MacroPass):
    """Insert explicit END_REPEAT/END_FLIP rows when a REPEAT/FLIP run is
    interrupted by a SET on the same reg (rather than terminated with val=0).

    Tokens are predictability markers for the LM; decoder ignores them.
    """

    def apply(self, df, args=None, palettes=None):
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        active_repeat = {}  # reg -> True
        active_flip = {}
        new_rows = []
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        ops = df["op"].to_numpy()
        diffs = df["diff"].to_numpy()
        indices = df.index.to_numpy()
        n = len(df)
        for i in range(n):
            reg = int(regs[i])
            if reg < 0:
                continue
            op = int(ops[i])
            val = int(vals[i])
            if op == REPEAT_OP:
                if val == 0:
                    active_repeat.pop(reg, None)
                else:
                    active_repeat[reg] = True
            elif op == FLIP_OP:
                if val == 0:
                    active_flip.pop(reg, None)
                else:
                    active_flip[reg] = True
            elif op == SET_OP:
                row_diff = int(diffs[i])
                row_idx = int(indices[i])
                if active_repeat.pop(reg, None):
                    new_rows.append(
                        {
                            "reg": reg,
                            "val": 0,
                            "diff": row_diff,
                            "op": int(END_REPEAT_OP),
                            "subreg": -1,
                            "__pos": row_idx,
                        }
                    )
                if active_flip.pop(reg, None):
                    new_rows.append(
                        {
                            "reg": reg,
                            "val": 0,
                            "diff": row_diff,
                            "op": int(END_FLIP_OP),
                            "subreg": -1,
                            "__pos": row_idx,
                        }
                    )
        return _splice_rows(df, [], new_rows)


# ---------------------------------------------------------------------------
# Loop-back encoding (LZ77 + structured DO/LOOP, frame-aligned)
# ---------------------------------------------------------------------------
def _slice_into_frames(df):
    """Return a list of (start_row_idx, end_row_idx) per frame. Frame
    boundaries are FRAME_REG / DELAY_REG rows -- those rows belong to the
    frame they start. The final frame extends to the end of df."""
    starts = df.index[df["reg"].isin(_FRAME_MARKER_REGS)].tolist()
    if not starts:
        return []
    ends = starts[1:] + [len(df)]
    return list(zip(starts, ends))


def _frame_content(df, start, end):
    """Hashable, comparable content tuple for a frame -- ignores diff and irq
    columns so that sequential identical-content frames at different stream
    times still match.

    Per-call pandas indexing is the dominant cost in ``LoopPass`` for any
    real song; ``_frame_contents_batch`` below builds all frames' tuples
    in one numpy pass and is what ``LoopPass.apply`` actually uses.
    """
    cols = ["reg", "val", "op", "subreg"]
    return tuple(tuple(int(v) for v in df.iloc[r][cols]) for r in range(start, end))


def _frame_contents_batch(df, frames):
    """Vectorised version of ``_frame_content`` for a whole frame list.

    Extracts the four content columns as numpy arrays once, then slices
    per frame -- avoids 1000s of per-row pandas indexing calls (the
    ``df.iloc[r][cols]`` pattern is ~3ms per row, dominating LoopPass).
    """
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    ops = df["op"].to_numpy()
    if "subreg" in df.columns:
        subregs = df["subreg"].to_numpy()
    else:
        subregs = np.full(len(df), -1, dtype=np.int64)
    out = []
    for s, e in frames:
        out.append(
            tuple(
                zip(
                    regs[s:e].tolist(),
                    vals[s:e].tolist(),
                    ops[s:e].tolist(),
                    subregs[s:e].tolist(),
                )
            )
        )
    return out


# Freq regs (pre-``_add_voice_reg``: absolute per-voice reg numbers
# 0, 7, 14). ``LoopPass`` uses this set to detect transposed pattern
# repeats: same body where every freq SET val differs from the source
# by a uniform delta (one semitone shift = a fixed quantized-note-index
# offset).
_FREQ_REGS_VOICED = frozenset(FREQ_REGS_BY_VOICE)


def _frame_stripped_contents_batch(df, frames):
    """Like ``_frame_contents_batch`` but freq SET vals are replaced by a
    placeholder so that stripped content matches across transpositions.
    Returned alongside per-frame freq-SET position lists for the
    transposed-match step.
    """
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    ops = df["op"].to_numpy()
    if "subreg" in df.columns:
        subregs = df["subreg"].to_numpy()
    else:
        subregs = np.full(len(df), -1, dtype=np.int64)
    is_freq_set = (
        np.isin(regs, list(_FREQ_REGS_VOICED)) & (ops == SET_OP) & (subregs == -1)
    )
    stripped = []
    for s, e in frames:
        rs = regs[s:e].tolist()
        vs = vals[s:e].tolist()
        os = ops[s:e].tolist()
        ss = subregs[s:e].tolist()
        is_fs = is_freq_set[s:e]
        # placeholder val 0 for freq SETs -- vals[i] is dropped.
        stripped.append(
            tuple((rs[k], 0 if is_fs[k] else vs[k], os[k], ss[k]) for k in range(e - s))
        )
    return stripped


try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:  # pragma: no cover  # numba is in requirements.txt;
    # this fallback is for bare-environment imports (jetson has its
    # own requirements file).
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        # Fallback: identity decorator. Slower but correct.
        def _inner(f):
            return f

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _inner


@njit(cache=True)
def _best_lz_njit(
    i,
    cands_arr,
    frame_data,
    frame_starts,
    frame_lens,
    sizes_cumsum,
    max_lz_length,
    n_frames,
    min_lz_match,
    ref_cost,
):
    """Inner loop of LoopPass.best_lz, JIT-compiled.

    Frame data is laid out as a flat (n_total_rows, 4) int64 array with
    ``frame_starts`` / ``frame_lens`` indexing each frame's slice. The
    seed lookup remains in Python; this function takes the candidate
    list (as int64 ndarray) and walks the inner match loop.
    """
    best_save = 0
    best_dist = 0
    best_len = 0
    n_cands = cands_arr.shape[0]
    for c_idx in range(n_cands - 1, -1, -1):
        cand = cands_arr[c_idx]
        if cand >= i:
            continue
        length = 0
        while length < max_lz_length and i + length < n_frames and cand + length < i:
            ai = i + length
            ac = cand + length
            la = frame_lens[ai]
            lc = frame_lens[ac]
            if la != lc:
                break
            sa = frame_starts[ai]
            sc = frame_starts[ac]
            ok = True
            for k in range(la):
                if (
                    frame_data[sa + k, 0] != frame_data[sc + k, 0]
                    or frame_data[sa + k, 1] != frame_data[sc + k, 1]
                    or frame_data[sa + k, 2] != frame_data[sc + k, 2]
                    or frame_data[sa + k, 3] != frame_data[sc + k, 3]
                ):
                    ok = False
                    break
            if not ok:
                break
            length += 1
        if length < min_lz_match:
            continue
        body_rows = sizes_cumsum[i + length] - sizes_cumsum[i]
        save = body_rows - ref_cost
        if save > best_save:
            best_save = save
            best_dist = i - cand
            best_len = length
    return best_save, best_dist, best_len


@njit(cache=True)
def _best_lz_fuzzy_match_njit(
    i,
    cands_arr,
    fps_arr,
    snap_arr,
    sizes_cumsum,
    max_fuzzy_length,
    n_frames,
    min_fuzzy_match,
    ref_cost,
):
    """Inner walk + overlay-count for LoopPass.best_lz_fuzzy.

    Returns ``(save, dist, length)`` for the best candidate -- save is
    computed as ``body_rows - (ref_cost + n_overlay_diffs)`` where
    ``n_overlay_diffs`` is the count of cells where ``snap_arr`` differs
    between the candidate and target ranges. The Python wrapper then
    calls ``compute_overlays`` ONCE on the winner to materialise the
    overlay list, instead of paying that cost per candidate.
    """
    best_save = 0
    best_dist = 0
    best_len = 0
    n_cands = cands_arr.shape[0]
    n_regs = snap_arr.shape[1]
    for c_idx in range(n_cands - 1, -1, -1):
        cand = cands_arr[c_idx]
        if cand >= i:
            continue
        length = 0
        while (
            length < max_fuzzy_length
            and i + length < n_frames
            and cand + length < i
            and fps_arr[i + length] == fps_arr[cand + length]
        ):
            length += 1
        if length < min_fuzzy_match:
            continue
        # Count overlay diffs inline on the snap_arr slices.
        n_overlays = 0
        for k in range(length):
            for r in range(n_regs):
                if snap_arr[i + k, r] != snap_arr[cand + k, r]:
                    n_overlays += 1
        body_rows = sizes_cumsum[i + length] - sizes_cumsum[i]
        save = body_rows - (ref_cost + n_overlays)
        if save > best_save:
            best_save = save
            best_dist = i - cand
            best_len = length
    return best_save, best_dist, best_len


@njit(cache=True)
def _best_lz_transposed_njit(
    i,
    cands_arr,
    frame_data,
    frame_starts,
    frame_lens,
    freq_set_mask,
    sizes_cumsum,
    max_lz_length,
    n_frames,
    min_lz_match,
    ref_cost,
):
    """Inner loop of LoopPass.best_lz_transposed, JIT-compiled.

    ``freq_set_mask`` is a flat bool array (n_total_rows,) marking rows
    whose ``(reg, op, subreg)`` is a freq-SET row -- those participate
    in the uniform-delta check, all other rows must match exactly.
    Returns ``(save, dist, length, delta)``.
    """
    best_save = 0
    best_dist = 0
    best_len = 0
    best_delta = 0
    n_cands = cands_arr.shape[0]
    for c_idx in range(n_cands - 1, -1, -1):
        cand = cands_arr[c_idx]
        if cand >= i:
            continue
        length = 0
        delta = 0
        delta_set = False
        while length < max_lz_length and i + length < n_frames and cand + length < i:
            ai = i + length
            ac = cand + length
            la = frame_lens[ai]
            lc = frame_lens[ac]
            if la != lc:
                break
            sa = frame_starts[ai]
            sc = frame_starts[ac]
            ok = True
            for k in range(la):
                fa = freq_set_mask[sa + k]
                fc = freq_set_mask[sc + k]
                if fa != fc:
                    ok = False
                    break
                if fa:
                    if (
                        frame_data[sa + k, 0] != frame_data[sc + k, 0]
                        or frame_data[sa + k, 2] != frame_data[sc + k, 2]
                        or frame_data[sa + k, 3] != frame_data[sc + k, 3]
                    ):
                        ok = False
                        break
                    d = frame_data[sa + k, 1] - frame_data[sc + k, 1]
                    if not delta_set:
                        delta = d
                        delta_set = True
                    elif d != delta:
                        ok = False
                        break
                else:
                    if (
                        frame_data[sa + k, 0] != frame_data[sc + k, 0]
                        or frame_data[sa + k, 1] != frame_data[sc + k, 1]
                        or frame_data[sa + k, 2] != frame_data[sc + k, 2]
                        or frame_data[sa + k, 3] != frame_data[sc + k, 3]
                    ):
                        ok = False
                        break
            if not ok:
                break
            length += 1
        if length < min_lz_match or not delta_set or delta == 0:
            continue
        body_rows = sizes_cumsum[i + length] - sizes_cumsum[i]
        save = body_rows - (ref_cost + 1)
        if save > best_save:
            best_save = save
            best_dist = i - cand
            best_len = length
            best_delta = delta
    return best_save, best_dist, best_len, best_delta


class LoopPass(MacroPass):
    """Hybrid encoder for repeated frame sequences.

    For each frame position, evaluates two candidate compressions and picks
    the cheaper one (or a literal if neither pays back):

      * **DO_LOOP**: longest run of M consecutive identical frame-groups
        starting here. Save = ``(N - 1) * body_rows - 2`` (BEGIN + END
        wrappers cost 2 tokens; body emitted once).

      * **BACK_REF**: longest match against any earlier position in the
        encoded stream. Save = ``body_rows - 1`` (replaces body_rows with
        one back-ref token).

    Greedy with one-frame lazy lookahead -- if i+1 has a meaningfully better
    match, emit a literal at i and let i+1 take its match. Per-rotation
    seed table; not retained across rotations.
    """

    min_lz_match = 2
    min_do_repeat = 2
    max_lz_length = 64
    max_do_body = 32
    max_do_repeat = 255
    ref_cost = 1  # one BACK_REF token
    do_wrap_cost = 2  # BEGIN + END

    def apply(self, df, args=None, palettes=None):
        if args is not None and not getattr(args, "loop_pass", True):
            return df
        loop_transposed = (
            getattr(args, "loop_transposed", True) if args is not None else True
        )
        fuzzy_loop = (
            getattr(args, "fuzzy_loop_pass", False) if args is not None else False
        )
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        frames = _slice_into_frames(df)
        n_frames = len(frames)
        if n_frames < self.min_lz_match:
            return df
        contents = _frame_contents_batch(df, frames)
        stripped = (
            _frame_stripped_contents_batch(df, frames) if loop_transposed else None
        )
        sizes = [e - s for s, e in frames]
        # ``sizes_cumsum[i]`` = sum(sizes[:i]). Lets every body_rows
        # computation be O(1) instead of an O(length) Python sum genexpr
        # called inside best_lz / best_lz_transposed / best_lz_fuzzy /
        # best_do (137M iterations on Techno_Aha before this).
        sizes_cumsum = np.zeros(n_frames + 1, dtype=np.int64)
        if sizes:
            sizes_cumsum[1:] = np.cumsum(np.asarray(sizes, dtype=np.int64))
        # Flat (n_total_rows, 4) int64 array of (reg, val, op, subreg)
        # plus per-frame start/len arrays. The numba inner loops below
        # take these instead of the per-frame Python tuples; equality
        # over numpy arrays in JIT'd code is much faster than Python
        # tuple compare on small frames.
        regs_full = df["reg"].to_numpy()
        vals_full = df["val"].to_numpy()
        ops_full = df["op"].to_numpy()
        subregs_full = (
            df["subreg"].to_numpy()
            if "subreg" in df.columns
            else np.full(len(df), -1, dtype=np.int64)
        )
        frame_data = np.column_stack(
            (
                regs_full.astype(np.int64),
                vals_full.astype(np.int64),
                ops_full.astype(np.int64),
                subregs_full.astype(np.int64),
            )
        )
        frame_starts_arr = np.asarray([s for s, _ in frames], dtype=np.int64)
        frame_lens_arr = np.asarray(sizes, dtype=np.int64)
        freq_set_mask = (
            np.isin(regs_full, list(_FREQ_REGS_VOICED))
            & (ops_full == SET_OP)
            & (subregs_full == -1)
        )

        # Fuzzy match needs per-frame musical fingerprints (so musically
        # equivalent frames hash the same despite byte-level state drift)
        # plus per-frame end-of-frame state snapshots (so the encoder can
        # compute exact byte-level overlay diffs between source and
        # target bodies). Both come from a single ``_per_frame_state_walk``
        # of the input df. Empty when fuzzy is off, which keeps the cost
        # off the default LoopPass-only path.
        fps = []
        snapshots = []
        if fuzzy_loop:
            try:
                fps, snapshots = _per_frame_state_walk(df)
            except Exception:
                fps, snapshots = [], []
            if len(fps) != n_frames:
                fps, snapshots = [], []

        # LZ77 seed tables. ``seed`` keys on exact byte content;
        # ``seed_stripped`` on freq-stripped content (for transposed
        # matching); ``seed_fp`` on musical fingerprint pairs (for fuzzy
        # matching). Each emit (literal / back-ref / pattern-replay /
        # do-loop) populates the seeds for the frames it covers so
        # subsequent positions can match against earlier emissions.
        seed = defaultdict(list)
        seed_stripped = defaultdict(list)
        seed_fp = defaultdict(list)
        out_rows = []
        sample_row = df.iloc[0]  # used to seed dtypes when constructing macro rows
        diff_default = int(sample_row["diff"]) if "diff" in df.columns else 0
        irq_default = int(df["irq"].iloc[0]) if "irq" in df.columns else -1
        # Pre-extract all rows as plain dicts once. Per-emission slicing
        # via ``all_records[s:e]`` avoids the 6000-call/song
        # ``df.iloc[r].to_dict()`` pattern (each call ~5ms in pandas).
        all_records = df.to_dict("records")
        snapshot_regs = sorted(snapshots[0].keys()) if snapshots else []
        # Materialise snapshots into a (n_frames, n_snapshot_regs) ndarray
        # so ``compute_overlays`` is a numpy slice + nonzero instead of
        # ``2 * len(snapshot_regs) * length`` dict.get's per call. With
        # snapshot_regs ~ 24 and ~1.15M overlay computations on the
        # largest training song, this drops 552M dict.get's (30s self).
        snapshot_regs_arr = np.asarray(snapshot_regs, dtype=np.int64)
        if snapshots:
            snap_arr = np.zeros((len(snapshots), len(snapshot_regs)), dtype=np.int64)
            for fi, snap in enumerate(snapshots):
                for ri, r in enumerate(snapshot_regs):
                    snap_arr[fi, ri] = int(snap.get(r, 0))
        else:
            snap_arr = np.zeros((0, 0), dtype=np.int64)
        # ``fps`` is a list of 64-bit musical fingerprints; numba's
        # JIT'd matcher takes them as an ndarray.
        fps_arr = (
            np.asarray(fps, dtype=np.int64) if fps else np.zeros(0, dtype=np.int64)
        )

        def best_do(i):
            best_save = 0
            best_body = 0
            best_n = 0
            for body_len in range(1, min(self.max_do_body, (n_frames - i) // 2) + 1):
                n = 1
                j = i + body_len
                while (
                    j + body_len <= n_frames
                    and n < self.max_do_repeat
                    and contents[i : i + body_len] == contents[j : j + body_len]
                ):
                    n += 1
                    j += body_len
                if n < self.min_do_repeat:
                    continue
                body_rows = int(sizes_cumsum[i + body_len] - sizes_cumsum[i])
                save = (n - 1) * body_rows - self.do_wrap_cost
                if save > best_save:
                    best_save, best_body, best_n = save, body_len, n
            return best_save, best_body, best_n

        def best_lz(i):
            if i + 1 >= n_frames:
                return 0, 0, 0
            cands = seed.get((contents[i], contents[i + 1]))
            if not cands:
                return 0, 0, 0
            cands_arr = np.asarray(cands, dtype=np.int64)
            return _best_lz_njit(
                i,
                cands_arr,
                frame_data,
                frame_starts_arr,
                frame_lens_arr,
                sizes_cumsum,
                self.max_lz_length,
                n_frames,
                self.min_lz_match,
                self.ref_cost,
            )

        def best_lz_transposed(i):
            """Like ``best_lz`` but matches frames whose freq SET vals
            differ from the source by a uniform delta. Returns
            ``(save, dist, length, delta)``. delta=0 implies exact match
            -- defer to ``best_lz`` for that case."""
            if i + 1 >= n_frames:
                return 0, 0, 0, 0
            cands = seed_stripped.get((stripped[i], stripped[i + 1]))
            if not cands:
                return 0, 0, 0, 0
            cands_arr = np.asarray(cands, dtype=np.int64)
            return _best_lz_transposed_njit(
                i,
                cands_arr,
                frame_data,
                frame_starts_arr,
                frame_lens_arr,
                freq_set_mask,
                sizes_cumsum,
                self.max_lz_length,
                n_frames,
                self.min_lz_match,
                self.ref_cost,
            )

        def _seed_pair(i):
            seed[(contents[i], contents[i + 1])].append(i)
            if loop_transposed:
                seed_stripped[(stripped[i], stripped[i + 1])].append(i)
            if fuzzy_loop and fps:
                seed_fp[(fps[i], fps[i + 1])].append(i)

        def emit_literal(i):
            s, e = frames[i]
            out_rows.extend(all_records[s:e])
            if i + 1 < n_frames:
                _seed_pair(i)

        def emit_back_ref(i, dist, length):
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": int(_pack_back_ref(dist, length)),
                    "diff": diff_default,
                    "op": int(BACK_REF_OP),
                    "subreg": -1,
                    "irq": irq_default,
                    "description": 0,
                }
            )
            for k in range(length):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        def emit_back_ref_transposed(i, dist, length, delta):
            # Subsumed: emitted as a PATTERN_REPLAY_OP with one
            # body-wide-mode PATTERN_OVERLAY_OP carrying the freq delta.
            # The body-wide mode is signalled by ``subreg=-1`` and a
            # sentinel ``target_reg=OVERLAY_BODY_FREQ_DELTA`` packed
            # into val's high bits. The decoder copies frames
            # cand..cand+length-1 and adds delta to every freq SET val.
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": int(_pack_back_ref(dist, length)),
                    "diff": diff_default,
                    "op": int(PATTERN_REPLAY_OP),
                    "subreg": 1,  # one overlay row follows
                    "irq": irq_default,
                    "description": 0,
                }
            )
            packed = (OVERLAY_BODY_FREQ_DELTA << 16) | (int(delta) & 0xFFFF)
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": int(packed),
                    "diff": diff_default,
                    "op": int(PATTERN_OVERLAY_OP),
                    "subreg": -1,  # body-wide mode marker
                    "irq": irq_default,
                    "description": 0,
                }
            )
            for k in range(length):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        def emit_do_loop(i, body, n):
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": int(n),
                    "diff": diff_default,
                    "op": int(DO_LOOP_OP),
                    "subreg": 0,
                    "irq": irq_default,
                    "description": 0,
                }
            )
            for k in range(body):
                s, e = frames[i + k]
                out_rows.extend(all_records[s:e])
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": 0,
                    "diff": diff_default,
                    "op": int(DO_LOOP_OP),
                    "subreg": 1,
                    "irq": irq_default,
                    "description": 0,
                }
            )
            covered = body * n
            for k in range(covered):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        def compute_overlays(src_idx, dst_idx, length):
            # Vectorised diff of two (length, n_snapshot_regs) views; one
            # numpy compare instead of length * n_snapshot_regs * 2 dict
            # lookups. ``np.nonzero`` returns the (k, ridx) pairs where
            # the snapshots differ; map ridx → reg via snapshot_regs_arr.
            src_view = snap_arr[src_idx : src_idx + length]
            dst_view = snap_arr[dst_idx : dst_idx + length]
            diff_mask = src_view != dst_view
            ks, ridxs = np.nonzero(diff_mask)
            if ks.size == 0:
                return []
            regs_out = snapshot_regs_arr[ridxs]
            vals_out = dst_view[ks, ridxs]
            return list(zip(ks.tolist(), regs_out.tolist(), vals_out.tolist()))

        # Fuzzy match: same byte cost as BACK_REF (1 token) plus one
        # PATTERN_OVERLAY_OP per differing reg per frame. Only beats
        # ``best_lz`` / ``best_lz_transposed`` when those miss the match
        # entirely (overlay count > 0 means byte-level divergence the
        # exact / freq-shift matchers can't handle).
        max_fuzzy_length = 16
        min_fuzzy_match = 2

        def best_lz_fuzzy(i):
            if not fuzzy_loop or not fps:
                return 0, 0, 0, []
            if i + 1 >= n_frames:
                return 0, 0, 0, []
            cands = seed_fp.get((fps[i], fps[i + 1]))
            if not cands:
                return 0, 0, 0, []
            cands_arr = np.asarray(cands, dtype=np.int64)
            # Pick the best candidate via numba (counts overlay diffs
            # via snap_arr without materialising the overlay tuple list).
            save, dist, length = _best_lz_fuzzy_match_njit(
                i,
                cands_arr,
                fps_arr,
                snap_arr,
                sizes_cumsum,
                max_fuzzy_length,
                n_frames,
                min_fuzzy_match,
                self.ref_cost,
            )
            if save <= 0:
                return 0, 0, 0, []
            # Materialise the overlay list ONCE for the winning match.
            overlays = compute_overlays(i - dist, i, length)
            return save, dist, length, overlays

        def emit_pattern_replay(i, dist, length, overlays):
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": int(_pack_back_ref(dist, length)),
                    "diff": diff_default,
                    "op": int(PATTERN_REPLAY_OP),
                    "subreg": int(len(overlays)),
                    "irq": irq_default,
                    "description": 0,
                }
            )
            for frame_offset, reg, val in overlays:
                packed = (int(reg) << 16) | (int(val) & 0xFFFF)
                out_rows.append(
                    {
                        "reg": int(LOOP_OP_REG),
                        "val": int(packed),
                        "diff": diff_default,
                        "op": int(PATTERN_OVERLAY_OP),
                        "subreg": int(frame_offset),
                        "irq": irq_default,
                        "description": 0,
                    }
                )
            for k in range(length):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        i = 0
        while i < n_frames:
            do_save, do_body, do_n = best_do(i)
            lz_save, lz_dist, lz_len = best_lz(i)
            if loop_transposed:
                tr_save, tr_dist, tr_len, tr_delta = best_lz_transposed(i)
            else:
                tr_save = tr_dist = tr_len = tr_delta = 0
            fz_save, fz_dist, fz_len, fz_overlays = best_lz_fuzzy(i)
            best_now = max(do_save, lz_save, tr_save, fz_save)
            if best_now > 0 and i + 1 < n_frames:
                la_do, _, _ = best_do(i + 1)
                la_lz, _, _ = best_lz(i + 1)
                if loop_transposed:
                    la_tr, _, _, _ = best_lz_transposed(i + 1)
                else:
                    la_tr = 0
                la_fz, _, _, _ = best_lz_fuzzy(i + 1)
                if max(la_do, la_lz, la_tr, la_fz) > best_now + 2:
                    emit_literal(i)
                    i += 1
                    continue
            if (
                do_save > 0
                and do_save >= lz_save
                and do_save >= tr_save
                and do_save >= fz_save
            ):
                emit_do_loop(i, do_body, do_n)
                i += do_body * do_n
            elif lz_save > 0 and lz_save >= tr_save and lz_save >= fz_save:
                emit_back_ref(i, lz_dist, lz_len)
                i += lz_len
            elif tr_save > 0 and tr_save >= fz_save:
                emit_back_ref_transposed(i, tr_dist, tr_len, tr_delta)
                i += tr_len
            elif fz_save > 0:
                emit_pattern_replay(i, fz_dist, fz_len, fz_overlays)
                i += fz_len
            else:
                emit_literal(i)
                i += 1

        if not out_rows:
            return df
        # Build new df preserving original column dtypes.
        orig_dtypes = df.dtypes.to_dict()
        new_df = pd.DataFrame(out_rows)
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = 0 if col == "description" else -1
        new_df = new_df[list(df.columns)]
        for col, dt in orig_dtypes.items():
            try:
                new_df[col] = new_df[col].astype(dt)
            except (TypeError, ValueError):
                pass
        new_df = new_df.reset_index(drop=True)
        if df.attrs:
            new_df.attrs.update(df.attrs)
        return new_df


def _musical_fingerprint(state):
    """Compact 64-bit musical-state fingerprint from a ``DecodeState``.

    Strips per-frame transient state (PWM phase, AD/SR counter, vibrato
    intermediate freqs) and keeps only the fields that distinguish
    "musically distinct" frames: per-voice (active note bucket,
    waveform high nibble, gate bit) + global filter cutoff bucket +
    volume nibble. Two frames with the same fingerprint are candidates
    for fuzzy-loop matching even if their byte-level register writes
    differ (e.g., from instrument-state drift).
    """
    fp = 0
    for v in range(VOICES):
        base = v * VOICE_REG_SIZE
        ctrl = int(state.last_val.get(base + 4, 0))
        freq = int(state.last_val.get(base + 0, 0))
        gate = ctrl & 0x01
        wave = (ctrl & 0xF0) >> 4
        note = freq & 0xFF if gate else 0xFF
        fp = (fp << 13) | (note << 5) | (wave << 1) | gate
    cutoff = (int(state.last_val.get(22, 0)) >> 4) & 0x0F
    modevol = int(state.last_val.get(24, 0)) & 0x0F
    fp = (fp << 8) | (cutoff << 4) | modevol
    return fp


def _per_frame_state_walk(df):
    """Walk df via ``_simulate_palette``-style dispatch and capture, per
    logical frame, (fingerprint, state.last_val snapshot). The state
    snapshot is the dict of register → end-of-frame byte value, used by
    ``LoopPass``'s fuzzy matcher to compute state-level overlay diffs
    between a candidate source body and the target. Only voice-bound
    regs and a handful of global regs are snapshotted (to bound
    memory)."""
    state = _build_decode_state(df)
    if state is None:
        return [], []
    arrs, frame_starts = _df_arrays_and_frames(df)
    regs = arrs["reg"]
    vals = arrs["val"]
    ops = arrs["op"]
    subregs = arrs["subreg"]
    diffs = arrs["diff"]
    n_total = len(df)
    n_frames = len(frame_starts)
    description_default = 0
    fps = []
    snapshots = []  # one dict per frame: reg -> end-of-frame val
    out_frame_idx = 0
    snapshot_regs = list(range(VOICES * VOICE_REG_SIZE)) + [22, 23, 24]
    for fi in range(n_frames):
        start = int(frame_starts[fi])
        end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
        f_writes = []
        marker_reg = int(regs[start])
        marker_val = int(vals[start])
        marker_diff = int(diffs[start])
        if marker_reg == FRAME_REG:
            f_writes.append((marker_reg, marker_val, marker_diff, description_default))
        elif marker_reg == DELAY_REG:
            for _ in range(marker_val - 1):
                delay_writes = [(FRAME_REG, 0, state.frame_diff, description_default)]
                delay_writes.extend(state.tick_frame())
                state.observe_frame(delay_writes, frame_idx=out_frame_idx)
            f_writes.append((FRAME_REG, 0, state.frame_diff, description_default))
        for i in range(start + 1, end):
            reg = int(regs[i])
            if reg < 0:
                continue
            op = int(ops[i])
            # SET fast-path: the most common op (~90% of rows on a
            # typical block) and its decode is just ``last_val[reg] =
            # val`` plus a write tuple for downstream simulators. Skip
            # the _FastRow construction + DECODERS dispatch + the
            # subreg-flush bookkeeping (which is moot here since
            # _per_frame_state_walk doesn't produce subreg-split rows
            # -- they were already collapsed by SubregPass before
            # this walk runs).
            subreg = int(subregs[i])
            if op == SET_OP and subreg == -1:
                val = int(vals[i])
                state.last_val[reg] = val
                state.last_diff[reg] = int(diffs[i])
                f_writes.append((reg, val, int(diffs[i]), description_default))
                continue
            decoder = DECODERS.get(op)
            if decoder is None:
                continue
            row = _fastrow_from_arrs(arrs, i)
            writes = decoder.expand(row, state)
            if writes:
                f_writes.extend(writes)
        f_writes.extend(state.tick_frame())
        state.observe_frame(f_writes, frame_idx=out_frame_idx)
        out_frame_idx += 1
        fps.append(_musical_fingerprint(state))
        snapshots.append({r: int(state.last_val.get(r, 0)) for r in snapshot_regs})
    return fps, snapshots


# ---------------------------------------------------------------------------
# Loop-back expansion (decode-side pre-pass for _expand_ops)
# ---------------------------------------------------------------------------
def _is_frame_marker_row(row):
    return row[0] in _FRAME_MARKER_REGS  # row is a tuple-like with reg first


def expand_loops(df):
    """Materialize BACK_REF and DO_LOOP rows into literal frame copies.

    Run as a pre-pass to ``RegLogParser._expand_ops`` so per-row decoders
    never see loop ops. Maintains a stack of pending DO_LOOP iterations and
    a list of output frame start positions for back-ref slicing. Distances
    and lengths are interpreted in *logical frame slots* -- each FRAME_REG
    or DELAY_REG row in the OUTPUT counts as one slot.
    """
    if "op" not in df.columns:
        return df
    has_loops = df["op"].isin([BACK_REF_OP, DO_LOOP_OP, PATTERN_REPLAY_OP]).any()
    if not has_loops:
        return df

    cols = list(df.columns)
    out = []
    output_frame_starts = []  # row idx in `out` where each output frame begins
    do_stack = []  # list of [body_start_row_in_input, remaining_iterations]

    # Pre-extract every column as a numpy array once. ``df.iloc[i]`` would
    # otherwise be called ~30K times per song and each call goes through
    # pandas' ``__finalize__`` (deepcopying the masked-int dtype state),
    # eating ~9ms per row -- 4-5 minutes of preload per song. Walking the
    # raw arrays drops the loop to milliseconds.
    col_arrays = {c: df[c].to_numpy() for c in cols}
    op_arr = col_arrays["op"]
    val_arr = col_arrays["val"]
    subreg_arr = col_arrays["subreg"] if "subreg" in cols else None

    def _row_to_dict(i):
        # Build a dict from the prefetched arrays. ``pd.isna`` works on
        # numpy scalars too (handles the masked-int NA sentinel). Cheaper
        # than the original ``{c: row[c] for c in cols}`` because no
        # per-row Series construction.
        return {c: col_arrays[c][i] for c in cols}

    def append_row(row_dict):
        out.append(row_dict)
        if row_dict["reg"] in _FRAME_MARKER_REGS:
            output_frame_starts.append(len(out) - 1)

    n = len(df)
    i = 0
    while i < n:
        op_raw = op_arr[i]
        op = int(op_raw) if not pd.isna(op_raw) else SET_OP
        if op == BACK_REF_OP:
            distance, length = _unpack_back_ref(val_arr[i])
            cur_frame = len(output_frame_starts)
            target = cur_frame - distance
            assert target >= 0, (
                f"BACK_REF target frame {target} reaches before output start "
                f"(cur_frame={cur_frame}, distance={distance})"
            )
            assert target + length <= cur_frame, (
                f"BACK_REF target range [{target},{target+length}) overlaps "
                f"present frame {cur_frame}"
            )
            for f in range(target, target + length):
                src_lo = output_frame_starts[f]
                src_hi = (
                    output_frame_starts[f + 1]
                    if f + 1 < len(output_frame_starts)
                    else len(out)
                )
                snapshot = list(out[src_lo:src_hi])
                for snap_row in snapshot:
                    append_row(dict(snap_row))
            i += 1
            continue
        if op == DO_LOOP_OP:
            subreg_raw = subreg_arr[i] if subreg_arr is not None else -1
            subreg = int(subreg_raw) if not pd.isna(subreg_raw) else -1
            if subreg == 0:
                n_iter = int(val_arr[i])
                assert n_iter >= 1, n_iter
                # Push: record where the body starts (i+1) and remaining iters
                do_stack.append([i + 1, n_iter - 1])
                i += 1
                continue
            # subreg == 1: END marker
            if do_stack and do_stack[-1][1] > 0:
                body_start, remaining = do_stack[-1]
                do_stack[-1][1] = remaining - 1
                i = body_start
            else:
                if do_stack:
                    do_stack.pop()
                i += 1
            continue
        if op == PATTERN_REPLAY_OP:
            distance, length = _unpack_back_ref(val_arr[i])
            sr_raw = subreg_arr[i] if subreg_arr is not None else 0
            num_overlays = int(sr_raw) if not pd.isna(sr_raw) else 0
            cur_frame = len(output_frame_starts)
            target = cur_frame - distance
            assert target >= 0, (
                f"PATTERN_REPLAY target frame {target} reaches before output "
                f"start (cur_frame={cur_frame}, distance={distance})"
            )
            assert target + length <= cur_frame, (
                f"PATTERN_REPLAY target range [{target},{target+length}) "
                f"overlaps present frame {cur_frame}"
            )
            # Read the overlay block: ``num_overlays`` rows immediately
            # following the PATTERN_REPLAY_OP. Each overlay row carries
            # reg=LOOP_OP_REG (so _norm_pr_order keeps the block
            # contiguous), with target_reg and new_val packed into val:
            # ``val = (target_reg << 16) | (new_val & 0xFFFF)``.
            # ``subreg = frame_offset_in_body``.
            # Per-frame overlays (subreg >= 0): applied as SET rows to
            # the matching frame_offset in the body.
            # Body-wide overlays (subreg == -1): mode-encoded patches
            # applied to all body rows. Currently the only body-wide
            # mode is OVERLAY_BODY_FREQ_DELTA -- subsumes the old
            # BACK_REF_TRANSPOSED_OP. Mode is signalled by target_reg
            # = OVERLAY_BODY_FREQ_DELTA in the packed val.
            overlays = []  # list of (frame_offset, reg, val) per-frame
            body_freq_delta = 0
            for k in range(num_overlays):
                ov_idx = i + 1 + k
                ov_op_raw = op_arr[ov_idx]
                ov_op = int(ov_op_raw) if not pd.isna(ov_op_raw) else SET_OP
                assert ov_op == PATTERN_OVERLAY_OP, (
                    f"PATTERN_REPLAY at row {i} expected {num_overlays} "
                    f"overlay rows but row {ov_idx} has op={ov_op}"
                )
                packed = int(val_arr[ov_idx])
                target_reg = (packed >> 16) & 0xFF
                ov_sr_raw = subreg_arr[ov_idx] if subreg_arr is not None else 0
                ov_subreg = int(ov_sr_raw) if not pd.isna(ov_sr_raw) else 0
                if ov_subreg < 0 and target_reg == OVERLAY_BODY_FREQ_DELTA:
                    # Sign-extend the 16-bit delta.
                    delta = packed & 0xFFFF
                    if delta >= 0x8000:
                        delta -= 0x10000
                    body_freq_delta = delta
                    continue
                new_val = packed & 0xFFFF
                overlays.append((ov_subreg, target_reg, new_val))
            # Group overlays by frame_offset for fast lookup during
            # body replay.
            ov_by_frame = defaultdict(list)
            for fo, r, v in overlays:
                ov_by_frame[fo].append((r, v))
            # Replay source body, applying overlays per frame and the
            # body-wide freq delta to every freq SET in the body.
            for f in range(target, target + length):
                src_lo = output_frame_starts[f]
                src_hi = (
                    output_frame_starts[f + 1]
                    if f + 1 < len(output_frame_starts)
                    else len(out)
                )
                snapshot = list(out[src_lo:src_hi])
                for snap_row in snapshot:
                    new_row = dict(snap_row)
                    if (
                        body_freq_delta
                        and int(new_row.get("reg", -1)) in _FREQ_REGS_VOICED
                        and int(new_row.get("op", SET_OP)) == SET_OP
                        and int(new_row.get("subreg", -1)) == -1
                    ):
                        new_row["val"] = int(new_row["val"]) + body_freq_delta
                    append_row(new_row)
                # Per-frame SET overlays AFTER the body's writes.
                frame_offset = f - target
                for r, v in ov_by_frame.get(frame_offset, ()):
                    template = dict(snapshot[0]) if snapshot else {}
                    template.update(
                        {
                            "reg": int(r),
                            "val": int(v),
                            "op": int(SET_OP),
                            "subreg": -1,
                        }
                    )
                    append_row(template)
            i += 1 + num_overlays
            continue
        if op == PATTERN_OVERLAY_OP:
            # Should have been consumed by a preceding PATTERN_REPLAY_OP;
            # if reached as a top-level row the encoder is broken.
            raise AssertionError(f"orphan PATTERN_OVERLAY_OP at row {i}")
        # Literal row.
        append_row(_row_to_dict(i))
        i += 1

    if not out:
        return df.iloc[0:0]
    expanded = pd.DataFrame(out, columns=cols)
    # Restore dtypes.
    for col, dt in df.dtypes.items():
        try:
            expanded[col] = expanded[col].astype(dt)
        except (TypeError, ValueError):
            pass
    expanded = expanded.reset_index(drop=True)
    if df.attrs:
        expanded.attrs.update(df.attrs)
    return expanded


def _build_last_diff(df):
    """Per-reg first-SET diff lookup. Returns ``{int(reg): int(diff)}``
    with ``MIN_DIFF`` as the fallback when a reg has no SET row.

    Hoisted so the various ``DecodeState``-building call sites
    (``_build_decode_state`` here + the per-pass apply methods) share
    the same lookup logic; previously they each open-coded the same
    loop.
    """
    last_diff = {}
    for reg in df["reg"].unique():
        sub = df[(df["reg"] == reg) & (df["op"] == SET_OP)]["diff"]
        last_diff[int(reg)] = int(sub.iloc[0]) if len(sub) else MIN_DIFF
    return last_diff


def _build_decode_state(df, args=None, palettes=None, strict=False):
    """Construct a ``DecodeState`` seeded the same way ``_expand_ops`` does:
    ``frame_diff`` from the first FRAME_REG row, ``last_diff`` per reg
    from each reg's first SET. Returns ``None`` if there is no FRAME_REG.

    ``args`` (optional Namespace): picks up ``gate_palette_cap`` from
    ``args.gate_palette_cap`` if present.
    ``palettes`` (optional ``Palettes``): if provided, seeds the state
    with frozen instrument / gate palettes so downstream walkers
    initialise from the encoder's authoritative palette instead of
    growing their own by observation.
    """
    fr = df[df["reg"] == FRAME_REG]["diff"]
    if fr.empty:
        return None
    last_diff = _build_last_diff(df)
    cap = getattr(args, "gate_palette_cap", None) if args is not None else None
    frozen_instr = palettes.instrument_palette if palettes is not None else None
    frozen_gate = palettes.gate_palette if palettes is not None else None
    return DecodeState(
        int(fr.iloc[0]),
        last_diff=last_diff,
        strict=strict,
        gate_palette_cap=cap,
        frozen_instrument_palette=frozen_instr,
        frozen_gate_palette=frozen_gate,
    )


def _simulate_palette(literal_df):
    """Walk a loop-expanded df through ``DECODERS`` exactly the way
    ``_expand_ops`` does and return the populated ``DecodeState`` (with
    ``gate_palette`` and ``gate_palette_def_frames`` filled in).

    Used by ``validate_gate_replays`` to verify every GATE_REPLAY_OP
    resolves to a palette slot defined by the time the row is reached
    -- ``GateReplayDecoder.expand`` raises if it doesn't.
    """
    state = _build_decode_state(literal_df)
    if state is None:
        return None
    df = literal_df.copy().reset_index(drop=True)
    out_frame_idx = 0
    description_default = 0

    def finalize(writes, advance=True):
        nonlocal out_frame_idx
        state.observe_frame(writes, frame_idx=out_frame_idx)
        if advance:
            out_frame_idx += 1

    arrs, frame_starts = _df_arrays_and_frames(df)
    regs = arrs["reg"]
    vals = arrs["val"]
    ops = arrs["op"]
    subregs = arrs["subreg"]
    diffs = arrs["diff"]
    descs = arrs["description"]
    indices = arrs["Index"]
    n_total = len(df)
    n_frames = len(frame_starts)
    for fi in range(n_frames):
        start = int(frame_starts[fi])
        end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
        f_writes = []
        marker_reg = int(regs[start])
        marker_val = int(vals[start])
        marker_diff = int(diffs[start])
        if marker_reg == FRAME_REG:
            f_writes.append((marker_reg, marker_val, marker_diff, description_default))
        elif marker_reg == DELAY_REG:
            for _ in range(marker_val - 1):
                delay_writes = [(FRAME_REG, 0, state.frame_diff, description_default)]
                delay_writes.extend(state.tick_frame())
                finalize(delay_writes, advance=False)
            f_writes.append((FRAME_REG, 0, state.frame_diff, description_default))
        for i in range(start + 1, end):
            reg = int(regs[i])
            if reg < 0:
                continue
            op = int(ops[i])
            decoder = DECODERS.get(op)
            if decoder is None:
                continue
            row = _FastRow(
                reg=reg,
                val=int(vals[i]),
                op=op,
                subreg=int(subregs[i]),
                diff=int(diffs[i]),
                description=int(descs[i]),
                Index=int(indices[i]),
            )
            writes = decoder.expand(row, state)
            if writes:
                f_writes.extend(writes)
        f_writes.extend(state.tick_frame())
        finalize(f_writes)
    return state


def expand_to_literal_form(df, args=None):
    """Fully expand all macros in ``df`` to literal SET rows.

    Walks the encoded form via ``RegLogParser._expand_ops`` (which
    dispatches every decoder including loop/back-ref expansion, gate
    replay, instrument program playback, and tick-driven bursts) and
    returns the resulting per-frame SID-write stream as a DataFrame
    with columns ``reg``, ``val``, ``diff``, ``description``, ``op``,
    ``subreg``. Every non-marker row has ``op=SET_OP`` and ``subreg=-1``;
    FRAME_REG markers separate frames. The result has no GATE_REPLAY_OP
    / PLAY_INSTRUMENT_OP / BACK_REF_OP rows at all -- only literal SETs
    -- so any frame-aligned slice is trivially self-contained.

    Used by ``self_contain_slice`` and the block iterator to produce
    slices whose macro references resolve locally without a definition
    preamble: each slice is later re-encoded by ``run_passes`` into its
    own self-contained macro form.
    """
    # Local import to avoid a top-level circular dep with reglogparser.
    from preframr.reglogparser import RegLogParser

    parser = RegLogParser(args)
    df_in = df.copy()
    # ``_expand_ops`` reads row.description; tests / lightweight callers
    # may pass a df without that column. Default to 0 so the walker never
    # attribute-errors on a synthetic input.
    if "description" not in df_in.columns:
        df_in["description"] = 0
    literal = parser._expand_ops(df_in, strict=False)
    # ``_expand_ops`` returns columns ["reg","val","diff","description"];
    # add the op/subreg columns ``run_passes`` and the row tokenizer
    # expect, set to the literal-SET defaults.
    if "op" not in literal.columns:
        literal["op"] = int(SET_OP)
    else:
        literal["op"] = literal["op"].fillna(int(SET_OP)).astype(int)
    if "subreg" not in literal.columns:
        literal["subreg"] = -1
    return literal.reset_index(drop=True)


def self_contain_slice(df, slice_lo_frame, slice_hi_frame, args=None):
    """Materialise a single ``[slice_lo_frame, slice_hi_frame)`` slice
    into a self-contained row DataFrame.

    Strategy: fully expand the source df to literal SID writes via
    ``expand_to_literal_form``, slice at frame-marker coords, then
    optionally re-run ``run_passes`` so the slice gets its own
    self-contained macro encoding (palette indices local to the slice,
    BACK_REF distances pointing only inside the slice). The re-encode
    step is what guarantees no dangling references regardless of where
    the slice starts -- the encoder rebuilds palettes from slot 0 on
    the slice's own first occurrences, and any GATE_REPLAY_OP /
    PLAY_INSTRUMENT_OP it emits points at slots it just defined.

    Pass ``args=None`` to skip the re-encode and return the literal
    slice (used by callers that only want the unencoded SID-write
    form, e.g. the prompt's accuracy-comparison df).
    """
    literal = expand_to_literal_form(df, args=args)
    is_marker = literal["reg"].isin({FRAME_REG, DELAY_REG})
    marker_idx = literal.index[is_marker].tolist()
    if slice_lo_frame >= len(marker_idx):
        return literal.iloc[0:0].reset_index(drop=True).copy()
    row_lo = int(marker_idx[slice_lo_frame])
    row_hi = (
        int(marker_idx[slice_hi_frame])
        if slice_hi_frame < len(marker_idx)
        else len(literal)
    )
    slice_df = literal.iloc[row_lo:row_hi].reset_index(drop=True).copy()
    if args is None:
        return slice_df
    # Fresh Palettes per slice -- palette indices are slice-local by
    # construction (the encoder rebuilds them from slot 0 within the
    # slice), so the container is scoped to this run_passes call.
    return run_passes(slice_df, args=args, palettes=Palettes())


def iter_self_contained_row_blocks(df, frames_per_block, args=None, stride=None):
    """Yield row-DataFrames each covering ``frames_per_block`` logical
    frame slots (= FRAME_REG/DELAY_REG markers) of ``df``. Every block
    has its out-of-block references (BACK_REF_OP, GATE_REPLAY_OP,
    PLAY_INSTRUMENT_OP, DO_LOOP_OP) rewritten to literals so the block
    can be tokenized and decoded standalone.

    The shared row-stream walker for both the training and inference
    paths. The training data path materializes blocks at parse time and
    saves them in a 2D ``.blocks.npy`` so per-batch ``__getitem__`` is
    just a slice. The predict path takes the first (or chosen) block as
    the prompt.

    ``stride`` (in frame slots) controls how far apart adjacent block
    starts are. Default ``stride=frames_per_block`` produces the
    historical non-overlapping tiling. Smaller values yield overlapping
    blocks for data augmentation -- e.g.  ``stride=frames_per_block//4``
    gives 4x as many blocks per song, each starting at a different
    musical phase. Each block is independently self-contained, so
    overlap doesn't compromise correctness.

    Notes:
    - Blocks always start and end at logical frame boundaries.
    - The materialized block may be slightly longer (in row count) than
      a naive slice because materializations add literal SET rows for
      out-of-block palette / loop targets. Layer 2 (tokenizer wrap)
      handles trim-to-N.
    - Callers can pass a small ``frames_per_block`` and let Layer 2
      bin-search the right size, or pass a precomputed value.
    """
    if stride is None or stride < 1:
        stride = frames_per_block

    if "op" not in df.columns:
        # No macros to materialize -- chunk by frame markers.
        is_marker = df["reg"].isin({FRAME_REG, DELAY_REG})
        marker_idx = df.index[is_marker].tolist()
        if not marker_idx:
            yield df.reset_index(drop=True).copy()
            return
        n_frames = len(marker_idx)
        for lo in range(0, n_frames, stride):
            hi = min(lo + frames_per_block, n_frames)
            row_lo = marker_idx[lo]
            row_hi = marker_idx[hi] if hi < n_frames else len(df)
            yield df.iloc[row_lo:row_hi].reset_index(drop=True).copy()
        return

    is_marker = df["reg"].isin({FRAME_REG, DELAY_REG})
    marker_count = int(is_marker.sum())
    if marker_count == 0:
        yield df.reset_index(drop=True).copy()
        return

    # Hoisted out of the per-block loop. Calling ``self_contain_slice``
    # per block re-expanded the full song each time -- O(song x
    # num_blocks). Compute literal + marker_idx once, slice cheaply,
    # and ``run_passes`` only on each per-block slice.
    literal = expand_to_literal_form(df, args=args)
    # Clear any inherited ``df.attrs`` (palettes published by run_passes
    # upstream). The literal stream has no macros left to resolve, so
    # the palette is dead weight here -- but every subsequent ``.iloc``
    # / ``.copy`` of ``literal`` would otherwise deep-copy attrs via
    # pandas ``__finalize__``. Skybox parse profile showed ~30s self
    # in copy.deepcopy attributable to that storm.
    literal.attrs.clear()
    lit_is_marker = literal["reg"].isin({FRAME_REG, DELAY_REG})
    marker_idx = literal.index[lit_is_marker].tolist()
    n_lit_frames = len(marker_idx)

    # Local import to avoid the module-load circular dep with reglogparser.
    from preframr.reglogparser import RegLogParser

    consolidator = RegLogParser(args)
    for lo_frame in range(0, marker_count, stride):
        if lo_frame >= n_lit_frames:
            break
        hi_frame = min(lo_frame + frames_per_block, n_lit_frames)
        row_lo = int(marker_idx[lo_frame])
        row_hi = int(marker_idx[hi_frame]) if hi_frame < n_lit_frames else len(literal)
        slice_df = literal.iloc[row_lo:row_hi].reset_index(drop=True).copy()
        if slice_df.empty:
            continue
        # Fresh Palettes per block -- palette indices are block-local,
        # so the container's lifetime is the block's run_passes call.
        block = (
            run_passes(slice_df, args=args, palettes=Palettes())
            if args is not None
            else slice_df
        )
        if block.empty:
            continue
        # ``run_passes`` attaches the per-block palette to ``block.attrs``
        # (one-time write at end of run_passes). Downstream consumers
        # (``_add_voice_reg``, ``_consolidate_frames``, BlockMapper)
        # don't need it -- the block is fully encoded -- but every
        # subsequent pandas op would deep-copy the attrs via
        # ``__finalize__``. Clear here so the block ships with empty
        # attrs.
        block.attrs.clear()
        # Re-pack runs of FRAME_REG back into DELAY_REG so blocks use
        # the compact encoding. ``expand_to_literal_form`` (above)
        # destructured DELAY_REG val=N into N FRAME_REGs so per-frame
        # state simulation is straightforward; consolidation re-packs
        # the result. Without this step blocks would carry N tokens
        # where DELAY_REG val=N carries one, inflating token counts
        # by ~5-10x on quiet sections.
        try:
            block = consolidator._consolidate_frames(block)
        except Exception:
            pass
        yield block


def validate_gate_replays(df):
    """Walk ``df`` and verify every ``GATE_REPLAY_OP`` resolves to a
    palette slot defined by the time the row is reached. Returns ``True``
    on success; raises ``AssertionError`` from inside
    ``GateReplayDecoder`` (caller decides whether to catch)."""
    if "op" not in df.columns or not df["op"].isin([GATE_REPLAY_OP]).any():
        return True
    # Reuse the simulator -- if any GATE_REPLAY_OP references a slot that
    # observe_frame hasn't filled yet, GateReplayDecoder.expand asserts.
    _simulate_palette(expand_loops(df.copy()))
    return True


def validate_back_refs(df, prompt_frame_count=0):
    """Walk ``df`` and verify every BACK_REF resolves within bounds.

    ``prompt_frame_count`` is the number of frames already in the output
    buffer at df's start (e.g. for an LM-generated continuation appended
    after a prompt). Returns True if all back-refs are valid; raises
    AssertionError with the offending row index otherwise.
    """
    if "op" not in df.columns:
        return True
    output_frame_count = prompt_frame_count
    for idx, row in df.iterrows():
        op = int(row["op"]) if not pd.isna(row["op"]) else SET_OP
        if op == BACK_REF_OP:
            distance, length = _unpack_back_ref(row["val"])
            target = output_frame_count - distance
            assert target >= 0, (
                f"row {idx}: BACK_REF distance={distance} reaches before "
                f"frame 0 (output_frame_count={output_frame_count})"
            )
            output_frame_count += length
            continue
        if op == DO_LOOP_OP:
            # DO_LOOP body is self-contained; skip frame-counting until END.
            # validate_back_refs is conservative -- we only sanity-check
            # back-refs at the top level here.
            continue
        if row["reg"] in _FRAME_MARKER_REGS:
            output_frame_count += 1
    return True


class GateMacroPass(MacroPass):
    """Compress repeated ``(ctrl, AD, SR)`` end-of-frame state on a voice
    around a gate-bit transition into a single ``GATE_REPLAY_OP`` row.

    Per (voice, direction), maintains a palette of distinct
    ``(ctrl, AD, SR)`` byte triples. The first occurrence emits literal
    SET rows (which ``SubregPass`` further splits into nibbles). Each
    subsequent occurrence whose end-of-frame state matches an existing
    palette slot is replaced by one ``GATE_REPLAY_OP`` token.

    Runs **before** ``SubregPass`` so it operates on full-byte SETs (the
    only op that ever lands on ctrl/AD/SR -- DIFF/REPEAT/FLIP/PWM/etc.
    only target freq/PWM/FC_LO).
    """

    target_kinds = ("ctrl", "ad", "sr")

    def apply(self, df, args=None, palettes=None):
        if args is not None and not getattr(args, "gate_macro_pass", True):
            return df
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        if "description" not in df.columns:
            df["description"] = 0

        state = _build_decode_state(df, args=args)
        if state is None:
            return df

        arrs, frame_starts = _df_arrays_and_frames(df)
        regs_all = arrs["reg"]
        vals_all = arrs["val"]
        ops_all = arrs["op"]
        subregs_all = arrs["subreg"]
        diffs_all = arrs["diff"]
        descs_all = arrs["description"]
        indices_all = arrs["Index"]

        drop_idx = []
        new_rows = []
        n_total = len(df)
        n_frames = len(frame_starts)

        for fi in range(n_frames):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            cur_frame = fi
            voice_candidate_idx = defaultdict(list)
            f_writes = []
            for i in range(start, end):
                reg = int(regs_all[i])
                if reg < 0:
                    if reg == DELAY_REG:
                        for _ in range(int(vals_all[i]) - 1):
                            state.tick_frame()
                            state.observe_frame(
                                [], frame_idx=cur_frame, track_instruments=False
                            )
                    continue
                op = int(ops_all[i])
                subreg = int(subregs_all[i])
                if op == SET_OP and subreg == -1 and reg in _GATE_REG_TO_VOICE:
                    voice_candidate_idx[_GATE_REG_TO_VOICE[reg]].append(
                        int(indices_all[i])
                    )
                decoder = DECODERS.get(op)
                if decoder is not None:
                    row = _FastRow(
                        reg=reg,
                        val=int(vals_all[i]),
                        op=op,
                        subreg=subreg,
                        diff=int(diffs_all[i]),
                        description=int(descs_all[i]),
                        Index=int(indices_all[i]),
                    )
                    writes = decoder.expand(row, state)
                    if writes:
                        f_writes.extend(writes)
            f_writes.extend(state.tick_frame())
            # Drive observe_frame so its gate_palette evolves identically
            # to the downstream's. ``last_gate_transitions`` reports each
            # transition this frame and whether observe_frame added the
            # bundle (first occurrence) or matched an existing slot
            # (replay). track_instruments=False keeps state.instrument_*
            # untouched -- this pass doesn't care about them.
            state.observe_frame(f_writes, frame_idx=cur_frame, track_instruments=False)

            for v, d, _bundle, was_added, slot in state.last_gate_transitions:
                cand_rows = voice_candidate_idx.get(v, [])
                if was_added or slot is None:
                    # First occurrence (or over-cap): leave literal so
                    # observe_frame on the post-pass form rebuilds the
                    # same palette entry at the same slot.
                    continue
                if not cand_rows:
                    # Transition without a SET candidate row this frame
                    # (e.g., ctrl write came from a tick_frame burst on a
                    # reg that ISN'T ctrl -- shouldn't happen in practice
                    # since ctrl regs aren't tick_frame targets, but be
                    # defensive). No rows to drop -> leave literal.
                    continue
                ctrl_reg = v * VOICE_REG_SIZE + 4
                drop_idx.extend(cand_rows)
                new_rows.append(
                    {
                        "reg": int(ctrl_reg),
                        "val": int(slot),
                        "diff": int(state.diff_for(ctrl_reg)),
                        "op": int(GATE_REPLAY_OP),
                        "subreg": int(d),
                        "__pos": int(min(cand_rows)),
                    }
                )

        out = _splice_rows(df, drop_idx, new_rows)
        # Publish the authoritative gate palette so downstream walkers
        # (DedupSet, Subreg, FuzzyLoopPass body replay, _expand_ops,
        # find_redundant_writes) initialise from it instead of growing
        # via observation -- the latter diverges when intermediate
        # passes change the order/content of gate-bundle writes.
        # Stored on the explicit Palettes container threaded through
        # ``run_passes``; previously rode ``df.attrs["gate_palette"]``
        # but pandas ``__finalize__`` deep-copied the whole structure
        # on every dataframe op (~30% of Skybox parse).
        if palettes is not None:
            palettes.gate_palette = {
                k: [tuple(b) for b in v] for k, v in state.gate_palette.items()
            }
        return out


class InstrumentProgramPass(MacroPass):
    """Capture multi-frame instrument programs at gate-on transitions and
    collapse identical programs into one ``PLAY_INSTRUMENT_OP`` token.

    Runs after ``GateMacroPass`` (so v1's ``GATE_REPLAY_OP`` rows are
    visible as candidates) and before ``SubregPass`` (so candidate rows
    are unsplit). Per-voice register writes inside the capture window
    become the program; writes from multi-voice ops (``TRANSPOSE_OP``)
    abort the macro for that voice's window because we can't drop a row
    without losing other voices' contributions.

    **Design D (deferred-decision single walk)**: a two-phase pass that
    avoids the slot-mismatch issues of designs A/B/C. Phase 1 walks the
    original df and *records* candidate captures (voice, frame range,
    program tuple, drop_rows); no replay decisions yet. Phase 2 walks
    the original df again with a fresh DecodeState and ``observe_frame
    (track_instruments=True)``; at each candidate's start_frame the
    encoder checks ``state.instrument_palette`` for the candidate's
    program — if present, REPLAY (mark drop_rows, emit + dispatch a
    PLAY_INSTRUMENT_OP); if absent, LITERAL (don't drop, let the rows
    dispatch normally). Because Phase 2 dispatches the same rows the
    downstream eventually walks (modulo the drops + emitted
    PLAY_INSTRUMENT_OP rows it dispatches itself), the palette state
    Phase 2 builds matches the downstream's exactly, so slot indices
    line up by construction.
    """

    _voice_confined_ops = frozenset(
        {
            SET_OP,
            DIFF_OP,
            REPEAT_OP,
            FLIP_OP,
            PWM_OP,
            FLIP2_OP,
            INTERVAL_OP,
            GATE_REPLAY_OP,
            SUBREG_FLUSH_OP,
            END_REPEAT_OP,
            END_FLIP_OP,
        }
    )
    # Burst-style ops carry per-reg state (last_repeat / last_flip /
    # pending_diffs) that crosses frame boundaries, with terminators
    # (val=0) that depend on the matching opener's run-time scheduling.
    # If a capture's window contains any of these, the dropped rows would
    # corrupt the burst's running state, causing the terminator to fire on
    # an empty pending state and emit a redundant write at decode time.
    # Abort captures whose window touches one of these ops.
    _burst_ops = frozenset(
        {
            REPEAT_OP,
            FLIP_OP,
            PWM_OP,
            FILTER_SWEEP_OP,
            FLIP2_OP,
            INTERVAL_OP,
        }
    )

    @staticmethod
    def _sort_strict_voice_order(df):
        """Sort rows within each frame by ``(voice, reg, op, original_index)``
        to match what ``RegLogParser._norm_pr_order`` produces downstream.

        Frame markers (FRAME_REG/DELAY_REG, both reg < 0) keep their
        original positions because their natural voice maps to the reg
        value (negative), which sorts before any voice-0 row, giving the
        same effective layout as a sort key of ``(f, v, reg, op, n)``
        with v = reg for negative regs.
        """
        if df.empty:
            return df
        df = df.copy()
        df["__order_n"] = np.arange(len(df), dtype=np.int64)
        is_marker = df["reg"].isin({FRAME_REG, DELAY_REG})
        df["__order_f"] = is_marker.astype(int).cumsum()
        # Voice = reg // VOICE_REG_SIZE for non-negative regs; for marker
        # rows fall back to the row order by setting voice to -1 so the
        # marker comes first within its frame group (sorts before any
        # non-marker row).
        df["__order_v"] = (df["reg"] // VOICE_REG_SIZE).astype(np.int64)
        df.loc[is_marker, "__order_v"] = -1
        df = df.sort_values(
            ["__order_f", "__order_v", "reg", "op", "__order_n"],
            kind="stable",
        )
        df = df.drop(columns=["__order_n", "__order_f", "__order_v"])
        return df.reset_index(drop=True)

    def apply(self, df, args=None, palettes=None):
        if args is not None and not getattr(args, "instrument_pass", True):
            return df
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        if "description" not in df.columns:
            df["description"] = 0
        # Sort rows within each frame to match the strict-numeric
        # voice ordering ``_norm_pr_order`` will apply downstream. This
        # makes Phase 2's walk dispatch rows in the same order the
        # post-pass simulators (DedupSetPass, SubregPass, _expand_ops)
        # will, so ``observe_frame``'s captured programs match and
        # PLAY_INSTRUMENT_OP slot indices stay aligned across walks.
        df = self._sort_strict_voice_order(df)
        fr = df[df["reg"] == FRAME_REG]["diff"]
        if fr.empty:
            return df
        frame_diff = int(fr.iloc[0])
        last_diff = _build_last_diff(df)

        gate_cap = getattr(args, "gate_palette_cap", None) if args is not None else None
        instr_window = getattr(args, "instrument_window", 8) if args is not None else 8
        instr_cap = (
            getattr(args, "instrument_palette_cap", None) if args is not None else None
        )

        # ----- Phase 1: collect candidate captures from a clean walk ------
        candidates = self._collect_candidates(
            df, frame_diff, last_diff, gate_cap, instr_window, instr_cap
        )
        candidates_by_start = defaultdict(list)
        for c in candidates:
            candidates_by_start[c["start_frame"]].append(c)

        # ----- Phase 2: walk again, decide replays, emit output -----------
        state = DecodeState(
            frame_diff,
            last_diff=last_diff,
            strict=False,
            gate_palette_cap=gate_cap,
            instrument_window=instr_window,
            instrument_palette_cap=instr_cap,
        )
        drop_idx_set = set()
        new_rows = []

        arrs, frame_starts = _df_arrays_and_frames(df)
        regs_all = arrs["reg"]
        vals_all = arrs["val"]
        ops_all = arrs["op"]
        subregs_all = arrs["subreg"]
        diffs_all = arrs["diff"]
        descs_all = arrs["description"]
        indices_all = arrs["Index"]
        n_total = len(df)
        n_frames = len(frame_starts)

        for fi in range(n_frames):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            cur_frame = fi
            f_writes = []
            starting = sorted(
                candidates_by_start.get(cur_frame, []),
                key=lambda c: min(c["drop_rows"]),
            )
            cand_idx = [0]

            # The closure captures ``starting`` / ``cand_idx`` / ``f_writes``
            # from the enclosing per-frame loop body. The function is also
            # invoked inside that same iteration so the per-frame state is
            # the intended capture; pylint's cell-var-from-loop warning
            # would only be a real bug if the function escaped the loop.
            # pylint: disable=cell-var-from-loop
            def fire_candidates(threshold):
                while cand_idx[0] < len(starting):
                    c = starting[cand_idx[0]]
                    cstart = min(c["drop_rows"])
                    if cstart > threshold:
                        break
                    cand_idx[0] += 1
                    program = c["program"]
                    if program not in state.instrument_palette:
                        continue
                    slot = state.instrument_palette.index(program)
                    v = c["voice"]
                    ctrl_reg = v * VOICE_REG_SIZE + 4
                    L = c["end_frame"] - c["start_frame"] + 1
                    drop_idx_set.update(c["drop_rows"])
                    new_rows.append(
                        {
                            "reg": int(ctrl_reg),
                            "val": int(slot),
                            "diff": int(state.diff_for(ctrl_reg)),
                            "op": int(PLAY_INSTRUMENT_OP),
                            "subreg": int(L),
                            "__pos": int(cstart),
                        }
                    )
                    synthetic = _FastRow(
                        reg=int(ctrl_reg),
                        val=int(slot),
                        subreg=int(L),
                        op=int(PLAY_INSTRUMENT_OP),
                        diff=int(state.diff_for(ctrl_reg)),
                        description=0,
                        Index=-1,
                    )
                    writes = DECODERS[PLAY_INSTRUMENT_OP].expand(synthetic, state)
                    if writes:
                        f_writes.extend(writes)

            for i in range(start, end):
                reg = int(regs_all[i])
                row_idx = int(indices_all[i])
                if reg < 0:
                    fire_candidates(row_idx)
                    if reg == DELAY_REG:
                        for _ in range(int(vals_all[i]) - 1):
                            state.tick_frame()
                            state.observe_frame(
                                [], frame_idx=cur_frame, track_instruments=True
                            )
                    continue
                fire_candidates(row_idx)
                if row_idx in drop_idx_set:
                    continue
                op = int(ops_all[i])
                decoder = DECODERS.get(op)
                if decoder is None:
                    continue
                row = _FastRow(
                    reg=reg,
                    val=int(vals_all[i]),
                    op=op,
                    subreg=int(subregs_all[i]),
                    diff=int(diffs_all[i]),
                    description=int(descs_all[i]),
                    Index=row_idx,
                )
                writes = decoder.expand(row, state)
                if writes:
                    f_writes.extend(writes)
            fire_candidates(float("inf"))

            f_writes.extend(state.tick_frame())
            state.observe_frame(f_writes, frame_idx=cur_frame, track_instruments=True)
        out = _splice_rows(df, list(drop_idx_set), new_rows)
        # Publish the authoritative palette so downstream passes
        # (DedupSetPass, SubregPass) and the final decoder walk
        # (_expand_ops, find_redundant_writes) initialise their
        # ``DecodeState.instrument_palette`` from it instead of
        # re-deriving by observation -- which would diverge because of
        # those passes' local dispatch choices. Stored on the explicit
        # Palettes container; previously rode ``df.attrs[...]`` which
        # pandas ``__finalize__`` deep-copied on every dataframe op.
        if palettes is not None:
            palettes.instrument_palette = list(state.instrument_palette)
        return out

    def _collect_candidates(
        self, df, frame_diff, last_diff, gate_cap, instr_window, instr_cap
    ):
        """Phase 1: walk the original df, building captures via the same
        observe_frame logic the downstream uses, and return the list of
        candidate captures with their drop-row indices. Each candidate is
        a dict ``{voice, start_frame, end_frame, program, drop_rows}``."""
        state = DecodeState(
            frame_diff,
            last_diff=last_diff,
            strict=False,
            gate_palette_cap=gate_cap,
            instrument_window=instr_window,
            instrument_palette_cap=instr_cap,
        )
        # Per-voice tracker mirrors state.open_instr_captures but holds row
        # indices instead of the bytes observe_frame appends.
        voice_open_rows = {}
        candidates = []
        arrs, frame_starts = _df_arrays_and_frames(df)
        regs_all = arrs["reg"]
        vals_all = arrs["val"]
        ops_all = arrs["op"]
        subregs_all = arrs["subreg"]
        diffs_all = arrs["diff"]
        descs_all = arrs["description"]
        indices_all = arrs["Index"]
        n_total = len(df)
        n_frames_collected = len(frame_starts)

        for fi in range(n_frames_collected):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames_collected else n_total
            cur_frame = fi
            f_writes = []
            voice_drops_this_frame = defaultdict(list)
            transpose_voices = set()
            burst_voices = set()
            for i in range(start, end):
                reg = int(regs_all[i])
                if reg < 0:
                    if reg == DELAY_REG:
                        for _ in range(int(vals_all[i]) - 1):
                            state.tick_frame()
                            state.observe_frame(
                                [], frame_idx=cur_frame, track_instruments=True
                            )
                    continue
                op = int(ops_all[i])
                if op == TRANSPOSE_OP:
                    mask = int(subregs_all[i])
                    for v in range(VOICES):
                        if mask & (1 << v):
                            transpose_voices.add(v)
                if 0 <= reg < VOICES * VOICE_REG_SIZE:
                    v = reg // VOICE_REG_SIZE
                    if (
                        op in self._voice_confined_ops
                        and reg not in _BUNDLE_REGS_FLAT
                        and op != GATE_REPLAY_OP
                    ):
                        voice_drops_this_frame[v].append(int(indices_all[i]))
                    if op in self._burst_ops:
                        burst_voices.add(v)
                if op == FILTER_SWEEP_OP:
                    for v in range(VOICES):
                        burst_voices.add(v)
                decoder = DECODERS.get(op)
                if decoder is not None:
                    row = _FastRow(
                        reg=reg,
                        val=int(vals_all[i]),
                        op=op,
                        subreg=int(subregs_all[i]),
                        diff=int(diffs_all[i]),
                        description=int(descs_all[i]),
                        Index=int(indices_all[i]),
                    )
                    writes = decoder.expand(row, state)
                    if writes:
                        f_writes.extend(writes)
            f_writes.extend(state.tick_frame())

            # Append this frame's voice-confined row indices to existing
            # open row trackers BEFORE observe_frame possibly closes them.
            # Mark trackers aborted on transpose or burst taint.
            for v in list(voice_open_rows.keys()):
                voice_open_rows[v]["drop_rows"].extend(voice_drops_this_frame[v])
                if v in transpose_voices or v in burst_voices:
                    voice_open_rows[v]["aborted"] = True

            state.observe_frame(f_writes, frame_idx=cur_frame, track_instruments=True)

            # observe_frame closed some captures -- finalise their row
            # trackers as candidates.
            for v, (program, _start_frame) in state.last_closed_instr_captures.items():
                vor = voice_open_rows.pop(v, None)
                if vor is None or vor.get("aborted"):
                    continue
                if not program or not vor["drop_rows"]:
                    continue
                candidates.append(
                    {
                        "voice": v,
                        "start_frame": vor["start_frame"],
                        "end_frame": cur_frame,
                        "program": program,
                        "drop_rows": list(vor["drop_rows"]),
                    }
                )

            # observe_frame opened new captures -- start row trackers for
            # them seeded with this frame's voice-confined row indices.
            # Abort immediately if any of this voice's burst-state queues
            # are non-empty at open time -- a burst spanning the capture
            # window would otherwise be double-emitted at decode time
            # (once via the burst row that survives in the post-pass form,
            # once via the program tuple that captured the burst's writes).
            for v in state.open_instr_captures:
                if v not in voice_open_rows:
                    base = v * VOICE_REG_SIZE
                    voice_regs = set(range(base, base + VOICE_REG_SIZE))
                    has_pending_burst = (
                        any(bool(state.pending_diffs.get(r)) for r in voice_regs)
                        or any(state.last_repeat.get(r, 0) for r in voice_regs)
                        or any(state.last_flip.get(r, 0) for r in voice_regs)
                    )
                    voice_open_rows[v] = {
                        "start_frame": state.open_instr_captures[v]["start_frame"],
                        "drop_rows": list(voice_drops_this_frame[v]),
                        "aborted": (
                            v in transpose_voices
                            or v in burst_voices
                            or has_pending_burst
                        ),
                    }

        # Discard any still-open trackers at end of stream (no end-of-stream
        # close in observe_frame, so the downstream never sees them either).
        return candidates


class DedupSetPass(MacroPass):
    """Drop full-byte SET tokens whose value already matches the running
    register state. The burst passes (PWM/FILTER_SWEEP/FLIP2/INTERVAL) and
    the DIFF/REPEAT/FLIP rewriter can together walk a register's running
    value to exactly the byte the next surviving SET intends to write.
    Without this pass, the encoder emits that SET unchanged and the
    decoder produces a redundant SID write.

    Runs last in PASSES so it sees the final pre-norm form.
    """

    def apply(self, df, args=None, palettes=None):
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        if "description" not in df.columns:
            df["description"] = 0
        state = _build_decode_state(df, args=args, palettes=palettes)
        if state is None:
            return df

        arrs, frame_starts = _df_arrays_and_frames(df)
        regs = arrs["reg"]
        vals = arrs["val"]
        ops = arrs["op"]
        subregs = arrs["subreg"]
        diffs = arrs["diff"]
        descs = arrs["description"]
        indices = arrs["Index"]
        drop_idx = []
        n_total = len(df)
        n_frames = len(frame_starts)
        for fi in range(n_frames):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            cur_frame = fi  # 0-indexed logical frame
            f_writes = []
            for i in range(start, end):
                reg = int(regs[i])
                if reg < 0:
                    if reg == DELAY_REG:
                        for _ in range(int(vals[i]) - 1):
                            state.tick_frame()
                            state.observe_frame([], frame_idx=cur_frame)
                    continue
                op = int(ops[i])
                subreg = int(subregs[i])
                val = int(vals[i])
                if op == SET_OP and subreg == -1 and state.last_val[reg] == val:
                    drop_idx.append(int(indices[i]))
                    continue
                decoder = DECODERS.get(op)
                if decoder is None:
                    continue
                row = _FastRow(
                    reg=reg,
                    val=val,
                    op=op,
                    subreg=subreg,
                    diff=int(diffs[i]),
                    description=int(descs[i]),
                    Index=int(indices[i]),
                )
                writes = decoder.expand(row, state)
                if writes:
                    f_writes.extend(writes)
            f_writes.extend(state.tick_frame())
            state.observe_frame(f_writes, frame_idx=cur_frame)
        if not drop_idx:
            return df
        return df.drop(index=drop_idx).reset_index(drop=True)


PASSES = [
    EndTerminatorPass(),
    PwmPass(),
    FilterSweepPass(),
    Flip2Pass(),
    TransposePass(),
    IntervalPass(),
    GateMacroPass(),
    # DedupSetPass MUST run before InstrumentProgramPass: redundant SETs
    # would otherwise land in instrument programs on the encoder side but
    # be dropped before downstream passes' simulators see them, causing
    # palette desync.
    DedupSetPass(),
    # InstrumentProgramPass publishes its authoritative palette via
    # ``df.attrs["instrument_palette"]``. DedupSetPass / SubregPass /
    # ``_expand_ops`` each initialise their ``DecodeState`` with that
    # frozen palette, so palette indices are aligned across all walks
    # by construction. Three earlier designs (parallel observe walks,
    # encoder-local palette, shadow-walk slot remap) all suffered from
    # palette-slot drift between encoder and downstream simulators;
    # publishing-via-attrs is what works.
    InstrumentProgramPass(),
    DedupSetPass(),
    SubregPass(),
]


# LoopPass runs in a separate later stage -- AFTER _norm_pr_order and
# _add_voice_reg have produced the final encoded form the LM sees. Frame
# matching in any earlier form would false-match: two frames that differ
# post-norm (different voice ordering, different VOICE_REG layout) can have
# identical row content pre-norm.
POST_NORM_PRE_VOICE_PASSES = [
    # Runs after _norm_pr_order but before _add_voice_reg. Regs are
    # absolute (not voice-rotated) so the fuzzy-loop walk inside
    # ``LoopPass`` (when ``--fuzzy-loop-pass`` is on) can dispatch
    # DECODERS correctly, and frame-level row order is canonical
    # (sorted by voice, reg, op) so frame-content hashes are stable
    # across occurrences.
    LoopPass(),
]


def attach_palettes_to_attrs(df, palettes):
    """Publish ``palettes`` to ``df.attrs`` so callers outside the
    orchestrator (e.g. ``iter_self_contained_row_blocks`` ->
    ``expand_to_literal_form`` -> ``_expand_ops``) can read them via
    the existing attrs path.

    NOT called from ``run_passes`` directly anymore -- between
    ``run_passes`` and the eventual ``_expand_ops`` consumer there are
    pandas-heavy ops (``_norm_pr_order``, ``_add_voice_reg``) that
    would deep-copy a populated attrs on every ``__finalize__``. The
    parse loop now calls this explicitly right before yield, so the
    intermediate phase runs with empty attrs.
    """
    if palettes is None:
        return df
    if palettes.gate_palette is not None:
        df.attrs["gate_palette"] = palettes.gate_palette
    if palettes.instrument_palette is not None:
        df.attrs["instrument_palette"] = palettes.instrument_palette
    return df


def run_post_norm_pre_voice_passes(df, args=None, palettes=None):
    """Apply passes that need post-norm row order but pre-voice-rotation
    regs. Currently just LoopPass (which does exact / transposed /
    fuzzy / DO_LOOP matching in a single sweep)."""
    for macro_pass in POST_NORM_PRE_VOICE_PASSES:
        df = macro_pass.apply(df, args=args, palettes=palettes)
    return df


def run_passes(df, args=None, palettes=None):
    """Apply every PRE-norm-order ``MacroPass`` in order. ``palettes`` is
    a ``Palettes`` instance threaded through producers (GateMacroPass,
    InstrumentProgramPass) and consumers (DedupSetPass, SubregPass).
    Caller passes ``Palettes()`` to opt into the explicit-threading
    path; ``None`` falls back to legacy observation-from-scratch in
    each downstream walk."""
    for macro_pass in PASSES:
        df = macro_pass.apply(df, args=args, palettes=palettes)
    return df
