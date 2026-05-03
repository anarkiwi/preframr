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

import pandas as pd

from preframr.stfconstants import (
    BACK_REF_OP,
    DELAY_REG,
    DIFF_OP,
    DO_LOOP_OP,
    END_FLIP_OP,
    END_REPEAT_OP,
    FC_LO_REG,
    FILTER_REG,
    FILTER_SWEEP_OP,
    FLIP2_OP,
    FLIP_OP,
    FRAME_REG,
    INTERVAL_OP,
    LOOP_OP_REG,
    MIN_DIFF,
    MODE_VOL_REG,
    PWM_OP,
    REPEAT_OP,
    SET_OP,
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


# Registers whose byte value carries two semantically-independent nibbles
# that SubregPass splits into separate (subreg=0/1) tokens.
SUBREG_REGS = (4, 5, 6, FILTER_REG, MODE_VOL_REG)

PWM_REGS_BY_VOICE = tuple(2 + v * VOICE_REG_SIZE for v in range(VOICES))
FREQ_REGS_BY_VOICE = tuple(0 + v * VOICE_REG_SIZE for v in range(VOICES))
CTRL_REGS_BY_VOICE = tuple(4 + v * VOICE_REG_SIZE for v in range(VOICES))


# ---------------------------------------------------------------------------
# Decode side
# ---------------------------------------------------------------------------
class DecodeState:
    """Per-stream state shared by all ``MacroDecoder`` invocations."""

    def __init__(self, frame_diff, last_diff=None, strict=False):
        self.frame_diff = frame_diff
        self.last_val = defaultdict(int)
        self.last_repeat = defaultdict(int)
        self.last_flip = defaultdict(int)
        self.last_diff = dict(last_diff) if last_diff else {}
        self.strict = strict
        # PWM/FILTER_SWEEP-style bursts: each pending entry is one frame's
        # delta. tick_frame consumes one per frame.
        self.pending_diffs = defaultdict(list)
        # INTERVAL: each entry is a dict {tgt, src, remaining}. tick_frame
        # mirrors the per-frame change in src's last_val into tgt.
        self.interval_links = []
        # Snapshot of last_val at the END of the previous frame's tick_frame.
        # Used by INTERVAL to compute "what diff did src receive this frame".
        self.prev_frame_val = {}

    def diff_for(self, reg):
        return self.last_diff.get(reg, MIN_DIFF)

    def tick_frame(self):
        """Apply pending REPEAT/FLIP/PWM/INTERVAL ops at a frame boundary."""
        writes = []
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
        # Snapshot for next frame's INTERVAL comparison.
        for reg, v in list(self.last_val.items()):
            self.prev_frame_val[reg] = v
        return writes


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
        if row.subreg == 0:
            # Low-nibble-only update. Apply to state and emit one write with
            # the consolidated byte (preserving the prior high nibble).
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0xF0) | int(row.val)
        elif row.subreg == 1:
            # High-nibble-only update.
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0x0F) | (
                int(row.val) << 4
            )
        else:
            # Full-byte SET (used both for non-subreg-eligible regs and for
            # the both-nibbles-changed case on eligible regs).
            state.last_val[row.reg] = row.val
        return [(row.reg, state.last_val[row.reg], row.diff, row.description)]


class DiffDecoder(MacroDecoder):
    op_code = DIFF_OP

    def expand(self, row, state):
        assert row.subreg == -1
        state.last_val[row.reg] += row.val
        return [(row.reg, state.last_val[row.reg], row.diff, row.description)]


class RepeatDecoder(MacroDecoder):
    op_code = REPEAT_OP

    def expand(self, row, state):
        assert row.subreg == -1
        if row.val == 0:
            state.last_val[row.reg] += state.last_repeat[row.reg]
            del state.last_repeat[row.reg]
            return [(row.reg, state.last_val[row.reg], row.diff, row.description)]
        if state.strict:
            assert row.reg not in state.last_repeat, (row.reg, state.last_repeat)
        state.last_repeat[row.reg] = row.val
        return None


class FlipDecoder(MacroDecoder):
    op_code = FLIP_OP

    def expand(self, row, state):
        assert row.subreg == -1
        if row.val == 0:
            state.last_val[row.reg] += state.last_flip[row.reg]
            del state.last_flip[row.reg]
            return [(row.reg, state.last_val[row.reg], row.diff, row.description)]
        if state.strict:
            assert row.reg not in state.last_flip, (row.reg, state.last_flip)
        state.last_flip[row.reg] = row.val
        return None


class _PendingDiffBurstDecoder(MacroDecoder):
    """Common decoder for burst ops that schedule N consecutive DIFFs.

    Subclasses set ``op_code``. Encodes:
      reg    -> target register
      val    -> per-frame delta
      subreg -> burst length (frames)
    """

    def expand(self, row, state):
        length = int(row.subreg)
        assert length > 0, (self.op_code, row)
        state.last_diff[row.reg] = row.diff
        # Queue N deltas; tick_frame consumes one per frame, including the
        # frame this row appears in (matches how REPEAT_OP/FLIP_OP behave).
        for _ in range(length):
            state.pending_diffs[row.reg].append(row.val)
        return None


class PwmDecoder(_PendingDiffBurstDecoder):
    op_code = PWM_OP


class FilterSweepDecoder(_PendingDiffBurstDecoder):
    op_code = FILTER_SWEEP_OP


class Flip2Decoder(MacroDecoder):
    """Asymmetric ±a/±b alternation across N frames."""

    op_code = FLIP2_OP

    def expand(self, row, state):
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
        return None


class TransposeDecoder(MacroDecoder):
    """Single-frame: apply same delta to multiple voices' freq regs."""

    op_code = TRANSPOSE_OP

    def expand(self, row, state):
        # val = delta (signed); subreg = voice mask (bit v set => voice v).
        delta = int(row.val)
        if delta >= 0x8000:
            delta -= 0x10000
        mask = int(row.subreg)
        writes = []
        for v in range(VOICES):
            if mask & (1 << v):
                reg = FREQ_REGS_BY_VOICE[v]
                state.last_val[reg] += delta
                state.last_diff[reg] = row.diff
                writes.append((reg, state.last_val[reg], row.diff, row.description))
        return writes if writes else None


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
        state.last_diff[tgt_reg] = row.diff
        state.interval_links.append(
            {"tgt": tgt_reg, "src": src_reg, "remaining": length}
        )
        return None


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
    )
}


# ---------------------------------------------------------------------------
# Encode side
# ---------------------------------------------------------------------------
class MacroPass:
    """Base class for encode-side passes operating on a token DataFrame."""

    def apply(self, df, args=None):
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
    return combined


class PwmPass(MacroPass):
    """Collapse runs of consecutive identical PWM DIFFs into a PWM_OP burst.

    Operates on the per-voice pulse-width register (reg 2/9/16). Runs of
    length >= 2 are absorbed; the burst row encodes ``(reg, step, length)``
    in ``(reg, val, subreg)``. Voice rotation rotates the burst's reg field.
    """

    target_regs = PWM_REGS_BY_VOICE
    min_run = 2

    def apply(self, df, args=None):
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

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        drop_idx = []
        new_rows = []
        # Per-frame: which voices have a freq-DIFF and what value
        for f, f_df in df.groupby("mf"):
            freq_diffs = f_df[
                (f_df["reg"].isin(self.target_regs)) & (f_df["op"] == DIFF_OP)
            ]
            if len(freq_diffs) < 2:
                continue
            # Group by val
            for val, grp in freq_diffs.groupby("val"):
                if len(grp) < 2:
                    continue
                voice_mask = 0
                for reg in grp["reg"]:
                    v = self.target_regs.index(reg)
                    voice_mask |= 1 << v
                idxs = grp.index.tolist()
                drop_idx.extend(idxs)
                first_reg = int(grp["reg"].min())
                new_rows.append(
                    {
                        "reg": first_reg,
                        "val": int(val) & 0xFFFF,
                        "diff": int(grp["diff"].iloc[0]),
                        "op": int(TRANSPOSE_OP),
                        "subreg": int(voice_mask),
                        "__pos": int(min(idxs)),
                    }
                )

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


class Flip2Pass(MacroPass):
    """Asymmetric ±a/±b alternation across consecutive frames per (reg, voice).

    Skips symmetric flips (handled by FLIP_OP) and zero-valued steps. Burst
    encodes ``(a, b, length)`` in ``(val_packed, subreg)``.
    """

    min_run = 3

    def apply(self, df, args=None):
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

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        # Per voice, build per-frame freq-DIFF map
        n_frames = int(f_idx.max() + 1) if len(f_idx) else 0
        per_voice = {v: {} for v in range(VOICES)}  # voice -> {frame: (idx, val, diff)}
        for v, reg in enumerate(self.target_regs):
            mask = (df["reg"] == reg) & (df["op"] == DIFF_OP)
            sub = df[mask]
            for r in sub.itertuples():
                per_voice[v][int(r.mf)] = (int(r.Index), int(r.val), int(r.diff))

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

    def apply(self, df, args=None):
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
    """Smart byte-to-nibble splitting for the subreg-eligible registers.

    For each SET on a reg in ``SUBREG_REGS``, compare to the last value seen
    on that reg and rewrite into a more LM-friendly form:

      - low nibble only changed -> ``subreg=0`` row carrying the new lo
      - high nibble only changed -> ``subreg=1`` row carrying the new hi
      - both nibbles changed -> leave as full-byte ``subreg=-1`` (no split)

    This keeps the SID write stream byte-identical to baseline (each
    encoded row produces exactly one SID write) while collapsing the
    per-reg vocab from 256 byte values to ~16 lo + ~16 hi + a handful of
    both-changed bytes. No sequence length growth; no intra-frame write
    consolidation needed.

    Subsumes the previous ``GateTogglePass`` (a "gate-only" SET is now a
    ``subreg=0`` row on reg 4) and ``FilterModeVolPass`` (master-volume,
    filter-route, filter-mode changes are nibble-only on regs 23/24).
    """

    target_regs = SUBREG_REGS

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        last_val_per_reg = {}
        drop_idx = []
        new_rows = []
        for row in df.itertuples():
            if row.reg not in self.target_regs or row.op != SET_OP:
                continue
            if row.subreg != -1:
                continue  # already split by an earlier pass
            cur = int(row.val)
            prev = last_val_per_reg.get(int(row.reg), 0)
            cur_lo = cur & 0x0F
            cur_hi = (cur & 0xF0) >> 4
            prev_lo = prev & 0x0F
            prev_hi = (prev & 0xF0) >> 4
            lo_changed = cur_lo != prev_lo
            hi_changed = cur_hi != prev_hi
            if lo_changed and not hi_changed:
                drop_idx.append(int(row.Index))
                new_rows.append(
                    {
                        "reg": int(row.reg),
                        "val": int(cur_lo),
                        "diff": int(row.diff),
                        "op": int(SET_OP),
                        "subreg": 0,
                        "__pos": int(row.Index),
                    }
                )
            elif hi_changed and not lo_changed:
                drop_idx.append(int(row.Index))
                new_rows.append(
                    {
                        "reg": int(row.reg),
                        "val": int(cur_hi),
                        "diff": int(row.diff),
                        "op": int(SET_OP),
                        "subreg": 1,
                        "__pos": int(row.Index),
                    }
                )
            # both-changed and no-change cases: leave row untouched.
            last_val_per_reg[int(row.reg)] = cur
        return _splice_rows(df, drop_idx, new_rows)


class EndTerminatorPass(MacroPass):
    """Insert explicit END_REPEAT/END_FLIP rows when a REPEAT/FLIP run is
    interrupted by a SET on the same reg (rather than terminated with val=0).

    Tokens are predictability markers for the LM; decoder ignores them.
    """

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        active_repeat = {}  # reg -> True
        active_flip = {}
        new_rows = []
        for row in df.itertuples():
            if row.reg < 0:
                continue
            if row.op == REPEAT_OP:
                if row.val == 0:
                    active_repeat.pop(row.reg, None)
                else:
                    active_repeat[row.reg] = True
            elif row.op == FLIP_OP:
                if row.val == 0:
                    active_flip.pop(row.reg, None)
                else:
                    active_flip[row.reg] = True
            elif row.op == SET_OP:
                if active_repeat.pop(row.reg, None):
                    new_rows.append(
                        {
                            "reg": int(row.reg),
                            "val": 0,
                            "diff": int(row.diff),
                            "op": int(END_REPEAT_OP),
                            "subreg": -1,
                            "__pos": int(row.Index) - 0,
                        }
                    )
                if active_flip.pop(row.reg, None):
                    new_rows.append(
                        {
                            "reg": int(row.reg),
                            "val": 0,
                            "diff": int(row.diff),
                            "op": int(END_FLIP_OP),
                            "subreg": -1,
                            "__pos": int(row.Index) - 0,
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
    times still match."""
    cols = ["reg", "val", "op", "subreg"]
    return tuple(tuple(int(v) for v in df.iloc[r][cols]) for r in range(start, end))


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

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        frames = _slice_into_frames(df)
        n_frames = len(frames)
        if n_frames < self.min_lz_match:
            return df
        contents = [_frame_content(df, s, e) for s, e in frames]
        sizes = [e - s for s, e in frames]

        # LZ77 seed table: (content_i, content_{i+1}) -> [start frame indices]
        seed = defaultdict(list)
        out_rows = []
        sample_row = df.iloc[0]  # used to seed dtypes when constructing macro rows
        diff_default = int(sample_row["diff"]) if "diff" in df.columns else 0
        irq_default = int(df["irq"].iloc[0]) if "irq" in df.columns else -1

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
                body_rows = sum(sizes[i + k] for k in range(body_len))
                save = (n - 1) * body_rows - self.do_wrap_cost
                if save > best_save:
                    best_save, best_body, best_n = save, body_len, n
            return best_save, best_body, best_n

        def best_lz(i):
            best_save = 0
            best_dist = 0
            best_len = 0
            if i + 1 >= n_frames:
                return 0, 0, 0
            cands = seed.get((contents[i], contents[i + 1]))
            if not cands:
                return 0, 0, 0
            for cand in reversed(cands):
                if cand >= i:
                    continue
                length = 0
                while (
                    length < self.max_lz_length
                    and i + length < n_frames
                    and cand + length < i
                    and contents[cand + length] == contents[i + length]
                ):
                    length += 1
                if length < self.min_lz_match:
                    continue
                body_rows = sum(sizes[i + k] for k in range(length))
                save = body_rows - self.ref_cost
                if save > best_save:
                    best_save, best_dist, best_len = save, i - cand, length
            return best_save, best_dist, best_len

        def emit_literal(i):
            s, e = frames[i]
            for r in range(s, e):
                out_rows.append(df.iloc[r].to_dict())
            if i + 1 < n_frames:
                seed[(contents[i], contents[i + 1])].append(i)

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
                    seed[(contents[i + k], contents[i + k + 1])].append(i + k)

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
                for r in range(s, e):
                    out_rows.append(df.iloc[r].to_dict())
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
                    seed[(contents[i + k], contents[i + k + 1])].append(i + k)

        i = 0
        while i < n_frames:
            do_save, do_body, do_n = best_do(i)
            lz_save, lz_dist, lz_len = best_lz(i)
            best_now = max(do_save, lz_save)
            # One-frame lazy lookahead: if deferring buys >2 more tokens,
            # emit a literal at i now.
            if best_now > 0 and i + 1 < n_frames:
                la_do, _, _ = best_do(i + 1)
                la_lz, _, _ = best_lz(i + 1)
                if max(la_do, la_lz) > best_now + 2:
                    emit_literal(i)
                    i += 1
                    continue
            if do_save > 0 and do_save >= lz_save:
                emit_do_loop(i, do_body, do_n)
                i += do_body * do_n
            elif lz_save > 0:
                emit_back_ref(i, lz_dist, lz_len)
                i += lz_len
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
        return new_df.reset_index(drop=True)


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
    has_loops = df["op"].isin([BACK_REF_OP, DO_LOOP_OP]).any()
    if not has_loops:
        return df

    cols = list(df.columns)
    out = []
    output_frame_starts = []  # row idx in `out` where each output frame begins
    do_stack = []  # list of [body_start_row_in_input, remaining_iterations]

    def append_row(row_dict):
        out.append(row_dict)
        if row_dict["reg"] in _FRAME_MARKER_REGS:
            output_frame_starts.append(len(out) - 1)

    n = len(df)
    i = 0
    while i < n:
        row = df.iloc[i]
        op = int(row["op"]) if not pd.isna(row["op"]) else SET_OP
        if op == BACK_REF_OP:
            distance, length = _unpack_back_ref(row["val"])
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
            # Copy the L source frames out of the existing output buffer.
            for f in range(target, target + length):
                src_lo = output_frame_starts[f]
                src_hi = (
                    output_frame_starts[f + 1]
                    if f + 1 < len(output_frame_starts)
                    else len(out)
                )
                # snapshot to avoid mutation while iterating
                snapshot = list(out[src_lo:src_hi])
                for snap_row in snapshot:
                    append_row(dict(snap_row))
            i += 1
            continue
        if op == DO_LOOP_OP:
            subreg = int(row["subreg"]) if not pd.isna(row["subreg"]) else -1
            if subreg == 0:
                n_iter = int(row["val"])
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
        # Literal row.
        append_row({c: row[c] for c in cols})
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
    return expanded.reset_index(drop=True)


def materialize_back_refs_outside(df, slice_lo_frame, slice_hi_frame):
    """For Case A: rewrite ``df`` so that any BACK_REF whose target falls
    outside ``[slice_lo_frame, slice_hi_frame)`` (in logical output frames)
    is replaced with the literal frames it would have copied. The result
    is still a valid encoded stream, but every surviving BACK_REF in the
    slice ``[slice_lo_frame, slice_hi_frame)`` resolves within the slice.

    Use this when extracting a prompt window from a longer parsed stream
    so the prompt is self-contained.
    """
    if "op" not in df.columns or not df["op"].isin([BACK_REF_OP]).any():
        return df

    # First, fully expand to obtain the literal frame-row layout.
    literal = expand_loops(df.copy())
    literal_frame_starts = literal.index[
        literal["reg"].isin(_FRAME_MARKER_REGS)
    ].tolist()
    literal_frame_starts.append(len(literal))

    cols = list(df.columns)
    out = []
    output_frame_count = 0
    n = len(df)
    i = 0
    while i < n:
        row = df.iloc[i]
        op = int(row["op"]) if not pd.isna(row["op"]) else SET_OP
        if op == BACK_REF_OP:
            distance, length = _unpack_back_ref(row["val"])
            target = output_frame_count - distance
            if target < slice_lo_frame:
                # Materialize: copy literal rows for frames [target, target+length).
                for f in range(target, target + length):
                    s = literal_frame_starts[f]
                    e = literal_frame_starts[f + 1]
                    for r in range(s, e):
                        out.append({c: literal.iloc[r][c] for c in cols})
                output_frame_count += length
                i += 1
                continue
            # Keep the back-ref as-is.
            out.append({c: row[c] for c in cols})
            output_frame_count += length
            i += 1
            continue
        out.append({c: row[c] for c in cols})
        if row["reg"] in _FRAME_MARKER_REGS:
            output_frame_count += 1
        i += 1

    rebuilt = pd.DataFrame(out, columns=cols)
    for col, dt in df.dtypes.items():
        try:
            rebuilt[col] = rebuilt[col].astype(dt)
        except (TypeError, ValueError):
            pass
    return rebuilt.reset_index(drop=True)


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


PASSES = [
    EndTerminatorPass(),
    PwmPass(),
    FilterSweepPass(),
    Flip2Pass(),
    TransposePass(),
    IntervalPass(),
    SubregPass(),
]


# LoopPass runs in a separate later stage -- AFTER _norm_pr_order and
# _add_voice_reg have produced the final encoded form the LM sees. Frame
# matching in any earlier form would false-match: two frames that differ
# post-norm (different voice ordering, different VOICE_REG layout) can have
# identical row content pre-norm.
POST_NORM_PASSES = [
    LoopPass(),
]


def run_passes(df, args=None):
    """Apply every PRE-norm-order ``MacroPass`` in order."""
    for macro_pass in PASSES:
        df = macro_pass.apply(df, args=args)
    return df


def run_post_norm_passes(df, args=None):
    """Apply post-norm-order passes (currently just LoopPass) on the final
    encoded form (post _add_voice_reg)."""
    for macro_pass in POST_NORM_PASSES:
        df = macro_pass.apply(df, args=args)
    return df
