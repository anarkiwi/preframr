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
    DELAY_REG,
    DIFF_OP,
    END_FLIP_OP,
    END_REPEAT_OP,
    FC_LO_REG,
    FILTER_MODE_OP,
    FILTER_REG,
    FILTER_ROUTE_OP,
    FILTER_SWEEP_OP,
    FLIP2_OP,
    FLIP_OP,
    FRAME_REG,
    GATE_TOGGLE_OP,
    INTERVAL_OP,
    MASTER_VOL_OP,
    MIN_DIFF,
    MODE_VOL_REG,
    PWM_OP,
    REPEAT_OP,
    SET_OP,
    TRANSPOSE_OP,
    VOICES,
    VOICE_REG_SIZE,
)

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
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0x11110000) + row.val
            return None
        if row.subreg == 1:
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0b00001111) + (
                row.val << 4
            )
        else:
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


class GateToggleDecoder(MacroDecoder):
    """Flip the gate (LSB) of a voice's control register."""

    op_code = GATE_TOGGLE_OP

    def expand(self, row, state):
        # row.reg = the control reg (4/11/18). val/subreg unused.
        state.last_val[row.reg] ^= 1
        state.last_diff[row.reg] = row.diff
        return [(row.reg, state.last_val[row.reg], row.diff, row.description)]


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


class FilterRouteDecoder(MacroDecoder):
    """Set the routing nibble (low bits) of FILTER_REG without touching res."""

    op_code = FILTER_ROUTE_OP

    def expand(self, row, state):
        prev = state.last_val[FILTER_REG] & 0xF0
        state.last_val[FILTER_REG] = prev | (int(row.val) & 0x0F)
        state.last_diff[FILTER_REG] = row.diff
        return [(FILTER_REG, state.last_val[FILTER_REG], row.diff, row.description)]


class FilterModeDecoder(MacroDecoder):
    """Set the filter mode nibble (high bits) of MODE_VOL_REG."""

    op_code = FILTER_MODE_OP

    def expand(self, row, state):
        prev = state.last_val[MODE_VOL_REG] & 0x0F
        state.last_val[MODE_VOL_REG] = ((int(row.val) & 0x0F) << 4) | prev
        state.last_diff[MODE_VOL_REG] = row.diff
        return [(MODE_VOL_REG, state.last_val[MODE_VOL_REG], row.diff, row.description)]


class MasterVolDecoder(MacroDecoder):
    """Set the master volume nibble (low bits) of MODE_VOL_REG."""

    op_code = MASTER_VOL_OP

    def expand(self, row, state):
        prev = state.last_val[MODE_VOL_REG] & 0xF0
        state.last_val[MODE_VOL_REG] = prev | (int(row.val) & 0x0F)
        state.last_diff[MODE_VOL_REG] = row.diff
        return [(MODE_VOL_REG, state.last_val[MODE_VOL_REG], row.diff, row.description)]


DECODERS = {
    d.op_code: d
    for d in (
        SetDecoder(),
        DiffDecoder(),
        RepeatDecoder(),
        FlipDecoder(),
        PwmDecoder(),
        TransposeDecoder(),
        GateToggleDecoder(),
        Flip2Decoder(),
        IntervalDecoder(),
        EndRepeatDecoder(),
        EndFlipDecoder(),
        FilterSweepDecoder(),
        FilterRouteDecoder(),
        FilterModeDecoder(),
        MasterVolDecoder(),
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


class GateTogglePass(MacroPass):
    """Replace ctrl-reg SETs that flip only the gate bit (LSB) with GATE_TOGGLE.

    Tracks per-voice ctrl-reg state by reg index. A toggle-only SET is one
    where ``new_val == prev_val ^ 1``.
    """

    target_regs = CTRL_REGS_BY_VOICE

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        last_seen = {}
        replace_idx = []
        for row in df.itertuples():
            if row.reg not in self.target_regs or row.op != SET_OP:
                continue
            prev = last_seen.get(row.reg)
            if prev is not None and (prev ^ int(row.val)) == 1:
                replace_idx.append((row.Index, row.reg, row.diff))
            last_seen[row.reg] = int(row.val)

        if not replace_idx:
            return df
        df = _ensure_subreg(df)
        for idx, reg, diff in replace_idx:
            df.at[idx, "op"] = int(GATE_TOGGLE_OP)
            df.at[idx, "val"] = 0
            df.at[idx, "subreg"] = -1
            df.at[idx, "diff"] = int(diff)
            df.at[idx, "reg"] = int(reg)
        return df


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


class FilterModeVolPass(MacroPass):
    """Split FILTER_REG and MODE_VOL_REG nibble-changes into typed ops.

    For FILTER_REG (23): low nibble = routing bits, high nibble = resonance.
    Resonance changes stay as SETs (rare, narrow vocab); routing changes
    become FILTER_ROUTE_OP.

    For MODE_VOL_REG (24): low nibble = master volume, high nibble = filter
    mode (LP/BP/HP). Each becomes its own op when only that nibble changes.

    Does not use the existing _add_subreg byte-split machinery -- operates
    purely on per-row values vs prior state.
    """

    def apply(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        last_filter = None
        last_modevol = None
        for row in list(df.itertuples()):
            if row.op != SET_OP:
                continue
            if row.reg == FILTER_REG:
                cur = int(row.val)
                if last_filter is not None:
                    prev_lo = last_filter & 0x0F
                    cur_lo = cur & 0x0F
                    prev_hi = last_filter & 0xF0
                    cur_hi = cur & 0xF0
                    if prev_hi == cur_hi and prev_lo != cur_lo:
                        df.at[row.Index, "op"] = int(FILTER_ROUTE_OP)
                        df.at[row.Index, "val"] = int(cur_lo)
                        df.at[row.Index, "subreg"] = -1
                last_filter = cur
            elif row.reg == MODE_VOL_REG:
                cur = int(row.val)
                if last_modevol is not None:
                    prev_lo = last_modevol & 0x0F
                    cur_lo = cur & 0x0F
                    prev_hi = (last_modevol >> 4) & 0x0F
                    cur_hi = (cur >> 4) & 0x0F
                    if prev_hi == cur_hi and prev_lo != cur_lo:
                        df.at[row.Index, "op"] = int(MASTER_VOL_OP)
                        df.at[row.Index, "val"] = int(cur_lo)
                        df.at[row.Index, "subreg"] = -1
                    elif prev_lo == cur_lo and prev_hi != cur_hi:
                        df.at[row.Index, "op"] = int(FILTER_MODE_OP)
                        df.at[row.Index, "val"] = int(cur_hi)
                        df.at[row.Index, "subreg"] = -1
                last_modevol = cur
        return df


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


PASSES = [
    EndTerminatorPass(),
    FilterModeVolPass(),
    PwmPass(),
    FilterSweepPass(),
    Flip2Pass(),
    TransposePass(),
    IntervalPass(),
    GateTogglePass(),
]


def run_passes(df, args=None):
    """Apply every registered ``MacroPass`` in order."""
    for macro_pass in PASSES:
        df = macro_pass.apply(df, args=args)
    return df
