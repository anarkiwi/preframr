"""Tests for ``preframr.macros``.

For each macro pass we verify two properties:
  - **Encode**: the pass replaces the expected source rows with a single
    macro op row carrying the right payload.
  - **Round-trip**: ``RegLogParser._expand_ops`` produces identical SID
    writes whether the macro pass was applied or not -- the lossless
    decode invariant.
"""

import unittest

import pandas as pd

from preframr.macros import (
    DecodeState,
    DedupSetPass,
    EndTerminatorPass,
    FilterSweepPass,
    Flip2Pass,
    GateMacroPass,
    InstrumentProgramPass,
    IntervalPass,
    LoopPass,
    PwmPass,
    SubregPass,
    TransposePass,
    _pack_back_ref,
    expand_loops,
    iter_self_contained_row_blocks,
    materialize_back_refs_outside,
    materialize_gate_palette_outside,
    materialize_instrument_palette_outside,
    run_passes,
    validate_back_refs,
    validate_gate_replays,
)
from preframr.reglogparser import RegLogParser
from preframr.stfconstants import (
    BACK_REF_OP,
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
    GATE_REPLAY_OP,
    INTERVAL_OP,
    PLAY_INSTRUMENT_OP,
    LOOP_OP_REG,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PWM_OP,
    REPEAT_OP,
    SET_OP,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICE_REG_SIZE,
)


class FakeArgs:
    """Args fixture; pass kwargs become attributes."""

    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame(diff=19000):
    return {
        "reg": FRAME_REG,
        "subreg": -1,
        "val": 0,
        "diff": diff,
        "op": SET_OP,
        "description": 0,
    }


def _row(reg, val, op=SET_OP, diff=32, subreg=-1):
    return {
        "reg": reg,
        "subreg": subreg,
        "val": val,
        "diff": diff,
        "op": op,
        "description": 0,
    }


def _expand(df):
    """Run the dispatcher and return the expanded SID write df."""
    return (
        RegLogParser()
        ._expand_ops(df, strict=False)
        .reset_index(drop=True)
        .astype(MODEL_PDTYPE)
    )


def _assert_round_trip(test, baseline_df, encoded_df):
    """Both forms must expand to byte-identical SID write streams."""
    pd.testing.assert_frame_equal(
        _expand(baseline_df.copy()),
        _expand(encoded_df.copy()),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# PWM
# ---------------------------------------------------------------------------
class TestPwmPass(unittest.TestCase):
    def _three_frame_pwm(self):
        return pd.DataFrame(
            [
                _frame(),
                _row(2, 32, op=DIFF_OP),
                _frame(),
                _row(2, 32, op=DIFF_OP),
                _frame(),
                _row(2, 32, op=DIFF_OP),
            ]
        )

    def test_encode_collapses_run(self):
        df = self._three_frame_pwm()
        result = PwmPass().apply(df, args=FakeArgs(pwm_pass=True))
        pwm = result[result["op"] == PWM_OP]
        self.assertEqual(len(pwm), 1)
        self.assertEqual(int(pwm.iloc[0]["reg"]), 2)
        self.assertEqual(int(pwm.iloc[0]["val"]), 32)
        self.assertEqual(int(pwm.iloc[0]["subreg"]), 3)
        # Original DIFFs gone
        leftover = result[(result["reg"] == 2) & (result["op"] == DIFF_OP)]
        self.assertEqual(len(leftover), 0)

    def test_min_run_2_is_replaced(self):
        df = pd.DataFrame(
            [_frame(), _row(2, 32, op=DIFF_OP), _frame(), _row(2, 32, op=DIFF_OP)]
        )
        result = PwmPass().apply(df, args=FakeArgs(pwm_pass=True))
        self.assertEqual(len(result[result["op"] == PWM_OP]), 1)
        self.assertEqual(int(result[result["op"] == PWM_OP].iloc[0]["subreg"]), 2)

    def test_isolated_diff_kept(self):
        df = pd.DataFrame([_frame(), _row(2, 32, op=DIFF_OP)])
        result = PwmPass().apply(df, args=FakeArgs(pwm_pass=True))
        self.assertEqual(len(result[result["op"] == PWM_OP]), 0)
        self.assertEqual(len(result[result["op"] == DIFF_OP]), 1)

    def test_round_trip(self):
        df = self._three_frame_pwm()
        encoded = PwmPass().apply(df.copy(), args=FakeArgs(pwm_pass=True))
        _assert_round_trip(self, df, encoded)


# ---------------------------------------------------------------------------
# SUBREG (smart nibble splitting; subsumes GATE_TOGGLE and FILTER_*/VOL split)
# ---------------------------------------------------------------------------
class TestSubregPass(unittest.TestCase):
    def test_lo_only_change_becomes_subreg_0(self):
        # default (0) -> 0x40 changes hi only -> subreg=1 with val=4
        # 0x40 -> 0x41 changes lo only -> subreg=0 with val=1
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x40, op=SET_OP),
                _frame(),
                _row(4, 0x41, op=SET_OP),
            ]
        )
        result = SubregPass().apply(df)
        rows = result[(result["reg"] == 4) & (result["op"] == SET_OP)]
        # Two SETs in, two SETs out (each split into a single subreg row).
        self.assertEqual(len(rows), 2)
        subregs = rows["subreg"].tolist()
        vals = rows["val"].tolist()
        self.assertEqual(subregs, [1, 0])
        self.assertEqual(vals, [4, 1])

    def test_both_nibbles_change_kept_as_full_byte_set(self):
        # default (0) -> 0x41 changes both nibbles. Conditional-split
        # policy (post b6cc564 design): keep as full-byte SET to avoid
        # the 2x row split + downstream SUBREG_FLUSH that this design
        # used to incur (~37% of corpus tokens).
        df = pd.DataFrame([_frame(), _row(4, 0x41, op=SET_OP)])
        result = SubregPass().apply(df)
        sub_rows = result[
            (result["reg"] == 4) & (result["op"] == SET_OP) & (result["subreg"] != -1)
        ]
        self.assertEqual(len(sub_rows), 0)
        full_rows = result[
            (result["reg"] == 4) & (result["op"] == SET_OP) & (result["subreg"] == -1)
        ]
        self.assertEqual(len(full_rows), 1)
        self.assertEqual(int(full_rows["val"].iloc[0]), 0x41)

    def test_no_change_left_alone(self):
        # Two identical SETs (an upstream squeeze should remove this case but
        # SubregPass should not split unchanged values).
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x41, op=SET_OP),
                _frame(),
                _row(4, 0x41, op=SET_OP),
            ]
        )
        result = SubregPass().apply(df)
        # First SET (both nibbles change from 0): kept as full-byte SET
        # under conditional-split. Second SET (no nibbles change): also
        # stays subreg=-1. Both rows visible at subreg=-1.
        sub01 = result[(result["reg"] == 4) & (result["subreg"].isin([0, 1]))]
        self.assertEqual(len(sub01), 0)
        full = result[(result["reg"] == 4) & (result["subreg"] == -1)]
        self.assertEqual(len(full), 2)

    def test_unaffected_regs_pass_through(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 100, op=SET_OP),  # reg 0 not in SUBREG_REGS
                _row(7, 200, op=SET_OP),  # reg 7 not in SUBREG_REGS
            ]
        )
        result = SubregPass().apply(df)
        self.assertEqual(len(result[result["subreg"] != -1]), 0)

    def test_round_trip_lone_nibbles(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x40, op=SET_OP),  # both change (subreg=-1)
                _frame(),
                _row(4, 0x41, op=SET_OP),  # gate-on, lo only
                _frame(),
                _row(4, 0x40, op=SET_OP),  # gate-off, lo only
                _frame(),
                _row(4, 0xC0, op=SET_OP),  # waveform change, hi only
                _frame(),
                _row(4, 0xC1, op=SET_OP),  # gate-on with new wave, lo only
            ]
        )
        encoded = SubregPass().apply(df.copy())
        _assert_round_trip(self, df, encoded)

    def test_round_trip_mixed_lone_and_paired(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(5, 0x00, op=SET_OP),
                _frame(),
                _row(5, 0x35, op=SET_OP),  # both nibbles change
                _frame(),
                _row(5, 0x36, op=SET_OP),  # lo only
                _frame(),
                _row(5, 0x86, op=SET_OP),  # hi only
            ]
        )
        encoded = SubregPass().apply(df.copy())
        _assert_round_trip(self, df, encoded)

    def test_case_3_inserts_flush_to_preserve_intermediate(self):
        # Two adjacent baseline SETs in the same frame, each touching only
        # one (different) nibble. Without FLUSH the decoder would coalesce
        # them and drop the intermediate write.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x05, op=SET_OP),  # lo only (0 -> 5)
                _row(4, 0x65, op=SET_OP),  # hi only (0 -> 6)
            ]
        )
        encoded = SubregPass().apply(df.copy())
        flushes = encoded[encoded["op"] == SUBREG_FLUSH_OP]
        self.assertEqual(len(flushes), 1)
        # Verify the round-trip preserves both writes.
        _assert_round_trip(self, df, encoded)

    def test_no_flush_for_both_nib_split(self):
        # A single both-nibble baseline SET produces (subreg=0, subreg=1) in
        # one go -- they SHOULD coalesce, no FLUSH between them.
        df = pd.DataFrame([_frame(), _row(4, 0x35, op=SET_OP)])
        encoded = SubregPass().apply(df.copy())
        self.assertEqual(int((encoded["op"] == SUBREG_FLUSH_OP).sum()), 0)
        _assert_round_trip(self, df, encoded)

    def test_no_flush_when_decoder_naturally_flushes(self):
        # Same nibble of same reg twice -> decoder's "same nib in pending"
        # rule already flushes, no encoder FLUSH needed.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x05, op=SET_OP),  # lo (0 -> 5)
                _row(4, 0x07, op=SET_OP),  # lo again (5 -> 7)
            ]
        )
        encoded = SubregPass().apply(df.copy())
        self.assertEqual(int((encoded["op"] == SUBREG_FLUSH_OP).sum()), 0)
        _assert_round_trip(self, df, encoded)

    def test_no_flush_across_different_regs(self):
        # Different reg between two subreg events -> decoder flushes
        # naturally, no encoder FLUSH needed.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x05, op=SET_OP),  # lo on reg 4
                _row(5, 0x03, op=SET_OP),  # any change on reg 5
                _row(4, 0x65, op=SET_OP),  # hi on reg 4
            ]
        )
        encoded = SubregPass().apply(df.copy())
        self.assertEqual(int((encoded["op"] == SUBREG_FLUSH_OP).sum()), 0)
        _assert_round_trip(self, df, encoded)


# ---------------------------------------------------------------------------
# TRANSPOSE
# ---------------------------------------------------------------------------
class TestTransposePass(unittest.TestCase):
    def test_encode_two_voices_same_delta(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 24, op=DIFF_OP),  # voice 0 freq +24
                _row(7, 24, op=DIFF_OP),  # voice 1 freq +24
                _row(14, -8, op=DIFF_OP),  # voice 2 different delta
            ]
        )
        result = TransposePass().apply(df, args=FakeArgs(transpose_pass=True))
        trans = result[result["op"] == TRANSPOSE_OP]
        self.assertEqual(len(trans), 1)
        self.assertEqual(int(trans.iloc[0]["val"]), 24)
        # mask = bits for voices 0 and 1
        self.assertEqual(int(trans.iloc[0]["subreg"]), 0b011)
        # Voice 2's DIFF still there
        v2 = result[(result["reg"] == 14) & (result["op"] == DIFF_OP)]
        self.assertEqual(len(v2), 1)

    def test_no_collapse_when_only_one_voice(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 24, op=DIFF_OP),
                _row(7, 8, op=DIFF_OP),
            ]
        )
        result = TransposePass().apply(df, args=FakeArgs(transpose_pass=True))
        self.assertEqual(len(result[result["op"] == TRANSPOSE_OP]), 0)

    def test_round_trip(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 24, op=DIFF_OP),
                _row(7, 24, op=DIFF_OP),
                _row(14, 24, op=DIFF_OP),
            ]
        )
        encoded = TransposePass().apply(df.copy(), args=FakeArgs(transpose_pass=True))
        _assert_round_trip(self, df, encoded)


# ---------------------------------------------------------------------------
# FLIP2
# ---------------------------------------------------------------------------
class TestFlip2Pass(unittest.TestCase):
    def test_encode_asymmetric_run(self):
        # +5, -3, +5, -3 on reg 2 across 4 consecutive frames
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
            ]
        )
        result = Flip2Pass().apply(df, args=FakeArgs(flip2_pass=True))
        flips = result[result["op"] == FLIP2_OP]
        self.assertEqual(len(flips), 1)
        self.assertEqual(int(flips.iloc[0]["subreg"]), 4)

    def test_skips_symmetric(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -5, op=DIFF_OP),
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -5, op=DIFF_OP),
            ]
        )
        result = Flip2Pass().apply(df, args=FakeArgs(flip2_pass=True))
        self.assertEqual(len(result[result["op"] == FLIP2_OP]), 0)

    def test_round_trip(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
            ]
        )
        encoded = Flip2Pass().apply(df.copy(), args=FakeArgs(flip2_pass=True))
        _assert_round_trip(self, df, encoded)


# ---------------------------------------------------------------------------
# INTERVAL
# ---------------------------------------------------------------------------
class TestIntervalPass(unittest.TestCase):
    def _parallel_motion_df(self):
        # Voices 0 and 1 both freq DIFF +10 across 3 frames
        return pd.DataFrame(
            [
                _frame(),
                _row(0, 10, op=DIFF_OP),
                _row(7, 10, op=DIFF_OP),
                _frame(),
                _row(0, 10, op=DIFF_OP),
                _row(7, 10, op=DIFF_OP),
                _frame(),
                _row(0, 10, op=DIFF_OP),
                _row(7, 10, op=DIFF_OP),
            ]
        )

    def test_encode_parallel_motion(self):
        df = self._parallel_motion_df()
        result = IntervalPass().apply(df, args=FakeArgs(interval_pass=True))
        intervals = result[result["op"] == INTERVAL_OP]
        # IntervalPass emits a burst replacing the *target* voice's DIFFs
        self.assertGreaterEqual(len(intervals), 1)
        # Source voice's DIFFs remain
        src = result[(result["reg"] == 0) & (result["op"] == DIFF_OP)]
        self.assertEqual(len(src), 3)

    def test_round_trip(self):
        df = self._parallel_motion_df()
        encoded = IntervalPass().apply(df.copy(), args=FakeArgs(interval_pass=True))
        _assert_round_trip(self, df, encoded)


# ---------------------------------------------------------------------------
# FILTER_SWEEP
# ---------------------------------------------------------------------------
class TestFilterSweepPass(unittest.TestCase):
    def test_encode_run(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FC_LO_REG, 16, op=DIFF_OP),
                _frame(),
                _row(FC_LO_REG, 16, op=DIFF_OP),
                _frame(),
                _row(FC_LO_REG, 16, op=DIFF_OP),
            ]
        )
        result = FilterSweepPass().apply(df, args=FakeArgs(filter_sweep_pass=True))
        sweeps = result[result["op"] == FILTER_SWEEP_OP]
        self.assertEqual(len(sweeps), 1)
        self.assertEqual(int(sweeps.iloc[0]["subreg"]), 3)

    def test_round_trip(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FC_LO_REG, 16, op=DIFF_OP),
                _frame(),
                _row(FC_LO_REG, 16, op=DIFF_OP),
            ]
        )
        encoded = FilterSweepPass().apply(
            df.copy(), args=FakeArgs(filter_sweep_pass=True)
        )
        _assert_round_trip(self, df, encoded)


# ---------------------------------------------------------------------------
# END_TERMINATORS
# ---------------------------------------------------------------------------
class TestEndTerminatorPass(unittest.TestCase):
    def test_inserts_terminator_after_implicit_end(self):
        # REPEAT_OP arms; SET_OP on same reg implicitly terminates
        df = pd.DataFrame(
            [
                _frame(),
                _row(7, 5, op=REPEAT_OP),
                _frame(),
                _row(7, 100, op=SET_OP),  # interrupts the repeat
            ]
        )
        result = EndTerminatorPass().apply(df, args=FakeArgs(end_terminator_pass=True))
        ends = result[result["op"] == END_REPEAT_OP]
        self.assertEqual(len(ends), 1)

    def test_no_terminator_when_explicit_end(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(7, 5, op=REPEAT_OP),
                _frame(),
                _row(7, 0, op=REPEAT_OP),  # explicit terminator
                _frame(),
                _row(7, 100, op=SET_OP),  # later SET is not implicit-end
            ]
        )
        result = EndTerminatorPass().apply(df, args=FakeArgs(end_terminator_pass=True))
        self.assertEqual(len(result[result["op"] == END_REPEAT_OP]), 0)


# ---------------------------------------------------------------------------
# DecodeState invariants
# ---------------------------------------------------------------------------
class TestDecodeState(unittest.TestCase):
    def test_tick_frame_consumes_pending_diffs(self):
        state = DecodeState(frame_diff=19000)
        state.last_diff[2] = 32
        state.pending_diffs[2].extend([32, 32])
        writes_a = state.tick_frame()
        self.assertEqual(len(writes_a), 1)
        self.assertEqual(writes_a[0], (2, 32, 32))
        writes_b = state.tick_frame()
        self.assertEqual(len(writes_b), 1)
        self.assertEqual(writes_b[0], (2, 64, 32))
        # Queue empty
        self.assertEqual(state.tick_frame(), [])

    def test_tick_frame_independent_pending_diffs_per_reg(self):
        state = DecodeState(frame_diff=19000)
        state.last_diff[2] = 32
        state.last_diff[9] = 32
        state.pending_diffs[2].extend([10, 10])
        state.pending_diffs[9].extend([5])
        writes = state.tick_frame()
        regs = sorted(w[0] for w in writes)
        self.assertEqual(regs, [2, 9])


# ---------------------------------------------------------------------------
# LOOP_BACK: encoder, expander, materializer, validator
# ---------------------------------------------------------------------------
def _back_ref_row(distance, length, diff=32):
    return {
        "reg": LOOP_OP_REG,
        "subreg": -1,
        "val": _pack_back_ref(distance, length),
        "diff": diff,
        "op": BACK_REF_OP,
        "description": 0,
    }


def _do_loop_begin_row(n, diff=32):
    return {
        "reg": LOOP_OP_REG,
        "subreg": 0,
        "val": n,
        "diff": diff,
        "op": DO_LOOP_OP,
        "description": 0,
    }


def _do_loop_end_row(diff=32):
    return {
        "reg": LOOP_OP_REG,
        "subreg": 1,
        "val": 0,
        "diff": diff,
        "op": DO_LOOP_OP,
        "description": 0,
    }


class TestExpandLoops(unittest.TestCase):
    def test_no_loops_passthrough(self):
        df = pd.DataFrame([_frame(), _row(4, 8, op=SET_OP)])
        out = expand_loops(df)
        # No loop ops -> df returned essentially unchanged
        self.assertEqual(len(out), len(df))

    def test_back_ref_copies_frames(self):
        # Two literal frames, then BACK_REF(distance=2, length=2) copies them.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),  # frame 0
                _frame(),
                _row(5, 10, op=SET_OP),  # frame 1
                _back_ref_row(distance=2, length=2),  # copies frames 0,1
            ]
        )
        out = expand_loops(df)
        # Expected: original 4 rows + 4 copied rows
        self.assertEqual(len(out), 8)
        # frames now: 0, 1, 0', 1' -- checking last two frames have correct content
        self.assertEqual(int(out.iloc[5]["val"]), 8)
        self.assertEqual(int(out.iloc[7]["val"]), 10)

    def test_do_loop_unrolls(self):
        df = pd.DataFrame(
            [
                _do_loop_begin_row(3),
                _frame(),
                _row(4, 8, op=SET_OP),
                _do_loop_end_row(),
            ]
        )
        out = expand_loops(df)
        # 1 frame body x 3 iterations = 3 frames, 6 rows
        self.assertEqual(len(out), 6)

    def test_do_loop_nested(self):
        # DO 2 [DO 3 [F R(4,8)] LOOP] LOOP -> 6 iterations
        df = pd.DataFrame(
            [
                _do_loop_begin_row(2),
                _do_loop_begin_row(3),
                _frame(),
                _row(4, 8, op=SET_OP),
                _do_loop_end_row(),
                _do_loop_end_row(),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 12)

    def test_back_ref_overlaps_present_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _back_ref_row(distance=2, length=4),  # overlaps current
            ]
        )
        with self.assertRaises(AssertionError):
            expand_loops(df)


class TestLoopPass(unittest.TestCase):
    def test_lz77_non_adjacent_match(self):
        # Two frames AB, then a divider, then AB again -- LZ77 territory
        # because the repeat is not consecutive (so DO_LOOP can't match it).
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),  # frame 0 (A)
                _frame(),
                _row(5, 10, op=SET_OP),  # frame 1 (B)
                _frame(),
                _row(6, 100, op=SET_OP),  # frame 2 (X, divider)
                _frame(),
                _row(4, 8, op=SET_OP),  # frame 3 (A)
                _frame(),
                _row(5, 10, op=SET_OP),  # frame 4 (B)
            ]
        )
        result = LoopPass().apply(df)
        backrefs = result[result["op"] == BACK_REF_OP]
        self.assertEqual(len(backrefs), 1)

    def test_do_loop_preferred_for_long_consecutive_runs(self):
        # 4 consecutive identical frames -- DO_LOOP saves more than LZ77
        df = pd.DataFrame([_frame(), _row(4, 8, op=SET_OP)] * 4)
        result = LoopPass().apply(df)
        do_begins = result[(result["op"] == DO_LOOP_OP) & (result["subreg"] == 0)]
        self.assertEqual(len(do_begins), 1)
        self.assertEqual(int(do_begins.iloc[0]["val"]), 4)

    def test_no_compression_for_unique_frames(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(6, 100, op=SET_OP),
            ]
        )
        result = LoopPass().apply(df)
        self.assertEqual(int((result["op"] == BACK_REF_OP).sum()), 0)
        self.assertEqual(int((result["op"] == DO_LOOP_OP).sum()), 0)

    def test_disabled_when_flag_off(self):
        # Two AB pairs would normally trigger a back-ref; with the flag
        # disabled LoopPass is a no-op.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(6, 100, op=SET_OP),
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
            ]
        )
        result = LoopPass().apply(df, args=FakeArgs(loop_pass=False, fuzzy_loop_pass=False))
        self.assertEqual(int((result["op"] == BACK_REF_OP).sum()), 0)
        self.assertEqual(int((result["op"] == DO_LOOP_OP).sum()), 0)
        self.assertEqual(len(result), len(df))

    def test_round_trip_lz77(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
            ]
        )
        encoded = LoopPass().apply(df.copy())
        # Round trip via expand_loops: should recover the original
        decoded = expand_loops(encoded)
        cols = ["reg", "val", "op", "subreg"]
        # Compare row content (ignoring index and ancillary columns)
        self.assertEqual(len(df), len(decoded))
        for i in range(len(df)):
            self.assertEqual(
                tuple(int(df.iloc[i][c]) for c in cols),
                tuple(int(decoded.iloc[i][c]) for c in cols),
            )


class TestMaterializeBackRefsOutside(unittest.TestCase):
    def test_keeps_self_contained_back_refs(self):
        # Build a stream where BACK_REF target is INSIDE the slice -- should keep it.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _back_ref_row(
                    distance=2, length=2
                ),  # target frames 0, 1 (within slice [0,2))
            ]
        )
        out = materialize_back_refs_outside(df, slice_lo_frame=0, slice_hi_frame=2)
        # back-ref kept (target 0, 1 are >= slice_lo_frame=0)
        self.assertEqual(int((out["op"] == BACK_REF_OP).sum()), 1)

    def test_materializes_escapee_back_refs(self):
        # back-ref target is BEFORE slice_lo -- materialize.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),  # frame 0
                _frame(),
                _row(5, 10, op=SET_OP),  # frame 1
                _back_ref_row(distance=2, length=2),  # target frames 0, 1
            ]
        )
        # Pretend slice starts at frame 2 (i.e. only the back-ref row is in slice)
        # The back-ref's targets (0, 1) are < slice_lo=2 -> escapee -> materialize.
        out = materialize_back_refs_outside(df, slice_lo_frame=2, slice_hi_frame=4)
        self.assertEqual(int((out["op"] == BACK_REF_OP).sum()), 0)
        # Should have inlined frames 0 and 1's literal rows
        # Original literal rows + 2 inlined frames (4 rows)
        self.assertEqual(len(out), 4 + 4)


class TestValidateBackRefs(unittest.TestCase):
    def test_valid_passes(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _back_ref_row(distance=2, length=2),
            ]
        )
        self.assertTrue(validate_back_refs(df, prompt_frame_count=0))

    def test_escapee_in_zero_prompt_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _back_ref_row(distance=5, length=1),  # reaches before output
            ]
        )
        with self.assertRaises(AssertionError):
            validate_back_refs(df, prompt_frame_count=0)

    def test_escapee_resolved_by_prompt(self):
        # Same escapee, but with a non-zero prompt frame count it's fine.
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _back_ref_row(distance=5, length=1),
            ]
        )
        self.assertTrue(validate_back_refs(df, prompt_frame_count=10))


# ---------------------------------------------------------------------------
# GATE_REPLAY_OP
# ---------------------------------------------------------------------------
def _gate_on_bundle(voice=0, ctrl=0x41, ad=0xF0, sr=0x20):
    base = voice * VOICE_REG_SIZE
    return [
        _row(base + 4, ctrl, op=SET_OP),
        _row(base + 5, ad, op=SET_OP),
        _row(base + 6, sr, op=SET_OP),
    ]


def _gate_off_bundle(voice=0, ctrl=0x40):
    base = voice * VOICE_REG_SIZE
    return [_row(base + 4, ctrl, op=SET_OP)]


class TestGateMacroPass(unittest.TestCase):
    def test_first_bundle_kept_literal(self):
        df = pd.DataFrame([_frame()] + _gate_on_bundle())
        result = GateMacroPass().apply(df)
        # First occurrence is a new palette slot -- no replay token.
        self.assertEqual(int((result["op"] == GATE_REPLAY_OP).sum()), 0)

    def test_repeated_bundle_replays(self):
        df = pd.DataFrame(
            [_frame()] + _gate_on_bundle()                       # frame 0: gate-on
            + [_frame()] + _gate_off_bundle()                    # frame 1: gate-off
            + [_frame()] + _gate_on_bundle()                     # frame 2: gate-on (same)
        )
        result = GateMacroPass().apply(df)
        replays = result[result["op"] == GATE_REPLAY_OP]
        self.assertEqual(len(replays), 1)
        # The replay's reg field is the voice 0 ctrl reg (4); subreg=1 (off->on);
        # val=0 (palette slot 0).
        self.assertEqual(int(replays.iloc[0]["reg"]), 4)
        self.assertEqual(int(replays.iloc[0]["subreg"]), 1)
        self.assertEqual(int(replays.iloc[0]["val"]), 0)
        # Second gate-on's three literal SETs were spliced out.
        ad_sets = result[(result["op"] == SET_OP) & (result["reg"] == 5)]
        self.assertEqual(len(ad_sets), 1)

    def test_distinct_bundles_get_distinct_slots(self):
        # Two off->on transitions with different AD bytes -> two palette slots
        # -> no replay.
        df = pd.DataFrame(
            [_frame()] + _gate_on_bundle(ctrl=0x41, ad=0xF0, sr=0x20)
            + [_frame()] + _gate_off_bundle()
            + [_frame()] + _gate_on_bundle(ctrl=0x41, ad=0x09, sr=0x20)
        )
        result = GateMacroPass().apply(df)
        self.assertEqual(int((result["op"] == GATE_REPLAY_OP).sum()), 0)

    def test_round_trip_against_dedup_baseline(self):
        # GateMacroPass + DedupSetPass should produce the same SID writes as
        # DedupSetPass alone. Otherwise the macro changes audible output.
        df = pd.DataFrame(
            [_frame()] + _gate_on_bundle()
            + [_frame()] + _gate_off_bundle()
            + [_frame()] + _gate_on_bundle()
            + [_frame()] + _gate_off_bundle()
            + [_frame()] + _gate_on_bundle()
        )
        baseline = DedupSetPass().apply(df.copy())
        encoded = DedupSetPass().apply(GateMacroPass().apply(df.copy()))
        self.assertGreater(int((encoded["op"] == GATE_REPLAY_OP).sum()), 0)
        _assert_round_trip(self, baseline, encoded)

    def test_round_trip_through_full_run_passes(self):
        # Regression for the SubregPass x GATE_REPLAY interaction: SubregPass's
        # nibble-split decisions must use the actual running register value
        # (which GATE_REPLAY mutates), not the value of the last SET it saw.
        df = pd.DataFrame(
            [_frame()] + _gate_on_bundle(ctrl=0x41, ad=0xF0, sr=0x20)
            + [_frame()] + _gate_off_bundle(ctrl=0x40)
            + [_frame()] + _gate_on_bundle(ctrl=0x41, ad=0xF0, sr=0x20)
            + [_frame()] + _gate_off_bundle(ctrl=0x40)
            + [_frame()] + _gate_on_bundle(ctrl=0x41, ad=0xF0, sr=0x20)
            # A second voice with its own palette.
            + [_frame()]
            + _gate_on_bundle(voice=1, ctrl=0x21, ad=0x09, sr=0xA0)
            + [_frame()] + _gate_off_bundle(voice=1, ctrl=0x20)
            + [_frame()]
            + _gate_on_bundle(voice=1, ctrl=0x21, ad=0x09, sr=0xA0)
        )
        encoded = run_passes(df.copy(), args=FakeArgs(loop_pass=False, fuzzy_loop_pass=False))
        # Round-trip must hold against just-DedupSetPass (which is what the
        # macroless baseline collapses to under run_passes minus GateMacroPass).
        baseline = DedupSetPass().apply(df.copy())
        _assert_round_trip(self, baseline, encoded)
        # And at least one GATE_REPLAY_OP must survive the chain.
        self.assertGreater(int((encoded["op"] == GATE_REPLAY_OP).sum()), 0)

    def test_palette_cap_keeps_over_cap_transitions_literal(self):
        # Three distinct off->on bundles with cap=2 -> only the first two
        # earn slots; the third stays literal and emits no GATE_REPLAY_OP.
        # cap=2 also means the second occurrence of the third bundle keeps
        # firing literally (no slot to reference).
        bundles = [
            (0x41, 0xF0, 0x20),  # slot 0
            (0x21, 0x09, 0xA0),  # slot 1
            (0x81, 0x33, 0x55),  # over cap -- stays literal
        ]
        rows = [_frame()]
        for ctrl, ad, sr in bundles:
            rows += _gate_on_bundle(ctrl=ctrl, ad=ad, sr=sr)
            rows += [_frame()]
            rows += _gate_off_bundle(ctrl=ctrl & ~1)
            rows += [_frame()]
            # Re-fire the same bundle -- if a slot was claimed it should
            # become a GATE_REPLAY_OP; otherwise it stays literal.
            rows += _gate_on_bundle(ctrl=ctrl, ad=ad, sr=sr)
            rows += [_frame()]
            rows += _gate_off_bundle(ctrl=ctrl & ~1)
            rows += [_frame()]
        df = pd.DataFrame(rows)
        result = GateMacroPass().apply(df, args=FakeArgs(gate_palette_cap=2))
        replays = result[result["op"] == GATE_REPLAY_OP]
        # Bundles 0 and 1 each fire twice (off->on + on->off) on their
        # second occurrence -> 4 replays. Bundle 2 never claims a slot, so
        # neither of its second-occurrence transitions replay.
        self.assertEqual(len(replays), 4)
        self.assertEqual(sorted(int(r) for r in replays["val"].tolist()), [0, 0, 1, 1])
        # All three "third bundle" SET rows still appear literally.
        third_ctrl = result[
            (result["op"] == SET_OP) & (result["reg"] == 4) & (result["val"] == 0x81)
        ]
        self.assertEqual(len(third_ctrl), 2)

    def test_per_voice_palettes_are_independent(self):
        # Voice 0 and voice 1 fire the same byte values, but each has its own
        # palette: the first occurrence on each voice is literal; the second
        # on each voice replays.
        v0 = _gate_on_bundle(voice=0, ctrl=0x41, ad=0xF0, sr=0x20)
        v1 = _gate_on_bundle(voice=1, ctrl=0x41, ad=0xF0, sr=0x20)
        v0_off = _gate_off_bundle(voice=0)
        v1_off = _gate_off_bundle(voice=1)
        df = pd.DataFrame(
            [_frame()] + v0                  # frame 0: v0 on
            + [_frame()] + v1                # frame 1: v1 on
            + [_frame()] + v0_off            # frame 2: v0 off
            + [_frame()] + v1_off            # frame 3: v1 off
            + [_frame()] + v0                # frame 4: v0 on (replay)
            + [_frame()] + v1                # frame 5: v1 on (replay)
        )
        result = GateMacroPass().apply(df)
        replays = result[result["op"] == GATE_REPLAY_OP]
        self.assertEqual(len(replays), 2)
        # Each replay targets a distinct voice's ctrl reg.
        self.assertEqual(
            sorted(int(r) for r in replays["reg"].tolist()),
            [4, 4 + VOICE_REG_SIZE],
        )


class TestMaterializeGatePaletteOutside(unittest.TestCase):
    def _three_transition_stream(self):
        return pd.DataFrame(
            [_frame()] + _gate_on_bundle()       # frame 0
            + [_frame()] + _gate_off_bundle()    # frame 1
            + [_frame()] + _gate_on_bundle()     # frame 2: replay slot 0
        )

    def test_keeps_in_slice_definition(self):
        # Slice covers all 3 frames -- the replay's slot is defined at frame 0
        # which is inside the slice, so leave the replay alone.
        encoded = GateMacroPass().apply(self._three_transition_stream())
        out = materialize_gate_palette_outside(
            encoded, slice_lo_frame=0, slice_hi_frame=3
        )
        self.assertEqual(int((out["op"] == GATE_REPLAY_OP).sum()), 1)

    def test_materializes_pre_slice_definition(self):
        # Slice starts at frame 2 -- the replay's slot was defined at frame 0
        # (before slice_lo_frame), so inline the literal SETs.
        encoded = GateMacroPass().apply(self._three_transition_stream())
        out = materialize_gate_palette_outside(
            encoded, slice_lo_frame=2, slice_hi_frame=3
        )
        self.assertEqual(int((out["op"] == GATE_REPLAY_OP).sum()), 0)
        # Three SET rows (ctrl, AD, SR) replace the GATE_REPLAY_OP row.
        materialised = out[
            (out["op"] == SET_OP) & (out["reg"].isin([4, 5, 6])) & (out["subreg"] == -1)
        ]
        # Frame 0 bundle (3 SETs on regs 4/5/6) + frame 1 gate-off (1 SET on
        # reg 4) + materialised replay at frame 2 (3 SETs) = 7.
        self.assertEqual(len(materialised), 7)


class TestValidateGateReplays(unittest.TestCase):
    def test_valid_passes(self):
        df = pd.DataFrame(
            [_frame()] + _gate_on_bundle()
            + [_frame()] + _gate_off_bundle()
            + [_frame()] + _gate_on_bundle()
        )
        encoded = GateMacroPass().apply(df)
        self.assertTrue(validate_gate_replays(encoded))

    def test_undefined_slot_raises(self):
        # Hand-craft a stream whose only GATE_REPLAY_OP references slot 0
        # of (voice=0, dir=1), but no prior literal bundle ever defined it.
        df = pd.DataFrame(
            [
                _frame(),
                {
                    "reg": 4,
                    "subreg": 1,
                    "val": 0,
                    "diff": 32,
                    "op": GATE_REPLAY_OP,
                    "description": 0,
                },
            ]
        )
        with self.assertRaises(AssertionError):
            validate_gate_replays(df)


class TestPlayInstrumentDispatch(unittest.TestCase):
    """b.1: the dispatcher schedules a multi-frame program from a single
    PLAY_INSTRUMENT_OP token. Decoder uses ``pending_program_writes`` and
    ``tick_frame`` to reach across frame boundaries with no other change to
    the per-row dispatcher.
    """

    def _build_df_with_palette(self, program, slot=0):
        # Synthetic stream: one frame with a PLAY_INSTRUMENT_OP firing
        # ``slot`` of voice 0, followed by 3 trailing empty frames so
        # tick_frame has somewhere to drain the queued writes.
        df = pd.DataFrame(
            [
                _frame(),
                {
                    "reg": 4,  # voice 0 ctrl reg, encodes voice via // VOICE_REG_SIZE
                    "subreg": int(len(program)),
                    "val": int(slot),
                    "diff": 32,
                    "op": int(PLAY_INSTRUMENT_OP),
                    "description": 0,
                },
                _frame(),
                _frame(),
                _frame(),
            ]
        )
        # Patch _expand_ops's state seed so we control instrument_palette.
        loader = RegLogParser()
        # Rather than refactor the seed point, run _expand_ops once to
        # observe what state it constructs, then inject the palette via a
        # subclass hook. Simpler: do the dispatch ourselves.
        from preframr.macros import expand_loops, DecodeState
        df = expand_loops(df.copy())
        last_diff = {int(r): 32 for r in df["reg"].unique()}
        state = DecodeState(
            int(df[df["reg"] == FRAME_REG]["diff"].iloc[0]),
            last_diff=last_diff,
            strict=False,
        )
        state.instrument_palette = [program]
        df["__f"] = (
            df["reg"]
            .isin({-128, -127})
            .astype(int)
            .cumsum()
        )
        per_frame_writes = []
        for _f, f_df in df.groupby("__f", sort=True):
            f_writes = []
            for row in f_df.itertuples():
                reg = int(row.reg)
                if reg < 0:
                    continue
                from preframr.macros import DECODERS
                decoder = DECODERS.get(int(row.op))
                if decoder is None:
                    continue
                writes = decoder.expand(row, state)
                if writes:
                    f_writes.extend(writes)
            f_writes.extend(state.tick_frame())
            per_frame_writes.append(f_writes)
        return per_frame_writes

    def test_writes_appear_at_correct_relative_frames(self):
        # Program: write reg 4 (voice 0 ctrl) at rel_frame 0,
        # reg 5 (AD) at rel_frame 1, reg 6 (SR) at rel_frame 3.
        program = (
            (0, 4, 0x41),
            (1, 5, 0xF0),
            (3, 6, 0x20),
        )
        per_frame = self._build_df_with_palette(program)
        # First frame contains the macro and tick_frame's pop of frame-0 writes.
        f0_writes = [(int(w[0]), int(w[1])) for w in per_frame[0] if int(w[0]) >= 0]
        self.assertIn((4, 0x41), f0_writes)
        self.assertNotIn((5, 0xF0), f0_writes)
        # Frame 1: AD write surfaces.
        f1_writes = [(int(w[0]), int(w[1])) for w in per_frame[1] if int(w[0]) >= 0]
        self.assertIn((5, 0xF0), f1_writes)
        # Frame 2: nothing (sparse rel_frame=2).
        f2_writes = [(int(w[0]), int(w[1])) for w in per_frame[2] if int(w[0]) >= 0]
        self.assertEqual(f2_writes, [])
        # Frame 3: SR write surfaces; queue empty after this.
        f3_writes = [(int(w[0]), int(w[1])) for w in per_frame[3] if int(w[0]) >= 0]
        self.assertIn((6, 0x20), f3_writes)

    def test_voice_offset_resolves_correctly(self):
        # Same program, but the macro fires for voice 2: reg field is the
        # voice's ctrl reg = 2*VOICE_REG_SIZE + 4 = 18. Resolves voice
        # offsets to absolute regs 18, 19, 20.
        program = (
            (0, 4, 0x41),
            (1, 5, 0xF0),
        )
        df = pd.DataFrame(
            [
                _frame(),
                {
                    "reg": 18,  # voice 2 ctrl reg
                    "subreg": int(len(program)),
                    "val": 0,
                    "diff": 32,
                    "op": int(PLAY_INSTRUMENT_OP),
                    "description": 0,
                },
                _frame(),
            ]
        )
        from preframr.macros import expand_loops, DECODERS, DecodeState
        df = expand_loops(df.copy())
        state = DecodeState(
            int(df[df["reg"] == FRAME_REG]["diff"].iloc[0]),
            last_diff={int(r): 32 for r in df["reg"].unique()},
            strict=False,
        )
        state.instrument_palette = [program]
        df["__f"] = df["reg"].isin({-128, -127}).astype(int).cumsum()
        all_writes_by_frame = []
        for _f, f_df in df.groupby("__f", sort=True):
            fw = []
            for row in f_df.itertuples():
                if int(row.reg) < 0:
                    continue
                writes = DECODERS[int(row.op)].expand(row, state)
                if writes:
                    fw.extend(writes)
            fw.extend(state.tick_frame())
            all_writes_by_frame.append(fw)
        # Frame 0 should have absolute reg 18 (= voice2 base 14 + offset 4).
        f0_regs = [int(w[0]) for w in all_writes_by_frame[0] if int(w[0]) >= 0]
        self.assertIn(18, f0_regs)
        # Frame 1 should have absolute reg 19 (= voice2 base + offset 5).
        f1_regs = [int(w[0]) for w in all_writes_by_frame[1] if int(w[0]) >= 0]
        self.assertIn(19, f1_regs)


class TestInstrumentProgramPass(unittest.TestCase):
    """b.2 encoder pass: collapse repeated multi-frame instrument programs
    into a single PLAY_INSTRUMENT_OP token. Verified end-to-end with the
    full ``run_passes`` chain so palette alignment between encoder and
    downstream simulators stays correct.
    """

    def _gate_on(self, voice, ctrl=0x41, ad=0xF0, sr=0x20):
        base = voice * VOICE_REG_SIZE
        return [
            _row(base + 4, ctrl, op=SET_OP),
            _row(base + 5, ad, op=SET_OP),
            _row(base + 6, sr, op=SET_OP),
        ]

    def _gate_off(self, voice, ctrl=0x40):
        base = voice * VOICE_REG_SIZE
        return [_row(base + 4, ctrl, op=SET_OP)]

    def _two_repeats(self):
        # voice 0: gate-on, gate-off, gate-on (replay), gate-off, gate-on (replay)
        rows = [_frame()]
        for _ in range(3):
            rows += self._gate_on(0)
            rows += [_frame()]
            rows += self._gate_off(0)
            rows += [_frame()]
        return pd.DataFrame(rows)

    def test_first_occurrence_left_literal(self):
        df = pd.DataFrame([_frame()] + self._gate_on(0))
        out = InstrumentProgramPass().apply(df, args=FakeArgs(
            gate_palette_cap=None,
            instrument_window=8,
            instrument_palette_cap=None,
        ))
        self.assertEqual(int((out["op"] == PLAY_INSTRUMENT_OP).sum()), 0)

    def test_round_trip_through_run_passes(self):
        df = self._two_repeats()
        encoded = run_passes(df.copy(), args=FakeArgs(
            gate_palette_cap=None,
            instrument_window=8,
            instrument_palette_cap=None,
            loop_pass=False, fuzzy_loop_pass=False,
        ))
        baseline = DedupSetPass().apply(df.copy())
        _assert_round_trip(self, baseline, encoded)

    def test_burst_in_window_aborts_capture(self):
        # PWM_OP within a voice's window should taint the capture so the
        # encoder doesn't drop rows whose burst state we can't replay.
        df = pd.DataFrame(
            [_frame()]
            + self._gate_on(0)
            + [_row(2, 32, op=PWM_OP, subreg=4)]  # 4-frame PWM burst on voice 0 PWM
            + [_frame()]
            + self._gate_off(0)
            + [_frame()]
            + self._gate_on(0)
            + [_row(2, 32, op=PWM_OP, subreg=4)]
            + [_frame()]
            + self._gate_off(0)
        )
        encoded = run_passes(df.copy(), args=FakeArgs(
            gate_palette_cap=None,
            instrument_window=8,
            instrument_palette_cap=None,
            loop_pass=False, fuzzy_loop_pass=False,
        ))
        # Round-trip remains correct regardless of whether the burst
        # case produces a PLAY_INSTRUMENT_OP -- the abort is correctness,
        # not a count assertion.
        baseline = DedupSetPass().apply(df.copy())
        _assert_round_trip(self, baseline, encoded)


class TestMaterializeInstrumentPaletteOutside(unittest.TestCase):
    """Mirror of TestMaterializeGatePaletteOutside for instrument programs.
    Programs are post-bundle-decoupling so they exclude ctrl/AD/SR; the
    materialised expansion is voice-confined to non-bundle regs.
    """

    def _gate_on(self, voice, ctrl=0x41, ad=0xF0, sr=0x20):
        base = voice * VOICE_REG_SIZE
        return [
            _row(base + 4, ctrl, op=SET_OP),
            _row(base + 5, ad, op=SET_OP),
            _row(base + 6, sr, op=SET_OP),
        ]

    def _gate_off(self, voice, ctrl=0x40):
        base = voice * VOICE_REG_SIZE
        return [_row(base + 4, ctrl, op=SET_OP)]

    def _two_freq_program(self):
        # voice 0 program with non-bundle freq writes that observe_frame
        # will capture into instrument_palette. Three repeats of:
        #   gate-on + freq write at frame 0
        #   freq write at frame 1
        #   gate-off
        rows = [_frame()]
        for _ in range(3):
            rows += self._gate_on(0)
            rows += [_row(0, 100, op=SET_OP)]  # freq_lo at rel_frame=0
            rows += [_frame()]
            rows += [_row(0, 200, op=SET_OP)]  # freq_lo at rel_frame=1
            rows += [_frame()]
            rows += self._gate_off(0)
            rows += [_frame()]
        return pd.DataFrame(rows)

    def test_in_slice_definition_kept(self):
        df = self._two_freq_program()
        encoded = run_passes(df.copy(), args=FakeArgs(
            gate_palette_cap=None,
            instrument_window=8,
            instrument_palette_cap=None,
            loop_pass=False, fuzzy_loop_pass=False,
        ))
        # Slice covering everything -- no PLAY_INSTRUMENT_OP should be
        # expanded (its slot is defined within the slice).
        play_count = int((encoded["op"] == PLAY_INSTRUMENT_OP).sum())
        out = materialize_instrument_palette_outside(
            encoded, slice_lo_frame=0, slice_hi_frame=100
        )
        self.assertEqual(int((out["op"] == PLAY_INSTRUMENT_OP).sum()), play_count)


class TestIterSelfContainedRowBlocks(unittest.TestCase):
    """Both training and inference funnel through this iterator. Each
    yielded block is self-contained: tokenizing and decoding it does not
    require any frames outside the block.
    """

    def _gate_on(self, voice, ctrl=0x41, ad=0xF0, sr=0x20):
        base = voice * VOICE_REG_SIZE
        return [
            _row(base + 4, ctrl, op=SET_OP),
            _row(base + 5, ad, op=SET_OP),
            _row(base + 6, sr, op=SET_OP),
        ]

    def _gate_off(self, voice, ctrl=0x40):
        base = voice * VOICE_REG_SIZE
        return [_row(base + 4, ctrl, op=SET_OP)]

    def _multi_replay_song(self):
        # Six gate cycles on voice 0 -- generates GATE_REPLAY_OPs and
        # potentially PLAY_INSTRUMENT_OPs, so block boundaries crossing
        # mid-song must rewrite cross-block refs.
        rows = [_frame()]
        for _ in range(6):
            rows += self._gate_on(0)
            rows += [_frame()]
            rows += self._gate_off(0)
            rows += [_frame()]
        return pd.DataFrame(rows)

    def test_blocks_cover_all_frames(self):
        df = self._multi_replay_song()
        encoded = run_passes(df.copy(), args=FakeArgs(
            gate_palette_cap=None,
            instrument_window=8,
            instrument_palette_cap=None,
            loop_pass=False, fuzzy_loop_pass=False,
        ))
        n_frames = int(
            encoded["reg"].isin([FRAME_REG, -127]).sum()  # FRAME_REG + DELAY_REG
        )
        total_block_frames = 0
        for block in iter_self_contained_row_blocks(encoded, frames_per_block=4):
            total_block_frames += int(
                block["reg"].isin([FRAME_REG, -127]).sum()
            )
        self.assertEqual(total_block_frames, n_frames)

    def test_each_block_validates(self):
        # Every yielded block must pass validate_gate_replays (no
        # undefined slot refs) and validate_back_refs.
        df = self._multi_replay_song()
        encoded = run_passes(df.copy(), args=FakeArgs(
            gate_palette_cap=None,
            instrument_window=8,
            instrument_palette_cap=None,
            loop_pass=False, fuzzy_loop_pass=False,
        ))
        for block in iter_self_contained_row_blocks(encoded, frames_per_block=3):
            self.assertTrue(validate_gate_replays(block))
            self.assertTrue(validate_back_refs(block))


if __name__ == "__main__":
    unittest.main()
