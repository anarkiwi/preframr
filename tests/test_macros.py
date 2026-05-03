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
    EndTerminatorPass,
    FilterSweepPass,
    Flip2Pass,
    IntervalPass,
    PwmPass,
    SubregPass,
    TransposePass,
)
from preframr.reglogparser import RegLogParser
from preframr.stfconstants import (
    DIFF_OP,
    END_FLIP_OP,
    END_REPEAT_OP,
    FC_LO_REG,
    FILTER_REG,
    FILTER_SWEEP_OP,
    FLIP2_OP,
    FLIP_OP,
    FRAME_REG,
    INTERVAL_OP,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PWM_OP,
    REPEAT_OP,
    SET_OP,
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

    def test_both_nibbles_change_stays_full_byte(self):
        # default (0) -> 0x41 changes both nibbles -> leave as subreg=-1
        df = pd.DataFrame([_frame(), _row(4, 0x41, op=SET_OP)])
        result = SubregPass().apply(df)
        sub_rows = result[(result["reg"] == 4) & (result["op"] == SET_OP)]
        self.assertEqual(len(sub_rows), 1)
        self.assertEqual(int(sub_rows.iloc[0]["subreg"]), -1)
        self.assertEqual(int(sub_rows.iloc[0]["val"]), 0x41)

    def test_no_change_left_alone(self):
        # Two identical SETs (an upstream squeeze should remove this case but
        # SubregPass should not split unchanged values either way).
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x41, op=SET_OP),
                _frame(),
                _row(4, 0x41, op=SET_OP),
            ]
        )
        result = SubregPass().apply(df)
        rows = result[(result["reg"] == 4) & (result["op"] == SET_OP)]
        # First: both nibbles change from 0 -> stays subreg=-1
        # Second: no nibbles change -> also stays subreg=-1
        self.assertTrue(all(int(s) == -1 for s in rows["subreg"]))
        self.assertEqual(len(rows), 2)

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
                _row(5, 0x35, op=SET_OP),  # both nibbles -> stays full-byte
                _frame(),
                _row(5, 0x36, op=SET_OP),  # lo only
                _frame(),
                _row(5, 0x86, op=SET_OP),  # hi only
            ]
        )
        encoded = SubregPass().apply(df.copy())
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


if __name__ == "__main__":
    unittest.main()
