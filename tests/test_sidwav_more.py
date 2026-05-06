"""Coverage tests for ``preframr.sidwav.write_reg`` branches and the
``AsidProxy.cue_frame`` invariant."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from pyresidfp import SoundInterfaceDevice

from preframr.reg_mappers import FreqMapper
from preframr.sidwav import AsidProxy, sidq, write_reg, write_samples
from preframr.stfconstants import DELAY_REG, FRAME_REG, SET_OP


class _FakeSid:
    def __init__(self):
        self.freq_mapper = FreqMapper(cents=10)
        self.writes = []

    def write_register(self, reg, val):
        self.writes.append((reg, val))


class TestWriteReg(unittest.TestCase):
    def test_freq_reg_uses_if_map(self):
        sid = _FakeSid()
        write_reg(sid, 0, 60, {})  # reg 0 = voice 0 freq lo; freq table look-up
        # Two byte writes (lo + hi).
        self.assertEqual(len(sid.writes), 2)

    def test_freq_reg_negative_clamps_to_zero(self):
        sid = _FakeSid()
        write_reg(sid, 7, -5, {})  # voice 1 freq lo, val < 0 path
        self.assertEqual(len(sid.writes), 2)

    def test_freq_reg_oversize_clamps_to_max(self):
        sid = _FakeSid()
        max_in = max(sid.freq_mapper.if_map.keys())
        write_reg(sid, 14, max_in + 1000, {})  # voice 2 freq lo, oversize path
        self.assertEqual(len(sid.writes), 2)

    def test_pwm_reg_two_bytes(self):
        sid = _FakeSid()
        write_reg(sid, 2, 256, {})  # voice 0 PWM
        self.assertEqual(len(sid.writes), 2)

    def test_fc_lo_reg_two_bytes(self):
        sid = _FakeSid()
        write_reg(sid, 21, 100, {})  # FC_LO_REG (2-byte)
        self.assertEqual(len(sid.writes), 2)

    def test_unknown_reg_uses_reg_widths(self):
        sid = _FakeSid()
        write_reg(sid, 23, 7, {23: 1})
        self.assertEqual(len(sid.writes), 1)

    def test_default_width_one(self):
        sid = _FakeSid()
        write_reg(sid, 24, 5, {})  # reg 24 not in any 2-byte set, width=1.
        self.assertEqual(len(sid.writes), 1)


class TestAsidProxyCueFrame(unittest.TestCase):
    def test_cue_frame_sets_pending(self):
        sid = SoundInterfaceDevice()
        proxy = AsidProxy(sid=sid, asid=None, cents=10)
        self.assertFalse(proxy.pending_frame)
        proxy.cue_frame()
        self.assertTrue(proxy.pending_frame)

    def test_cue_frame_double_assert(self):
        sid = SoundInterfaceDevice()
        proxy = AsidProxy(sid=sid, asid=None, cents=10)
        proxy.cue_frame()
        with self.assertRaises(AssertionError):
            proxy.cue_frame()

    def test_clock_frequency_property(self):
        sid = SoundInterfaceDevice()
        proxy = AsidProxy(sid=sid, asid=None, cents=10)
        self.assertEqual(proxy.clock_frequency, sid.clock_frequency)

    def test_sampling_frequency_property(self):
        sid = SoundInterfaceDevice()
        proxy = AsidProxy(sid=sid, asid=None, cents=10)
        self.assertEqual(proxy.sampling_frequency, sid.sampling_frequency)


class TestWriteSamplesFrameMarkerPath(unittest.TestCase):
    def test_frame_marker_triggers_cue_frame(self):
        # Frame marker rows + a real-reg write + description transition;
        # exercises the cue_frame branch and the pbar set_description
        # path inside write_samples.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.wav")
            df = pd.DataFrame(
                [
                    {
                        "reg": FRAME_REG,
                        "val": 1,
                        "diff": 24,
                        "op": SET_OP,
                        "description": 0,
                    },
                    {"reg": 24, "val": 15, "diff": 24, "op": SET_OP, "description": 0},
                    {
                        "reg": FRAME_REG,
                        "val": 1,
                        "diff": 24,
                        "op": SET_OP,
                        "description": 1,
                    },
                    {"reg": 24, "val": 0, "diff": 24, "op": SET_OP, "description": 1},
                ]
            )
            df["delay"] = df["diff"] * sidq()
            write_samples(
                df,
                path,
                cents=10,
                reg_widths={},
                descriptions=["prompt", "predictions"],
            )
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
