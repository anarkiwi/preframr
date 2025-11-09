from datetime import timedelta
import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd
import scipy

from pyresidfp import SoundInterfaceDevice
from preframr.sidwav import write_samples, default_sid


class TestSidwav(unittest.TestCase):
    def test_write_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_wav_name = os.path.join(tmpdir, "test.wav")
            test_df = pd.DataFrame(
                [(1, 24, 0), (1024, 24, 15), (2048, 24, 0)],
                columns=["diff", "reg", "val"],
            )
            write_samples(test_df, test_wav_name, {}, irq=1)
            rate, data = scipy.io.wavfile.read(test_wav_name)
            self.assertEqual(rate, 48000)
            data = np.round(data, 2)
            self.assertTrue(abs(data.sum()))

    def test_ring_effects(self):
        def _test_samples(ctrl_val):
            sid = SoundInterfaceDevice(sampling_frequency=8e3)
            sec = timedelta(seconds=1)
            sid.write_register(24, 15)
            sid.write_register(6, 240)
            sid.write_register(0, 8)
            sid.write_register(1, 8)
            sid.write_register(20, 240)
            sid.write_register(14, 4)
            sid.write_register(15, 4)
            sid.clock(sec)
            sid.write_register(4, ctrl_val)
            samples = sid.clock(sec)
            return np.array(samples)

        ring = 2**2
        gate = 2**0

        # ring affects tri
        waveform = 16
        a = _test_samples(waveform + gate)
        b = _test_samples(waveform + gate + ring)
        assert not np.allclose(a, b, rtol=2, atol=2)

        # ring does not affect other
        for waveform in (32, 64, 128):
            a = _test_samples(waveform + gate)
            b = _test_samples(waveform + gate + ring)
            assert np.allclose(a, b, rtol=2, atol=2)

    def test_sync_effects(self):
        def _test_samples(ctrl_val1, ctrl_val2):
            sid = SoundInterfaceDevice(sampling_frequency=8e3)
            sec = timedelta(seconds=1)
            sid.write_register(24, 15)
            sid.write_register(6, 240)
            sid.write_register(0, 8)
            sid.write_register(1, 8)
            sid.write_register(20, 240)
            sid.write_register(14, 4)
            sid.write_register(15, 4)
            sid.write_register(4, ctrl_val1)
            sid.clock(sec)
            sid.write_register(4, ctrl_val2)
            samples = sid.clock(sec)
            return np.array(samples)

        sync = 2**1
        gate = 2**0

        # sync with no waveform has no effect.
        for waveform in (16, 32, 64, 128):
            a = _test_samples(waveform + gate, 0)
            b = _test_samples(waveform + gate, sync)
            assert np.allclose(a, b, rtol=2, atol=2)
