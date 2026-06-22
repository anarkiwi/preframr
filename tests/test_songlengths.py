"""Songlengths.md5 -> per-subtune frame budget (the sid-only nframes source)."""

import os
import tempfile
import unittest

from preframr.songlengths import (
    PAL_CPF,
    PAL_PHI,
    _token_seconds,
    sid_md5,
    subtune_frames,
)


class TestTokenSeconds(unittest.TestCase):
    def test_minutes_seconds(self):
        self.assertEqual(_token_seconds("0:56"), 56.0)
        self.assertEqual(_token_seconds("1:17"), 77.0)

    def test_milliseconds(self):
        self.assertAlmostEqual(_token_seconds("4:33.108"), 273.108)

    def test_hours(self):
        self.assertEqual(_token_seconds("1:02:03"), 3723.0)


class TestSubtuneFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sid = os.path.join(self.tmp, "Tune.sid")
        with open(self.sid, "wb") as fh:
            fh.write(b"PSID-fake-bytes")
        self.md5 = sid_md5(self.sid)
        self.songlengths = os.path.join(self.tmp, "Songlengths.md5")
        with open(self.songlengths, "w", encoding="utf-8") as fh:
            fh.write("[Database]\n; /MUSICIANS/T/Tune.sid\n")
            fh.write(f"{self.md5}=2:00 1:00\n")

    def test_frame_budget_matches_cycle_math(self):
        frames = subtune_frames(self.sid, 1, self.songlengths)
        self.assertEqual(frames, int(PAL_PHI * 120) // PAL_CPF)

    def test_second_subtune(self):
        self.assertEqual(
            subtune_frames(self.sid, 2, self.songlengths),
            int(PAL_PHI * 60) // PAL_CPF,
        )

    def test_ntsc_differs(self):
        pal = subtune_frames(self.sid, 1, self.songlengths)
        ntsc = subtune_frames(self.sid, 1, self.songlengths, ntsc=True)
        self.assertNotEqual(pal, ntsc)

    def test_unknown_md5_raises(self):
        other = os.path.join(self.tmp, "Other.sid")
        with open(other, "wb") as fh:
            fh.write(b"different")
        with self.assertRaises(KeyError):
            subtune_frames(other, 1, self.songlengths)

    def test_subtune_out_of_range_raises(self):
        with self.assertRaises(KeyError):
            subtune_frames(self.sid, 3, self.songlengths)


if __name__ == "__main__":
    unittest.main()
