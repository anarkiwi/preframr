"""Unit tests for ``preframr.predict_lib``: the pure-helper layer that
predict.py imports. predict.py itself is integration-tested via
run_*_int_test.sh; these tests cover the bits that are pure logic
and don't need a full checkpoint + dataset stack."""

import os
import tempfile
import time
import unittest

from preframr.predict_lib import add_ext, describe_cycles, get_ckpt


class TestDescribeCycles(unittest.TestCase):
    def test_format(self):
        s = describe_cycles(0)
        self.assertIn("0 cycles", s)
        self.assertIn("seconds", s)

    def test_pal_irq_window(self):
        # PAL IRQ ≈ 19656 cycles ≈ 0.02 s. Format must round to 2dp.
        s = describe_cycles(19656)
        self.assertIn("19656 cycles", s)
        self.assertIn("0.02 seconds", s)

    def test_int_cast(self):
        # Float input is truncated to int in the cycles slot.
        s = describe_cycles(123.7)
        self.assertIn("123 cycles", s)


class TestAddExt(unittest.TestCase):
    def test_p_zero_passes_through(self):
        self.assertEqual(add_ext("/tmp/out.wav", 0), "/tmp/out.wav")

    def test_p_positive_inserts_index(self):
        self.assertEqual(add_ext("/tmp/out.wav", 1), "/tmp/out.1.wav")
        self.assertEqual(add_ext("/tmp/out.wav", 7), "/tmp/out.7.wav")

    def test_no_directory(self):
        # Path with no parent dir.
        self.assertEqual(add_ext("out.csv", 2), "out.2.csv")

    def test_compound_suffix_only_drops_last(self):
        # Path.stem only strips the final suffix.
        self.assertEqual(add_ext("/x/y.tar.gz", 1), "/x/y.tar.1.gz")


class TestGetCkpt(unittest.TestCase):
    def test_explicit_ckpt_passes_through(self):
        # When --model-state is supplied, get_ckpt returns it verbatim
        # without touching the filesystem.
        self.assertEqual(
            get_ckpt("/explicit/path.ckpt", "/nonexistent/tb_logs"),
            "/explicit/path.ckpt",
        )

    def test_picks_best_ckpt_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, "preframr", "version_0", "checkpoints")
            os.makedirs(d)
            best = os.path.join(d, "best-epoch=10-val_loss=1.0.ckpt")
            old = os.path.join(d, "epoch=5-step=10.ckpt")
            with open(old, "w") as f:
                f.write("old")
            time.sleep(0.01)  # ensure mtime ordering for the sort
            with open(best, "w") as f:
                f.write("best")
            self.assertEqual(get_ckpt(None, tmpdir), best)

    def test_falls_back_to_latest_per_epoch(self):
        # No ``best-*`` files; pick the most recent ``*.ckpt``.
        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, "preframr", "version_0", "checkpoints")
            os.makedirs(d)
            older = os.path.join(d, "epoch=5-step=10.ckpt")
            newer = os.path.join(d, "epoch=20-step=40.ckpt")
            with open(older, "w") as f:
                f.write("o")
            time.sleep(0.01)
            with open(newer, "w") as f:
                f.write("n")
            self.assertEqual(get_ckpt(None, tmpdir), newer)

    def test_picks_newest_best_when_multiple(self):
        # Multiple ``best-*`` -- mtime-newest wins (resumed training
        # leaves multiple).
        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, "preframr", "version_0", "checkpoints")
            os.makedirs(d)
            old = os.path.join(d, "best-epoch=10-val_loss=1.0.ckpt")
            new = os.path.join(d, "best-epoch=50-val_loss=0.5.ckpt")
            with open(old, "w") as f:
                f.write("o")
            time.sleep(0.01)
            with open(new, "w") as f:
                f.write("n")
            self.assertEqual(get_ckpt(None, tmpdir), new)

    def test_raises_when_nothing_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(IndexError):
                get_ckpt(None, tmpdir)


if __name__ == "__main__":
    unittest.main()
