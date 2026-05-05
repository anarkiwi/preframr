"""Unit tests for ``preframr.predict``.

Covers the pure helpers (``describe_cycles``, ``add_ext``, ``get_ckpt``)
plus a smoke import to catch undefined-name regressions like the one
``decoded_prompt`` introduced in the log line. The model-loading and
generation paths require a checkpoint and a GPU and are exercised by
``run_int_test.sh`` instead.
"""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path

from preframr import predict
from preframr.predict import add_ext, describe_cycles, get_ckpt


class TestDescribeCycles(unittest.TestCase):
    def test_zero(self):
        out = describe_cycles(0)
        self.assertIn("0 cycles", out)
        self.assertIn("0.00 seconds", out)

    def test_nonzero_format(self):
        # 985248 cycles ~= 1 second @ PAL CPU.
        out = describe_cycles(985248)
        self.assertRegex(out, r"^\d+ cycles \d+\.\d{2} seconds$")

    def test_seconds_precision(self):
        # Two decimals.
        self.assertTrue(describe_cycles(123).endswith(" seconds"))
        self.assertRegex(describe_cycles(123), r"\d+\.\d{2} seconds")


class TestAddExt(unittest.TestCase):
    def test_p_zero_returns_path_unchanged(self):
        self.assertEqual(add_ext("/tmp/foo.wav", 0), "/tmp/foo.wav")

    def test_p_positive_inserts_index_before_suffix(self):
        self.assertEqual(add_ext("/tmp/foo.wav", 3), "/tmp/foo.3.wav")

    def test_no_extension(self):
        # Path with no ``.`` extension — index appended via ``.0`` style.
        self.assertEqual(add_ext("/tmp/foo", 5), "/tmp/foo.5")

    def test_relative_path(self):
        self.assertEqual(add_ext("out.csv", 1), "out.1.csv")


class TestGetCkpt(unittest.TestCase):
    def test_explicit_ckpt_returned_unchanged(self):
        self.assertEqual(
            get_ckpt("/some/explicit.ckpt", "/ignored"), "/some/explicit.ckpt"
        )

    def test_finds_latest_in_tb_logs(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "version_0" / "checkpoints"
            sub.mkdir(parents=True)
            old = sub / "old.ckpt"
            new = sub / "new.ckpt"
            old.write_text("x")
            time.sleep(0.05)  # ensure mtime ordering
            new.write_text("y")
            self.assertEqual(get_ckpt(None, d), str(new))

    def test_raises_when_no_ckpt(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(IndexError):
                get_ckpt(None, d)


class TestModuleSurface(unittest.TestCase):
    """Catches undefined-name regressions: importing the module compiles
    every top-level + nested function. ``decoded_prompt`` shipped briefly
    as an undefined free variable; a smoke import won't catch unbound-on-
    invocation references inside a function body, but verifies the module
    parses and its public attributes resolve."""

    def test_public_callables_exist(self):
        for name in (
            "describe_cycles",
            "add_ext",
            "get_ckpt",
            "load_model",
            "generate_sequence",
            "run_predict",
            "Predictor",
            "main",
        ):
            self.assertTrue(
                callable(getattr(predict, name, None)),
                f"predict.{name} missing or not callable",
            )


if __name__ == "__main__":
    unittest.main()
