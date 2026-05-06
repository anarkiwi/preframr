"""Smoke tests for ``preframr.utils`` + ``preframr.args``.

Both are small enough that import-and-call exercises most of the
statements; we mainly want them in coverage so the gate doesn't
falsely shrug off untested entry-point glue.
"""

import argparse
import logging
import unittest

from preframr.args import add_args
from preframr.utils import get_logger


class TestGetLogger(unittest.TestCase):
    def test_returns_logger(self):
        logger = get_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_level_string_uppercased(self):
        logger = get_logger("info")
        self.assertEqual(logger.level, logging.INFO)

    def test_level_string_uppercase_already(self):
        logger = get_logger("DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_handler_attached_once(self):
        # First call attaches a handler; second call should not double-add.
        logger = get_logger()
        n1 = len(logger.handlers)
        logger2 = get_logger()
        n2 = len(logger2.handlers)
        self.assertEqual(n1, n2)

    def test_handler_added_on_clean_logger(self):
        # Drop existing handlers so the ``if not logger.hasHandlers()``
        # branch fires and we cover the ``addHandler`` line.
        target = logging.getLogger("preframr.utils")
        prior = list(target.handlers)
        for h in prior:
            target.removeHandler(h)
        try:
            target.propagate = False  # avoid double counts via root
            logger = get_logger()
            self.assertGreaterEqual(len(logger.handlers), 1)
        finally:
            target.propagate = True
            for h in list(target.handlers):
                target.removeHandler(h)
            for h in prior:
                target.addHandler(h)


class TestAddArgs(unittest.TestCase):
    def test_argparse_defaults(self):
        parser = add_args(argparse.ArgumentParser())
        # Defaults the int tests rely on.
        args = parser.parse_args([])
        self.assertEqual(args.seq_len, 8192)
        self.assertEqual(args.tkvocab, 4096)
        self.assertEqual(args.predictions, 1)
        self.assertEqual(args.cents, 50)
        self.assertEqual(args.constrained_decode, False)

    def test_constrained_decode_flag(self):
        parser = add_args(argparse.ArgumentParser())
        args = parser.parse_args(["--constrained-decode"])
        self.assertTrue(args.constrained_decode)
        args = parser.parse_args(["--no-constrained-decode"])
        self.assertFalse(args.constrained_decode)

    def test_known_kv_args(self):
        parser = add_args(argparse.ArgumentParser())
        args = parser.parse_args(
            [
                "--layers",
                "12",
                "--heads",
                "16",
                "--max-epochs",
                "42",
            ]
        )
        self.assertEqual(args.layers, 12)
        self.assertEqual(args.heads, 16)
        self.assertEqual(args.max_epochs, 42)


if __name__ == "__main__":
    unittest.main()
