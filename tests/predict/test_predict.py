"""Unit tests for ``preframr.predict``."""

import tempfile
import time
import unittest
from pathlib import Path

import torch

from preframr.inference import predict
from preframr.inference.predict import (
    _last_token_logits,
    add_ext,
    describe_cycles,
    get_ckpt,
)


class TestDescribeCycles(unittest.TestCase):
    def test_zero(self):
        out = describe_cycles(0)
        self.assertIn("0 cycles", out)
        self.assertIn("0.00 seconds", out)

    def test_nonzero_format(self):
        out = describe_cycles(985248)
        self.assertRegex(out, r"^\d+ cycles \d+\.\d{2} seconds$")

    def test_seconds_precision(self):
        self.assertTrue(describe_cycles(123).endswith(" seconds"))
        self.assertRegex(describe_cycles(123), r"\d+\.\d{2} seconds")


class TestAddExt(unittest.TestCase):
    def test_p_zero_returns_path_unchanged(self):
        self.assertEqual(add_ext("/tmp/foo.wav", 0), "/tmp/foo.wav")

    def test_p_positive_inserts_index_before_suffix(self):
        self.assertEqual(add_ext("/tmp/foo.wav", 3), "/tmp/foo.3.wav")

    def test_no_extension(self):
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
            time.sleep(0.05)
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


class TestLastTokenLogits(unittest.TestCase):
    """``Model.set_num_output_chunks > 0`` makes the lm_head return a
    ``list[Tensor]`` of seq-dim chunks instead of one ``(B, S, V)``
    tensor. ``_last_token_logits`` slices the right last-token row in
    either shape; without this, predict-time inference crashed with
    ``TypeError: list indices must be integers or slices`` at
    """

    def test_single_tensor_returns_last_row(self):
        out = torch.arange(12).reshape(1, 4, 3)
        logits = _last_token_logits(out)
        self.assertEqual(logits.shape, (1, 3))
        self.assertTrue(torch.equal(logits, torch.tensor([[9, 10, 11]])))

    def test_chunked_list_picks_last_chunk_last_row(self):
        chunk0 = torch.arange(6).reshape(1, 2, 3)
        chunk1 = torch.arange(6, 12).reshape(1, 2, 3)
        logits = _last_token_logits([chunk0, chunk1])
        self.assertEqual(logits.shape, (1, 3))
        self.assertTrue(torch.equal(logits, torch.tensor([[9, 10, 11]])))

    def test_single_chunk_list(self):
        chunk = torch.arange(3).reshape(1, 1, 3)
        logits = _last_token_logits([chunk])
        self.assertTrue(torch.equal(logits, torch.tensor([[0, 1, 2]])))


if __name__ == "__main__":
    unittest.main()
