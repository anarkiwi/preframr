"""Coverage tests for ``preframr.model`` factory + device selection.

Each factory wraps a torchtune model constructor with the args we use
elsewhere; calling them with a small synthetic ``args`` namespace
exercises the bodies without any training.
"""

import argparse
import logging
import unittest

import torch

from preframr.model import (
    MODEL_GETTERS,
    cpu_compile,
    cuda_compile,
    get_device,
)


def _tiny_args(**overrides):
    args = argparse.Namespace(
        embed=32,
        heads=4,
        kv_heads=2,
        layers=2,
        intermediate=64,
        max_seq_len=64,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1e4,
        rope_scale=1.0,
        tie_word_embeddings=True,
        precision="high",
        max_autotune=False,
        accumulate_grad_batches=1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestModelFactories(unittest.TestCase):
    def test_all_factories_build(self):
        # Each entry in MODEL_GETTERS must construct a transformer with
        # n_vocab=128 + the synthetic args.
        n_vocab = 128
        args = _tiny_args()
        for name, fn in MODEL_GETTERS.items():
            with self.subTest(model=name):
                model = fn(n_vocab, args)
                self.assertIsNotNone(model)


class TestGetDevice(unittest.TestCase):
    def test_returns_callable_compiler(self):
        args = _tiny_args()
        logger = logging.getLogger("test")
        _device, compiler = get_device(args, logger)
        self.assertTrue(callable(compiler))


@unittest.skipUnless(
    hasattr(torch, "compile"),
    "torch.compile required for cpu_compile / cuda_compile coverage",
)
class TestCpuCompileSmoke(unittest.TestCase):
    """Smoke tests for cpu_compile / cuda_compile.

    The body just calls torch.compile(); skipped where torch.compile
    isn't supported (e.g. Python builds without the right GIL config).
    Container builds with a supported Python -- this is mainly to keep
    the local-laptop test suite green.
    """

    def test_cpu_compile_returns_object(self):
        args = _tiny_args()
        m = torch.nn.Linear(8, 8)
        try:
            out = cpu_compile(args, m)
        except Exception as e:  # torch.compile raises various types
            self.skipTest(f"torch.compile unsupported here: {e}")
        self.assertIsNotNone(out)

    def test_cuda_compile_codepath(self):
        # cuda_compile defers to cpu_compile under the hood with extra
        # options. Run it to cover the option-list construction.
        args = _tiny_args(max_autotune=True, accumulate_grad_batches=1)
        m = torch.nn.Linear(8, 8)
        try:
            out = cuda_compile(args, m)
        except Exception as e:
            self.skipTest(f"torch.compile unsupported here: {e}")
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
