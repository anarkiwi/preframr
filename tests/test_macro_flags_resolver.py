"""``apply_macro_flags_to_args`` resolves ``--macro-flags`` / ``--macro-config`` against the
preframr-tokens registry: unknown names and conflicting passes raise, dependencies are added
automatically, presets expand, and the empty default leaves every macro pass off."""

import argparse
import unittest

import pytest

pytest.importorskip("torch")

from preframr.args import add_args, apply_macro_flags_to_args
from preframr_tokens.macros.flag_registry import macro_flag_names
from preframr_tokens.tokenizer_config import REGISTERED_MACROS


def _resolve(flags="", config=""):
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args(["x", "--macro-flags", flags, "--macro-config", config])
    apply_macro_flags_to_args(args)
    return args


class TestMacroFlagsResolver(unittest.TestCase):
    def test_unknown_flag_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve("frobnicate")
        self.assertIn("frobnicate", str(ctx.exception))

    def test_conflicting_flags_raise(self):
        with self.assertRaises(ValueError):
            _resolve("skeleton_pass,freq_trajectory_pass")

    def test_dependency_auto_added(self):
        args = _resolve("melody_skeleton")
        self.assertTrue(args.generator_pass)
        self.assertTrue(args.melody_skeleton)

    def test_codebook_pipeline_reachable(self):
        args = _resolve("melody_skeleton,instrument_program")
        self.assertTrue(args.generator_pass)
        self.assertTrue(args.melody_skeleton)
        self.assertTrue(args.instrument_program)

    def test_named_config_with_override(self):
        args = _resolve("coarsen_pass", "full_macros")
        for flag in REGISTERED_MACROS:
            self.assertTrue(getattr(args, flag), flag)
        self.assertTrue(args.coarsen_pass)
        self.assertFalse(args.melody_skeleton)

    def test_empty_default_all_off(self):
        args = _resolve()
        for flag in macro_flag_names():
            self.assertFalse(getattr(args, flag), flag)

    def test_resolved_macro_flags_is_canonical_csv(self):
        args = _resolve("melody_skeleton")
        self.assertNotIn("{", args.macro_flags)
        self.assertNotIn("@", args.macro_flags)
        self.assertIn("generator_pass", args.macro_flags.split(","))


if __name__ == "__main__":
    unittest.main()
