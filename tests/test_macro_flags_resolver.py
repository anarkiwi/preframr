"""``apply_macro_flags_to_args`` resolves ``--macro-flags`` / ``--macro-config`` against the
preframr-tokens registry: unknown names and conflicting passes raise, dependencies are added
automatically, presets expand, and the empty default leaves every macro pass off."""

import argparse
import unittest

import pytest

pytest.importorskip("torch")

from preframr.args import add_args, apply_macro_flags_to_args
from preframr_tokens.macros.flag_registry import (
    FLAG_CONFLICTS,
    FLAG_REQUIRES,
    macro_flag_names,
)
from preframr_tokens.tokenizer_config import REGISTERED_MACROS


def _resolve(flags="", config=""):
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args(["x", "--macro-flags", flags, "--macro-config", config])
    apply_macro_flags_to_args(args)
    return args


def _a_conflicting_pair():
    for a, against in FLAG_CONFLICTS.items():
        for b in against:
            return (a, b)
    return None


def _a_dependency_pair():
    for flag, required in FLAG_REQUIRES.items():
        for req in required:
            return (flag, req)
    return None


def _required_by(flag):
    return set(FLAG_REQUIRES.get(flag, frozenset()))


def _some_flag():
    names = sorted(macro_flag_names())
    return names[0] if names else None


class TestMacroFlagsResolver(unittest.TestCase):
    def test_unknown_flag_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve("frobnicate")
        self.assertIn("frobnicate", str(ctx.exception))

    def test_conflicting_flags_raise(self):
        conflict = _a_conflicting_pair()
        if conflict is None:
            self.skipTest("FLAG_CONFLICTS empty in this tokens build")
        with self.assertRaises(ValueError):
            _resolve(",".join(conflict))

    def test_dependency_auto_added(self):
        dep = _a_dependency_pair()
        if dep is None:
            self.skipTest("FLAG_REQUIRES empty in this tokens build")
        flag, required = dep
        args = _resolve(flag)
        self.assertTrue(getattr(args, flag))
        self.assertTrue(getattr(args, required))

    def test_single_flag_sets_only_itself(self):
        flag = _some_flag()
        if flag is None:
            self.skipTest("no macro flags in this tokens build")
        args = _resolve(flag)
        self.assertTrue(getattr(args, flag))
        for other in macro_flag_names():
            if other != flag and other not in _required_by(flag):
                self.assertFalse(getattr(args, other), other)

    def test_named_config_with_override(self):
        extra = next(
            (f for f in sorted(macro_flag_names()) if f not in REGISTERED_MACROS), None
        )
        if extra is None:
            self.skipTest("no non-registered macro flag in this tokens build")
        args = _resolve(extra, "full_macros")
        for flag in REGISTERED_MACROS:
            self.assertTrue(getattr(args, flag), flag)
        self.assertTrue(getattr(args, extra))

    def test_empty_default_all_off(self):
        args = _resolve()
        for flag in macro_flag_names():
            self.assertFalse(getattr(args, flag), flag)

    def test_resolved_macro_flags_is_canonical_csv(self):
        flag = _some_flag()
        if flag is None:
            self.skipTest("no macro flags in this tokens build")
        args = _resolve(flag)
        self.assertNotIn("{", args.macro_flags)
        self.assertNotIn("@", args.macro_flags)
        self.assertIn(flag, args.macro_flags.split(","))


if __name__ == "__main__":
    unittest.main()
