"""The args.py pipeline-name -> flag bridge stays tied to the preframr-tokens
macro flag registry: every bridged flag is a real registered macro flag, and the
legato clusters derive from the registered legato_pass_c* flags (no hand-listed
dead cluster)."""

import unittest

import pytest

pytest.importorskip("torch")

from preframr.args import _LEGATO_CLUSTERS, _PIPELINE_NAME_TO_FLAG
from preframr_tokens.macros.flag_registry import macro_flag_names


class TestPipelineFlagBridge(unittest.TestCase):
    def test_bridge_flags_are_registered(self):
        flags = macro_flag_names()
        bridge = {flag for flag, _ in _PIPELINE_NAME_TO_FLAG.values()}
        undeclared = bridge - flags
        self.assertEqual(
            undeclared,
            set(),
            f"_PIPELINE_NAME_TO_FLAG maps to flags no pass registers: {sorted(undeclared)}",
        )

    def test_legato_clusters_are_registered(self):
        flags = macro_flag_names()
        self.assertTrue(_LEGATO_CLUSTERS)
        for cluster in _LEGATO_CLUSTERS:
            self.assertIn(f"legato_pass_c{cluster}", flags, cluster)


if __name__ == "__main__":
    unittest.main()
