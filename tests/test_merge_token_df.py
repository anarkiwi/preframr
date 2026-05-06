"""Unit tests for ``RegTokenizer.merge_token_df`` substitution behaviour.

Captures the failures surfaced by the integration test:

  - The substitution path used to filter tokens by ``reg`` only,
    silently swapping a missing macro-op row for a wrong-op nearest-val
    token at the same reg. The fix filters by ``(op, reg, subreg)`` and
    refuses to substitute on macro ops at all.

  - The "substitute reg X val Y with val Y" no-op log line was a sign
    that the substitution wasn't actually finding a different val --
    the second merge then failed and the assert fired silently inside
    a try/except in ``materialize_block_array``.

These tests build a hand-crafted tokens df + input df, run
``merge_token_df``, and check the result. No parser, no docker, no
real-world dump file required.
"""

import unittest

import pandas as pd

from preframr.regtokenizer import RegTokenizer
from preframr.stfconstants import (
    BACK_REF_OP,
    DIFF_OP,
    FRAME_REG,
    GATE_REPLAY_OP,
    LOOP_OP_REG,
    MODEL_PDTYPE,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    PLAY_INSTRUMENT_OP,
    SET_OP,
)


class FakeArgs:
    def __init__(self):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = 128
        self.tkvocab = 0
        self.tkmodel = None
        self.max_files = 1
        self.diffq = 64
        self.tokenizer = "bpe"


def _tokens(rows):
    """Build a tokens df from a list of (op, reg, subreg, val) tuples."""
    return pd.DataFrame(
        [
            {
                "op": op,
                "reg": reg,
                "subreg": subreg,
                "val": val,
                "n": n,
                "count": 1,
            }
            for n, (op, reg, subreg, val) in enumerate(rows)
        ],
        dtype=MODEL_PDTYPE,
    )


def _df(rows, frame_diff=19000):
    """Build an input df. Always prefixes a FRAME_REG row so
    ``_merged_and_missing`` has its irq pivot."""
    out = [
        {
            "op": SET_OP,
            "reg": FRAME_REG,
            "subreg": -1,
            "val": 0,
            "diff": frame_diff,
            "description": 0,
        }
    ]
    for op, reg, subreg, val in rows:
        out.append(
            {
                "op": op,
                "reg": reg,
                "subreg": subreg,
                "val": val,
                "diff": 32,
                "description": 0,
            }
        )
    return pd.DataFrame(out, dtype=MODEL_PDTYPE)


class TestMergeTokenDfPassThrough(unittest.TestCase):
    def test_complete_alphabet_passes_through(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 1, -1, 5),
                (SET_OP, 1, -1, 7),
            ]
        )
        df = _df([(SET_OP, 1, -1, 5), (SET_OP, 1, -1, 7)])
        result = loader.merge_token_df(tokens, df)
        self.assertIsNotNone(result)
        # FRAME_REG + two SET rows -> 3 rows with valid n.
        self.assertEqual(len(result), 3)
        self.assertFalse(result["n"].isna().any())


class TestSetOpSubstitution(unittest.TestCase):
    """SET op has a continuous val space; nearest-val substitution within
    the same (op, reg, subreg) is allowed."""

    def test_substitutes_nearest_val_in_same_op_reg_subreg(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 1, -1, 5),
                (SET_OP, 1, -1, 9),
            ]
        )
        # val=7 missing; should substitute val=5 or val=9 (both equidistant,
        # implementation picks the first one ties give).
        df = _df([(SET_OP, 1, -1, 7)])
        result = loader.merge_token_df(tokens, df)
        self.assertIsNotNone(result)
        self.assertFalse(result["n"].isna().any())
        # Substituted to one of the two available SET vals on reg 1.
        substituted_val = int(
            result[(result["reg"] == 1) & (result["op"] == SET_OP)]["val"].iloc[0]
        )
        self.assertIn(substituted_val, (5, 9))

    def test_does_not_substitute_across_op_boundary(self):
        """A row with op=DIFF must not be substituted from SET tokens at
        the same reg, even if the nearest val is shared."""
        loader = RegTokenizer(FakeArgs(), tokens=None)
        # Tokens have SET reg=1 val=5 but no DIFF reg=1 val=*.
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 1, -1, 5),
            ]
        )
        df = _df([(DIFF_OP, 1, -1, 5)])  # missing -- DIFF op not in alphabet
        # Should refuse to substitute (no matching op-reg-subreg).
        with self.assertRaises(KeyError):
            loader.merge_token_df(tokens, df)


class TestMacroOpsRefuseSubstitution(unittest.TestCase):
    """Macro ops carry categorical val (back-ref distance, palette slot,
    program length, etc.); near-val substitution would silently corrupt
    the encoding. The substitution path raises KeyError instead."""

    def _assert_refuses(self, missing_row):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                # Plenty of nearby SET rows to tempt the old broken
                # reg-only substitution code.
                (SET_OP, missing_row[1], -1, 5),
                (SET_OP, missing_row[1], -1, 9),
            ]
        )
        df = _df([missing_row])
        with self.assertRaises(KeyError):
            loader.merge_token_df(tokens, df)

    def test_back_ref_op_refuses(self):
        # BACK_REF lives on LOOP_OP_REG; missing val must NOT be
        # substituted from any SET token at LOOP_OP_REG.
        self._assert_refuses((BACK_REF_OP, LOOP_OP_REG, -1, 1234))

    def test_pattern_replay_op_refuses(self):
        self._assert_refuses((PATTERN_REPLAY_OP, LOOP_OP_REG, 2, 5678))

    def test_pattern_overlay_op_refuses(self):
        self._assert_refuses((PATTERN_OVERLAY_OP, LOOP_OP_REG, 0, 4096))

    def test_gate_replay_op_refuses(self):
        self._assert_refuses((GATE_REPLAY_OP, 4, 0, 3))

    def test_play_instrument_op_refuses(self):
        # The exact failing row pattern from the integration test logs:
        # PLAY_INSTRUMENT slot 6 with subreg=10 (program length 10),
        # not present in a vocab learned from a different rotation.
        self._assert_refuses((PLAY_INSTRUMENT_OP, 4, 10, 6))


class TestSubstitutionRespectsSubreg(unittest.TestCase):
    """Two SET rows on the same reg with different subregs are distinct
    tokens; a missing (op, reg, subreg=A) must not substitute from
    (op, reg, subreg=B)."""

    def test_subreg_zero_does_not_match_subreg_one(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                # Only subreg=1 present.
                (SET_OP, 4, 1, 5),
            ]
        )
        df = _df([(SET_OP, 4, 0, 5)])  # subreg=0 missing
        with self.assertRaises(KeyError):
            loader.merge_token_df(tokens, df)


if __name__ == "__main__":
    unittest.main()
