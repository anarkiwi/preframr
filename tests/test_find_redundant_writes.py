"""Tests for ``find_redundant_writes`` (pure-pandas redundancy scan).

The CLI driver ``find_redundant_writes.py`` re-parses every dump under a
directory which is far too slow to run in CI. The detection logic itself
-- track each register's running value through ``_expand_ops`` and flag
any emitted write whose value matches the tracked value -- is small and
easy to exercise on a synthetic token DataFrame, which is what these
tests do.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from find_redundant_writes import find_redundant_writes  # noqa: E402
from preframr.stfconstants import FRAME_REG, MIN_DIFF, SET_OP  # noqa: E402


def _frame_marker(diff=20000):
    return {
        "reg": FRAME_REG,
        "val": 0,
        "op": SET_OP,
        "subreg": -1,
        "diff": diff,
        "description": 0,
    }


def _set_row(reg, val, diff=MIN_DIFF):
    return {
        "reg": reg,
        "val": val,
        "op": SET_OP,
        "subreg": -1,
        "diff": diff,
        "description": 0,
    }


def _make_token_df(rows):
    df = pd.DataFrame(rows)
    df.attrs["instrument_palette"] = None
    df.attrs["gate_palette"] = None
    return df


class FindRedundantWritesTest(unittest.TestCase):
    def test_empty_returns_empty(self):
        df = pd.DataFrame(columns=["reg", "val", "op", "subreg", "diff", "description"])
        df.attrs["instrument_palette"] = None
        df.attrs["gate_palette"] = None
        self.assertEqual(find_redundant_writes(df), [])

    def test_only_marker_no_writes_returns_empty(self):
        df = _make_token_df([_frame_marker()])
        self.assertEqual(find_redundant_writes(df), [])

    def test_single_set_no_redundancy(self):
        # First write of a register can never be redundant -- the tracker
        # has no prior value to match against.
        df = _make_token_df([_frame_marker(), _set_row(reg=4, val=10)])
        self.assertEqual(find_redundant_writes(df), [])

    def test_distinct_writes_no_redundancy(self):
        df = _make_token_df(
            [
                _frame_marker(),
                _set_row(reg=4, val=10),
                _frame_marker(),
                _set_row(reg=4, val=11),
            ]
        )
        self.assertEqual(find_redundant_writes(df), [])

    def test_repeated_write_is_flagged(self):
        df = _make_token_df(
            [
                _frame_marker(),
                _set_row(reg=4, val=10),
                _frame_marker(),
                _set_row(reg=4, val=10),  # redundant: reg=4 already 10
            ]
        )
        events = find_redundant_writes(df)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["write_reg"], 4)
        self.assertEqual(ev["write_val"], 10)
        # Source label for SET decoder rows is "op=<op> reg=<reg> val=<val>".
        self.assertIn("op=", ev["source"])
        self.assertIn("reg=4", ev["source"])

    def test_repeated_write_in_same_frame_is_flagged(self):
        # Two SETs of the same (reg, val) inside a single frame: the
        # second one is redundant relative to the first.
        df = _make_token_df(
            [
                _frame_marker(),
                _set_row(reg=4, val=10),
                _set_row(reg=4, val=10),
            ]
        )
        events = find_redundant_writes(df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["write_reg"], 4)
        self.assertEqual(events[0]["write_val"], 10)

    def test_multiple_redundancies_each_reported(self):
        df = _make_token_df(
            [
                _frame_marker(),
                _set_row(reg=4, val=10),
                _set_row(reg=5, val=20),
                _frame_marker(),
                _set_row(reg=4, val=10),  # redundant
                _set_row(reg=5, val=20),  # redundant
                _set_row(reg=5, val=21),  # not redundant -- reg=5 now 21
            ]
        )
        events = find_redundant_writes(df)
        self.assertEqual(len(events), 2)
        flagged = sorted((e["write_reg"], e["write_val"]) for e in events)
        self.assertEqual(flagged, [(4, 10), (5, 20)])

    def test_per_register_tracking(self):
        # Writes to *different* registers never collide, even with
        # identical values.
        df = _make_token_df(
            [
                _frame_marker(),
                _set_row(reg=4, val=10),
                _set_row(reg=5, val=10),  # different reg -- not redundant
            ]
        )
        self.assertEqual(find_redundant_writes(df), [])

    def test_event_dict_shape(self):
        df = _make_token_df(
            [
                _frame_marker(),
                _set_row(reg=4, val=10),
                _set_row(reg=4, val=10),
            ]
        )
        events = find_redundant_writes(df)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            set(events[0].keys()),
            {"token_idx", "source", "write_reg", "write_val"},
        )


if __name__ == "__main__":
    unittest.main()
