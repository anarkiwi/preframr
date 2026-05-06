"""Coverage tests for ``RegTokenizer`` helpers not covered by the
existing ``test_regtokenizer.py`` make_tokens path."""

import unittest

import pandas as pd

from preframr.regtokenizer import RegTokenizer
from preframr.stfconstants import MODEL_PDTYPE, SET_OP


class FakeArgs:
    tkvocab = 0
    tokenizer = "unigram"
    tkmodel = None


class TestRegMaxAndWidths(unittest.TestCase):
    def test_get_reg_max_picks_max_per_reg(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        df = pd.DataFrame(
            [
                {"reg": 0, "val": 5},
                {"reg": 0, "val": 200},
                {"reg": 1, "val": 7},
            ]
        )
        out = loader.get_reg_max(df, {})
        self.assertEqual(out[0], 200)
        self.assertEqual(out[1], 7)

    def test_get_reg_max_keeps_existing_max(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        df = pd.DataFrame([{"reg": 0, "val": 5}])
        out = loader.get_reg_max(df, {0: 999})
        self.assertEqual(out[0], 999)

    def test_get_reg_width_from_max(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        widths = loader.get_reg_width_from_max({0: 10, 1: 256, 2: 2**16, 3: 2**24})
        # val < 2^8 -> width 1; < 2^16 -> 2; < 2^24 -> 3; < 2^32 -> 4.
        self.assertEqual(widths[0], 1)
        self.assertEqual(widths[1], 2)
        self.assertEqual(widths[2], 3)
        self.assertEqual(widths[3], 4)


class TestTokenMetadataNoTkmodel(unittest.TestCase):
    def test_metadata_format(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        # Pretend a 2-token vocab without a unigram model attached.
        loader.tokens = pd.DataFrame(
            [
                {"op": SET_OP, "reg": 1, "subreg": -1, "val": 5},
                {"op": SET_OP, "reg": 2, "subreg": -1, "val": 9},
            ],
            dtype=MODEL_PDTYPE,
        )
        meta = loader.token_metadata()
        self.assertEqual(len(meta), 2)
        self.assertEqual(meta[0], "0 1 -1 5")


class TestEncodeDecodeNoTkmodel(unittest.TestCase):
    def test_pass_through(self):
        # Without a tkmodel, encode/decode are identity.
        loader = RegTokenizer(FakeArgs(), tokens=None)
        import numpy as np

        seq = np.array([1, 2, 3, 4], dtype=np.int64)
        self.assertTrue((loader.encode(seq) == seq).all())
        self.assertTrue((loader.decode(seq) == seq).all())


class TestAccumulateAutoCrunch(unittest.TestCase):
    def test_auto_crunches_at_threshold(self):
        # accumulate_tokens auto-calls crunch_tokens when frame_tokens
        # grows past 64 entries (line 189-190 path). Just make sure the
        # branch fires without error; the post-crunch length tracks the
        # remaining post-trigger accumulations.
        loader = RegTokenizer(FakeArgs(), tokens=None)
        df = pd.DataFrame(
            [{"op": SET_OP, "reg": 1, "subreg": -1, "val": 1}], dtype=MODEL_PDTYPE
        )
        # Simulate >64 accumulations to fire the crunch_tokens branch.
        for i in range(70):
            loader.accumulate_tokens(df.copy(), f"file{i}")
        # Crunched at 65, then 5 more accumulations -> 6 entries.
        self.assertGreater(len(loader.frame_tokens), 0)
        self.assertLess(len(loader.frame_tokens), 70)


if __name__ == "__main__":
    unittest.main()
