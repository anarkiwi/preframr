import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr.regtokenizer import RegTokenizer
from preframr.stfconstants import MODEL_PDTYPE, UNICODE_BASE


class FakeArgs:
    def __init__(self, seq_len=128, tkvocab=0, diffq=64, tkmodel=None, tokenizer="bpe"):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.tkvocab = tkvocab
        self.tkmodel = tkmodel
        self.max_files = 1
        self.diffq = diffq
        self.tokenizer = tokenizer


class TestRegTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        max_n = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=2048,
                tkvocab=(max_n * 2),
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
            )
            loader = RegTokenizer(args, tokens=None)
            x = []
            for _ in range(100):
                x.extend([random.randint(0, max_n - 1) for _ in range(args.seq_len)])
            df = pd.DataFrame(x, dtype=pd.UInt16Dtype(), columns=["n"])
            df.loc[(df["n"] == 0) & (df["n"].shift() == 0), "n"] = 1

            for tokenizer in ("bpe", "unigram"):
                args.tokenizer = tokenizer
                loader = RegTokenizer(args, tokens=None)
                loader.train_tokenizer([(f"{tmpdir}/tune.dump.parquet", df, 1)])
                orig = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9], dtype=np.uint16)
                encoded = loader.encode(orig)
                decoded = loader.decode(encoded)
                self.assertTrue(
                    np.array_equal(orig, decoded), (tokenizer, orig, decoded)
                )

    def test_unicode(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        x = np.array([65, 0, 66, 67, 68, 69], dtype=np.uint16)
        y = loader.decode_unicode(loader.encode_unicode(x))
        self.assertTrue(np.array_equal(x, y), (x, y))
        for i in range(10):
            x = [i for i in range(65536 - UNICODE_BASE)]
            random.shuffle(x)
            x = np.array(x, dtype=np.uint16)
            y = loader.decode_unicode(loader.encode_unicode(x))
            self.assertTrue(np.array_equal(x, y), (x, y))

    def test_make_tokens(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        test_df = pd.DataFrame(
            [
                {"op": 0, "reg": 1, "subreg": -1, "val": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 2},
                {"op": 0, "reg": 1, "subreg": -1, "val": 2},
                {"op": 0, "reg": 1, "subreg": -1, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        # Vocab idx 0 is the synthetic pad token (PAD_REG=-1, val=0); real
        # tokens shift to 1..N. See ``RegTokenizer.make_tokens`` for the
        # PAD_ID/token-0 collision fix this enforces.
        tokens_df = pd.DataFrame(
            [
                {"op": 0, "reg": -1, "subreg": -1, "val": 0, "count": 0, "n": 0},
                {"op": 0, "reg": 1, "subreg": -1, "val": 1, "count": 3, "n": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 2, "count": 10, "n": 2},
                {"op": 0, "reg": 1, "subreg": -1, "val": 3, "count": 1, "n": 3},
                {"op": 0, "reg": 2, "subreg": -1, "val": 1, "count": 3, "n": 4},
                {"op": 0, "reg": 2, "subreg": -1, "val": 3, "count": 1, "n": 5},
                {"op": 0, "reg": 3, "subreg": -1, "val": 1, "count": 3, "n": 6},
                {"op": 0, "reg": 3, "subreg": -1, "val": 3, "count": 1, "n": 7},
                {"op": 0, "reg": 4, "subreg": -1, "val": 1, "count": 3, "n": 8},
                {"op": 0, "reg": 4, "subreg": -1, "val": 3, "count": 1, "n": 9},
                {"op": 0, "reg": 5, "subreg": -1, "val": 1, "count": 3, "n": 10},
                {"op": 0, "reg": 5, "subreg": -1, "val": 3, "count": 1, "n": 11},
            ],
            dtype=MODEL_PDTYPE,
        )
        for i in range(5):
            loader.accumulate_tokens(test_df, "test")
            test_df.loc[test_df["val"] != 2, "reg"] += 1
        result_df = loader.make_tokens().astype(MODEL_PDTYPE)
        self.assertTrue(
            tokens_df.equals(result_df), result_df.to_dict(orient="records")
        )
