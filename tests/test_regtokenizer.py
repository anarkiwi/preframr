import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr.regtokenizer import RegTokenizer
from preframr.stfconstants import MODEL_PDTYPE, UNICODE_BASE


class FakeArgs:
    def __init__(self, seq_len=128, tkvocab=0, diffq=64, tkmodel=None):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.tkvocab = tkvocab
        self.max_files = 1
        self.diffq = diffq


class TestRegTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        max_n = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=8192,
                tkvocab=(max_n * 2),
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
            )
            loader = RegTokenizer(args, tokens=None)
            x = []
            for _ in range(100):
                x.extend([random.randint(0, max_n - 1) for _ in range(args.seq_len)])
            df = pd.DataFrame(x, dtype=pd.UInt16Dtype(), columns=["n"])

            for tokenizer in ("bpe", "unigram"):
                loader.train_tokenizer([df], tokenizer=tokenizer)
                orig = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9], dtype=np.uint16)
                encoded = loader.encode(orig)
                decoded = loader.decode(encoded)
                self.assertTrue(np.array_equal(orig, decoded), (orig, decoded))

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
                {"reg": 1, "val": 1},
                {"reg": 1, "val": 1},
                {"reg": 1, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        tokens_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1, "n": 0},
                {"reg": 1, "val": 2, "n": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        loader.accumulate_tokens(test_df, "test")
        result_df = loader.make_tokens().astype(MODEL_PDTYPE)
        self.assertTrue(tokens_df.equals(result_df), result_df)
