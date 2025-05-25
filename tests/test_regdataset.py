import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from preframr.regdataset import SeqMapper, RegDataset, MODEL_PDTYPE, VAL_PDTYPE


class FakeArgs:
    def __init__(self, seq_len=128, tkvocab=0, diffq=64, tkmodel=None):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.tkvocab = tkvocab
        self.tkmodel = tkmodel
        self.max_files = 1
        self.diffq = diffq
        self.token_csv = None


class TestRegDatasetLoader(unittest.TestCase):
    def test_tokenizer(self):
        max_n = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=8192,
                tkvocab=(max_n * 2),
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
            )
            loader = RegDataset(args)
            x = []
            for _ in range(100):
                x.extend([random.randint(0, max_n - 1) for _ in range(args.seq_len)])
            df = pd.DataFrame(x, dtype=pd.UInt16Dtype(), columns=["n"])
            loader.train_tokenizer([df], min_frequency=2)
            orig = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9], dtype=np.uint16)
            encoded = loader.encode(orig)
            decoded = loader.decode(encoded)
            self.assertTrue(np.array_equal(orig, decoded))

    def test_get_reg_widths(self):
        loader = RegDataset(FakeArgs())
        results = loader.get_reg_widths(
            [
                pd.DataFrame([{"reg": 1, "val": 7}]),
                pd.DataFrame([{"reg": 3, "val": 256}]),
                pd.DataFrame([{"reg": 24, "val": 1024}]),
            ]
        )
        self.assertEqual(results, {24: 2, 1: 1, 3: 2})

    def test_seq_mapper(self):
        s = SeqMapper(2)
        s.add([1, 2, 3, 4])
        s.add([8, 9, 10, 11, 12, 13, 14])
        s.add([99, 100, 101])
        self.assertEqual(len(s), 8)
        results = [tuple(x.tolist() for x in s[i]) for i in range(len(s))]
        self.assertEqual(
            results,
            [
                ([1, 2], [2, 3]),
                ([2, 3], [3, 4]),
                ([8, 9], [9, 10]),
                ([9, 10], [10, 11]),
                ([10, 11], [11, 12]),
                ([11, 12], [12, 13]),
                ([12, 13], [13, 14]),
                ([99, 100], [100, 101]),
            ],
        )

    def test_make_tokens(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1, "diff": 1},
                {"reg": 1, "val": 1, "diff": 1},
                {"reg": 1, "val": 2, "diff": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        tokens_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1, "diff": 1, "n": 0},
                {"reg": 1, "val": 2, "diff": 1, "n": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._make_tokens([test_df]).astype(MODEL_PDTYPE)
        self.assertTrue(tokens_df.equals(result_df), result_df)

    def test_highbitmask(self):
        loader = RegDataset(FakeArgs())
        self.assertEqual(loader.highbitmask(7), 128)
        self.assertEqual(loader.highbitmask(4), 240)
        self.assertEqual(loader.highbitmask(1), 254)

    def test_maskregbits(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 255},
                {"reg": 1, "val": 128},
            ]
        )
        loader._maskregbits(test_df, 1, 1)
        mask_df = pd.DataFrame(
            [
                {"reg": 1, "val": 254},
                {"reg": 1, "val": 128},
            ]
        )
        self.assertTrue(mask_df.equals(test_df))

    def test_squeeze_changes(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 1, "irq": 1, "reg": 1, "val": 1},
                {"clock": 2, "irq": 2, "reg": 1, "val": 1},
                {"clock": 3, "irq": 3, "reg": 2, "val": 1},
                {"clock": 4, "irq": 4, "reg": 2, "val": 2},
            ]
        )
        squeeze_df = pd.DataFrame(
            [
                {"clock": 1, "irq": 1, "reg": 1, "val": 1},
                {"clock": 3, "irq": 3, "reg": 2, "val": 1},
                {"clock": 4, "irq": 4, "reg": 2, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._squeeze_changes(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(squeeze_df.equals(result), result)

    def test_combine_reg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"diff": 8, "reg": 1, "val": 1},
                {"diff": 8, "reg": 1, "val": 1},
                {"diff": 8, "reg": 2, "val": 1},
                {"diff": 8, "reg": 2, "val": 2},
                {"diff": 8, "reg": 1, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        combine_df = pd.DataFrame(
            [
                {"diff": 8, "reg": 1, "val": 0},
                {"diff": 8, "reg": 1, "val": 256},
                {"diff": 8, "reg": 1, "val": 514},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 16, bits=1).astype(MODEL_PDTYPE)
        self.assertTrue(combine_df.equals(result_df), result_df)

    def test_combine_vreg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 64, "reg": 2, "val": 2},
                {"clock": 80, "reg": 4, "val": 4},
            ],
            dtype=pd.Int64Dtype(),
        )
        combine_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 64, "reg": 0, "val": (2 << (2 * 8)) + 1},
                {"clock": 80, "reg": 0, "val": (2 << (2 * 8)) + 1 + (4 << (4 * 8))},
            ],
            dtype=pd.Int64Dtype(),
        )
        self.assertTrue(
            combine_df.equals(loader._combine_vreg(test_df, 0, dtype=pd.Int64Dtype()))
        )

    def test_rotate_voice_augment(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 12, "reg": 11, "val": 2},
                {"clock": 16, "reg": 23, "val": 1 + 4},
                {"clock": 32, "reg": 7, "val": 2},
                {"clock": 64, "reg": 14, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        rotate_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 12, "reg": 11, "val": 2},
                {"clock": 16, "reg": 23, "val": 1 + 4},
                {"clock": 32, "reg": 7, "val": 2},
                {"clock": 64, "reg": 14, "val": 3},
                {"clock": 0, "reg": 7, "val": 1},
                {"clock": 8, "reg": 11, "val": 1},
                {"clock": 12, "reg": 18, "val": 2},
                {"clock": 16, "reg": 23, "val": 2 + 1},
                {"clock": 32, "reg": 14, "val": 2},
                {"clock": 64, "reg": 0, "val": 3},
                {"clock": 0, "reg": 14, "val": 1},
                {"clock": 8, "reg": 18, "val": 1},
                {"clock": 12, "reg": 4, "val": 2},
                {"clock": 16, "reg": 23, "val": 4 + 2},
                {"clock": 32, "reg": 0, "val": 2},
                {"clock": 64, "reg": 7, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = pd.concat(loader._rotate_voice_augment(test_df)).reset_index(
            drop=True
        )
        self.assertTrue(rotate_df.equals(result_df), result_df)

    def test_norm_voice_reg_order(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 21, "val": 9},
                {"reg": 1, "val": 1},
                {"reg": 0, "val": 1},
                {"reg": 8, "val": 1},
                {"reg": 11, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 16, "val": 2},
                {"reg": 7, "val": 1},
                {"reg": 4, "val": 1},
                {"reg": -99, "val": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        test_df["diff"] = 8
        test_df = test_df.astype(MODEL_PDTYPE)
        norm_df = pd.DataFrame(
            [
                {"reg": 21, "val": 8, "diff": 8},
                {"reg": 22, "val": 0, "diff": 8},
                {"reg": 0, "val": 0, "diff": 8},
                {"reg": 1, "val": 1, "diff": 8},
                {"reg": 11, "val": 1, "diff": 8},
                {"reg": 4, "val": 2, "diff": 8},
                {"reg": 16, "val": 0, "diff": 8},
                {"reg": 17, "val": 0, "diff": 8},
                {"reg": 7, "val": 0, "diff": 8},
                {"reg": 8, "val": 1, "diff": 8},
                {"reg": 4, "val": 1, "diff": 8},
                {"reg": -99, "val": 0, "diff": 8},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._norm_voice_reg_order(test_df).reset_index(drop=True)
        self.assertTrue(result_df.equals(norm_df), result_df)

    def test_norm_reg_order(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 21, "val": 9},
                {"reg": 0, "val": 1},
                {"reg": 11, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 16, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": -99, "val": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        norm_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": 11, "val": 1},
                {"reg": 16, "val": 2},
                {"reg": 21, "val": 9},
                {"reg": -99, "val": 0},
                {"reg": 0, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": 16, "val": 2},
                {"reg": 11, "val": 1},
                {"reg": 21, "val": 9},
                {"reg": -99, "val": 0},
                {"reg": 11, "val": 1},
                {"reg": 0, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": 16, "val": 2},
                {"reg": 21, "val": 9},
                {"reg": -99, "val": 0},
                {"reg": 11, "val": 1},
                {"reg": 16, "val": 2},
                {"reg": 0, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": 21, "val": 9},
                {"reg": -99, "val": 0},
                {"reg": 16, "val": 2},
                {"reg": 0, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": 11, "val": 1},
                {"reg": 21, "val": 9},
                {"reg": -99, "val": 0},
                {"reg": 16, "val": 2},
                {"reg": 11, "val": 1},
                {"reg": 0, "val": 1},
                {"reg": 4, "val": 2},
                {"reg": 4, "val": 1},
                {"reg": 21, "val": 9},
                {"reg": -99, "val": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = pd.concat(loader._norm_reg_order(test_df)).reset_index(drop=True)
        self.assertTrue(result_df.equals(norm_df), result_df)

    def test_unicode(self):
        loader = RegDataset(FakeArgs())
        x = np.array([65, 0, 66, 67, 68, 69])
        y = loader.decode_unicode(loader.encode_unicode(x))
        self.assertTrue(np.array_equal(x, y))

    def test_derange_voiceorder(self):
        loader = RegDataset(FakeArgs())
        self.assertEqual([[0, 1, 2], (1, 2, 0), (2, 0, 1)], loader.derange_voiceorder())
