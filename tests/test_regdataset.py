import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr.regdataset import SeqMapper, RegDataset, MODEL_PDTYPE, VOICES, FRAME_REG
from preframr.stfconstants import UNICODE_BASE


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
            self.assertTrue(np.array_equal(orig, decoded), (orig, decoded))

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

    def test_simplfy_ctrl(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 4, "val": 33 + 2**2},
                {"reg": 4, "val": 17 + 2**2},
                {"reg": 4, "val": 33 + 2**1},
                {"reg": 4, "val": 0 + 2**1},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._simplify_ctrl(test_df).astype(dtype=MODEL_PDTYPE)
        expected_df = pd.DataFrame(
            [
                {"reg": 4, "val": 33},
                {"reg": 4, "val": 17 + 2**2},
                {"reg": 4, "val": 33 + 2**1},
                {"reg": 4, "val": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        self.assertTrue(expected_df.equals(result_df))

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

    def test_squeeze_frames(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 1, "reg": 1, "val": 99},
                {"clock": 2, "reg": 1, "val": 2},
                {"clock": 3, "reg": 2, "val": 2},
                {"clock": 4, "reg": 2, "val": 3},
                {"clock": 5, "reg": 4, "val": 2},
                {"clock": 6, "reg": 3, "val": 4},
                {"clock": 7, "reg": 1, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 9, "reg": 4, "val": 2},
                {"clock": 10, "reg": FRAME_REG, "val": 0},
            ]
        )
        squeeze_df = pd.DataFrame(
            [
                {"clock": 4, "reg": 2, "val": 3},
                {"clock": 5, "reg": 4, "val": 2},
                {"clock": 6, "reg": 3, "val": 4},
                {"clock": 7, "reg": 1, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 9, "reg": 4, "val": 2},
                {"clock": 10, "reg": FRAME_REG, "val": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._squeeze_frames(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(squeeze_df.equals(result), result)

    def test_combine_reg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1},
                {"reg": 1, "val": 1},
                {"reg": 2, "val": 1},
                {"reg": 2, "val": 2},
                {"reg": 1, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        test_df["diff"] = 8
        test_df["clock"] = test_df["diff"].cumsum()
        combine_df = pd.DataFrame(
            [
                {"reg": 1, "val": 0, "diff": 8, "clock": 8},
                {"reg": 1, "val": 256, "diff": 8, "clock": 24},
                {"reg": 1, "val": 514, "diff": 8, "clock": 40},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 16, bits=1).astype(MODEL_PDTYPE)
        self.assertTrue(combine_df.equals(result_df), result_df)

    def test_norm_pr_order(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 3},
                {"reg": 7, "val": 2},
                {"reg": 14, "val": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        norm_df = pd.DataFrame(
            [
                {"reg": 14, "val": 1},
                {"reg": 7, "val": 2},
                {"reg": 0, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._norm_pr_order(test_df)
        self.assertTrue(norm_df.equals(result_df), result_df)

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
        result_df = pd.concat(
            loader._rotate_voice_augment(test_df, VOICES)
        ).reset_index(drop=True)
        self.assertTrue(rotate_df.equals(result_df), result_df)

    def test_unicode(self):
        loader = RegDataset(FakeArgs())
        x = np.array([65, 0, 66, 67, 68, 69], dtype=np.uint16)
        y = loader.decode_unicode(loader.encode_unicode(x))
        self.assertTrue(np.array_equal(x, y), (x, y))
        for i in range(10):
            x = [i for i in range(65536 - UNICODE_BASE)]
            random.shuffle(x)
            x = np.array(x, dtype=np.uint16)
            y = loader.decode_unicode(loader.encode_unicode(x))
            self.assertTrue(np.array_equal(x, y), (x, y))

    def test_derange_voiceorder(self):
        loader = RegDataset(FakeArgs())
        self.assertEqual([[0, 1, 2], (1, 2, 0), (2, 0, 1)], loader.derange_voiceorder())

    def test_add_frame_reg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1, "irq": 0},
                {"clock": 256, "reg": 4, "val": 1, "irq": 0},
                {"clock": 32768, "reg": 11, "val": 2, "irq": 19000},
                {"clock": 32768 + 8, "reg": 23, "val": 1 + 4, "irq": 19000},
                {"clock": 32768 + 16, "reg": 7, "val": 2, "irq": 19000},
                {"clock": 32768 + 32, "reg": 14, "val": 3, "irq": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        frame_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1, "diff": 32},
                {"reg": 4, "val": 1, "diff": 32},
                {"reg": -99, "val": 0, "diff": 19000},
                {"reg": 11, "val": 2, "diff": 32},
                {"reg": 23, "val": 5, "diff": 32},
                {"reg": 7, "val": 2, "diff": 32},
                {"reg": 14, "val": 3, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        irq, result_df = loader._add_frame_reg(test_df, 512)
        self.assertEqual(irq, 19000)
        self.assertTrue(frame_df.equals(result_df))
