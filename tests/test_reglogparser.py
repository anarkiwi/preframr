import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr.reglogparser import RegLogParser
from preframr.stfconstants import FRAME_REG, MODEL_PDTYPE, VOICES


class FakeArgs:
    def __init__(self, seq_len=128, tkvocab=0, diffq=64, tkmodel=None):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.max_files = 1


class TestRegLogParser(unittest.TestCase):
    def test_highbitmask(self):
        loader = RegLogParser(FakeArgs())
        self.assertEqual(loader.highbitmask(7), 128)
        self.assertEqual(loader.highbitmask(4), 240)
        self.assertEqual(loader.highbitmask(1), 254)

    def test_simplfy_ctrl(self):
        loader = RegLogParser(FakeArgs())
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
        loader = RegLogParser(FakeArgs())
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
        loader = RegLogParser(FakeArgs())
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
        loader = RegLogParser(FakeArgs())
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
        loader = RegLogParser(FakeArgs())
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

    def test_rotate_voice_augment(self):
        loader = RegLogParser(FakeArgs())
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

    def test_add_frame_reg(self):
        loader = RegLogParser(FakeArgs())
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
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
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
