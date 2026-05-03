import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr.reglogparser import (
    RegLogParser,
    prepare_df_for_audio,
)
from preframr.stfconstants import (
    DELAY_REG,
    FC_LO_REG,
    FILTER_REG,
    FRAME_REG,
    MAX_REG,
    MIN_DIFF,
    MODEL_PDTYPE,
    VOICE_REG_SIZE,
    VOICES,
    VOICE_REG,
    DIFF_OP,
    FLIP_OP,
    REPEAT_OP,
    SET_OP,
)


class FakeArgs:
    def __init__(
        self,
        seq_len=128,
        tkvocab=0,
        diffq=64,
        tkmodel=None,
        cents=10,
        min_irq=0,
        max_irq=100000,
    ):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.max_files = 1
        self.cents = cents
        self.min_irq = min_irq
        self.max_irq = max_irq


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
                {"reg": 1, "val": 1, "diff": 8, "clock": 8},
                {"reg": 1, "val": 257, "diff": 8, "clock": 24},
                {"reg": 1, "val": 514, "diff": 8, "clock": 40},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 16, bits=0).astype(MODEL_PDTYPE)
        self.assertTrue(combine_df.equals(result_df), result_df)
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
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 3},
                {"reg": 2, "val": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        test_df["diff"] = 8
        test_df["clock"] = test_df["diff"].cumsum()
        combine_df = pd.DataFrame(
            [
                {"reg": 1, "val": 11, "diff": 8, "clock": 16},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 32, bits=0, lobits=3).astype(
            MODEL_PDTYPE
        )
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
        irq, result_df = loader._add_frame_reg(test_df, 512, min_irq_prop=0.5)
        self.assertEqual(irq, 19000)
        self.assertTrue(frame_df.equals(result_df))

    def test_last_reg_val_frame(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": 7, "val": 2, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 3, "diff": 32},
                {"reg": 7, "val": 4, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        last_df = pd.DataFrame(
            [
                {"f": 1, "v": 1, "val": 2, "pval": 0},
                {"f": 2, "v": 1, "val": 4, "pval": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = list(loader._last_reg_val_frame(test_df, [0]))[0]
        self.assertTrue(last_df.equals(result_df))

    def test_add_change_regs(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 2, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 3, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 65, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        change_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 1, "diff": 32, "op": REPEAT_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 0, "diff": 32, "op": REPEAT_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 65, "diff": 32, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._add_change_regs(
            test_df, opcodes=[DIFF_OP, FLIP_OP, REPEAT_OP]
        ).astype(MODEL_PDTYPE)
        self.assertTrue(change_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 0, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 65, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        change_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 1, "diff": 32, "op": FLIP_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 0, "diff": 32, "op": FLIP_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 65, "diff": 32, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._add_change_regs(
            test_df, opcodes=[DIFF_OP, FLIP_OP, REPEAT_OP]
        ).astype(MODEL_PDTYPE)
        self.assertTrue(change_df.equals(result_df))

    def test_norm_pr_order(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 7, "val": 255, "diff": 32, "op": 0},
                {"reg": 0, "val": 2, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 0},
                {"reg": 14, "val": 3, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        order_df = pd.DataFrame(
            [
                {"reg": 0, "val": 2, "diff": 32, "op": 0},
                {"reg": 7, "val": 255, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 0},
                {"reg": 14, "val": 3, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._norm_pr_order(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(order_df.equals(result_df))

    def test_add_voice_reg(self):
        meta_freq_bits = 6
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 1},
                {"reg": 14, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 1},
                {"reg": 14, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        voice_df = pd.DataFrame(
            [
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 2, "diff": 32, "op": 1},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 57, "diff": 19000, "op": 0},
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 2, "diff": 32, "op": 1},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 1, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._add_voice_reg(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(voice_df.equals(result_df))

    def test_expand_ops(self):
        loader = RegLogParser()
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 0,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        expand_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._expand_ops(test_df, strict=True).astype(MODEL_PDTYPE)
        self.assertTrue(expand_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
                {
                    "reg": DELAY_REG,
                    "subreg": -1,
                    "val": 3,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 0,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        expand_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._expand_ops(test_df, strict=True).astype(MODEL_PDTYPE)
        self.assertTrue(expand_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": DELAY_REG,
                    "subreg": -1,
                    "val": 2,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": 9,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        expand_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": 7, "val": 2, "diff": 32, "description": 0},
                {"reg": 9, "val": 2, "diff": 32, "description": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._expand_ops(test_df, strict=True).astype(MODEL_PDTYPE)
        self.assertTrue(expand_df.equals(result_df))

    def test_consolidate_frames(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        consolidate_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 3, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._consolidate_frames(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(consolidate_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        consolidate_df = pd.DataFrame(
            [
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._consolidate_frames(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(consolidate_df.equals(result_df))

    def test_consolidate_frames_adjacent_delays(self):
        # Two adjacent DELAY_REGs trigger the merge-consecutive-delays branch (line 838)
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 19000},
                {"reg": DELAY_REG, "val": 3, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._consolidate_frames(test_df)
        # Adjacent DELAY_REGs should be merged into one
        self.assertEqual(len(result_df[result_df["reg"] == DELAY_REG]), 1)

    def test_remove_voice_reg(self):
        loader = RegLogParser(FakeArgs())
        # No VOICE_REG: identity
        test_df = pd.DataFrame([{"reg": 0, "val": 1, "diff": 32}], dtype=MODEL_PDTYPE)
        result_df, result_widths = loader._remove_voice_reg(test_df, {})
        self.assertTrue(test_df.equals(result_df))

        # With VOICE_REG rows: they should be expanded and removed
        # FRAME_REG val=57 encodes voices 0,1,2 packed as 2-bit fields
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 57, "diff": 19000},
                {"reg": VOICE_REG, "val": 0, "diff": 32},
                {"reg": 0, "val": 100, "diff": 32},
                {"reg": VOICE_REG, "val": 0, "diff": 32},
                {"reg": 0, "val": 200, "diff": 32},
                {"reg": VOICE_REG, "val": 0, "diff": 32},
                {"reg": 0, "val": 150, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        reg_widths = {0: 8}
        result_df, result_widths = loader._remove_voice_reg(test_df, reg_widths)
        self.assertFalse((result_df["reg"] == VOICE_REG).any())
        # reg_widths expanded for each voice offset
        for v in range(VOICES):
            self.assertIn(v * VOICE_REG_SIZE, result_widths)

    def test_reset_diffs(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        # irq=None: infer from first FRAME_REG diff
        result = loader._reset_diffs(test_df, None, 1)
        self.assertIn("delay", result.columns)
        delay_diff = result.loc[result["reg"] == DELAY_REG, "diff"].iloc[0]
        self.assertEqual(delay_diff, 2 * 19000)

        # Explicit irq gives same result
        result2 = loader._reset_diffs(test_df, 19000, 1)
        self.assertTrue(result.equals(result2))

    def test_expand_ops_diff_op(self):
        loader = RegLogParser()
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 5,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 3,
                    "diff": 32,
                    "op": DIFF_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._expand_ops(test_df, strict=True)
        reg7_vals = result[result["reg"] == 7]["val"].tolist()
        self.assertEqual(reg7_vals, [5, 8])

    def test_expand_ops_repeat_op(self):
        loader = RegLogParser()
        # REPEAT_OP val!=0 sets pending repeat; apply_ops at frame end applies it each frame
        # REPEAT_OP val==0 terminates the repeat
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": REPEAT_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 0,
                    "diff": 32,
                    "op": REPEAT_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._expand_ops(test_df, strict=True)
        reg7_vals = result[result["reg"] == 7]["val"].tolist()
        # Frame 1: SET val=1 -> last_val=1
        # Frame 2: REPEAT begin val=2 -> apply_ops -> last_val+=2=3
        # Frame 3: REPEAT end val=0 -> last_val+=2=5, written out
        self.assertEqual(reg7_vals[0], 1)
        self.assertGreater(reg7_vals[-1], 1)

    def test_expand_ops_subreg(self):
        loader = RegLogParser()
        # subreg==0 stores low nibble; subreg==1 stores high nibble
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": 0,
                    "val": 5,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": 1,
                    "val": 3,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._expand_ops(test_df, strict=True)
        reg7_vals = result[result["reg"] == 7]["val"].tolist()
        # subreg=0 emits a write with the new low nibble (high preserved=0):
        # last_val[7] becomes 5. subreg=1 then emits with combined byte 53.
        # The redesigned SetDecoder emits one write per subreg row -- the
        # SID receives the intermediate state, matching the behavior of two
        # back-to-back full-byte SETs (5, then 53).
        self.assertEqual(reg7_vals, [5, 48 + 5])

    def test_prepare_df_for_audio(self):
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP, "subreg": -1},
                {"reg": 7, "val": 1, "diff": 32, "op": SET_OP, "subreg": -1},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP, "subreg": -1},
                {"reg": 7, "val": 2, "diff": 32, "op": SET_OP, "subreg": -1},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df, _ = prepare_df_for_audio(test_df, {}, irq=19000, sidq=1)
        self.assertIn("delay", result_df.columns)

        # With prompt_len < len(df): description column marks rows after prompt
        result_df2, _ = prepare_df_for_audio(
            test_df, {}, irq=19000, sidq=1, prompt_len=2
        )
        self.assertTrue((result_df2["description"] > 0).any())

    def test_matcher_methods(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1},
                {"reg": 2, "val": 1},
                {"reg": 4, "val": 1},
                {"reg": 5, "val": 1},
                {"reg": 6, "val": 1},
                {"reg": FC_LO_REG, "val": 1},
                {"reg": FILTER_REG, "val": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        self.assertTrue(loader._freq_match(test_df).any())
        self.assertTrue(loader._pcm_match(test_df).any())
        self.assertTrue(loader._adsr_match(test_df).any())
        self.assertTrue(loader._ad_match(test_df).any())
        self.assertTrue(loader._sr_match(test_df).any())

    def test_read_df(self):
        loader = RegLogParser(FakeArgs())

        with self.assertRaises(ValueError):
            loader._read_df("/nonexistent/path.parquet")

        # Single chip: all rows returned (filtered to reg <= MAX_REG)
        test_df = pd.DataFrame(
            [
                {"clock": 1, "irq": 100, "reg": 0, "val": 1, "chipno": 0},
                {"clock": 2, "irq": 200, "reg": 25, "val": 2, "chipno": 0},
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname = f.name
        try:
            test_df.to_parquet(fname)
            result = loader._read_df(fname)
            self.assertEqual(list(result.columns), ["clock", "irq", "reg", "val"])
            self.assertEqual(len(result), 1)  # reg 25 > MAX_REG filtered out
        finally:
            os.unlink(fname)

        # Multi-chip: only rows with clock < 0 returned
        test_df2 = pd.DataFrame(
            [
                {"clock": -1, "irq": 100, "reg": 0, "val": 1, "chipno": 0},
                {"clock": 2, "irq": 200, "reg": 1, "val": 2, "chipno": 1},
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname2 = f.name
        try:
            test_df2.to_parquet(fname2)
            result2 = loader._read_df(fname2)
            self.assertTrue((result2["clock"] < 0).all())
        finally:
            os.unlink(fname2)

    def test_rotate_filter(self):
        loader = RegLogParser(FakeArgs())
        # r=0: no rotation, val unchanged
        test_df = pd.DataFrame([{"reg": FILTER_REG, "val": 1}], dtype=MODEL_PDTYPE)
        result0 = loader._rotate_filter(test_df.copy(), 0)
        self.assertEqual(result0.loc[result0["reg"] == FILTER_REG, "val"].iloc[0], 1)

        # r=1: exercises the loop body (lines 382-385)
        # wrapbits(1, 3) = 2, so after one rotation val should shift from 1 to 2
        result1 = loader._rotate_filter(test_df.copy(), 1)
        self.assertEqual(result1.loc[result1["reg"] == FILTER_REG, "val"].iloc[0], 2)

    def test_add_frame_reg_no_irqdiff(self):
        # When no irqdiff > diffmax: IndexError caught, irq=0
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1, "irq": 0},
                {"clock": 1, "reg": 4, "val": 1, "irq": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        irq, result_df = loader._add_frame_reg(test_df, 512)
        self.assertEqual(irq, 0)

    def test_cap_delay(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": DELAY_REG, "val": 1000},
                {"reg": DELAY_REG, "val": 30},
                {"reg": DELAY_REG, "val": 5},
                {"reg": 7, "val": 1000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._cap_delay(test_df.copy())
        self.assertEqual(result.iloc[0]["val"], 255)
        # 30 > q^2=25, rounded: (30//5)*5 = 30
        self.assertEqual(result.iloc[1]["val"], 30)
        # 5 <= q^2=25: unchanged
        self.assertEqual(result.iloc[2]["val"], 5)
        # non-delay reg unchanged
        self.assertEqual(result.iloc[3]["val"], 1000)

    def test_split_reg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 512, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._split_reg(test_df, 0)
        # 512 = 2*256 -> lo=0 stays at reg 0, hi=2 goes to reg 1
        self.assertIn(1, result["reg"].values)
        hi_rows = result[result["reg"] == 1]
        self.assertEqual(hi_rows["val"].iloc[0], 2)
        lo_rows = result[result["reg"] == 0]
        self.assertEqual(lo_rows["val"].iloc[0], 0)

    def test_reduce_val_res(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [{"reg": 7, "val": 0b11111111}, {"reg": 8, "val": 0b11111111}],
            dtype=MODEL_PDTYPE,
        )
        result = loader._reduce_val_res(test_df.copy(), reg=7, bits=2)
        # Lower 2 bits cleared: 0b11111100 = 252
        self.assertEqual(result.loc[result["reg"] == 7, "val"].iloc[0], 252)
        # reg 8 unchanged
        self.assertEqual(result.loc[result["reg"] == 8, "val"].iloc[0], 255)

    def test_quantize_freq_to_cents(self):
        loader = RegLogParser(FakeArgs())
        # _vreg_match(0) = {0, 7, 14} for VOICE_REG_SIZE=7, VOICES=3
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 0},
                {"reg": 7, "val": 0},
                {"reg": 8, "val": 100},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._quantize_freq_to_cents(test_df.copy())
        # Non-freq reg (8) unchanged
        self.assertEqual(result.loc[result["reg"] == 8, "val"].iloc[0], 100)
        # Freq regs mapped through fi_map
        self.assertIsInstance(
            result.loc[result["reg"] == 0, "val"].iloc[0], (int, np.integer)
        )

    def test_add_voice_reg_empty_regs(self):
        # When no voice-range registers, returns orig_df unchanged (line 543)
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_voice_reg(test_df)
        self.assertTrue(test_df.equals(result))

    def test_add_voice_reg_zero_false(self):
        # zero_voice_reg=False: VOICE_REG val encodes freq/ctrl metadata
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": SET_OP},
                {"reg": 7, "val": 2, "diff": 32, "op": SET_OP},
                {"reg": 14, "val": 3, "diff": 32, "op": SET_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_voice_reg(test_df, zero_voice_reg=False)
        self.assertIn(VOICE_REG, result["reg"].values)

    def test_simplify_pcm(self):
        loader = RegLogParser(FakeArgs())
        # Voice 0: pcm_reg=2, ctrl_reg=4
        test_df = pd.DataFrame(
            [
                {"reg": 2, "val": 128},
                {"reg": 4, "val": 0b01000001},  # pulse enabled (bit 6)
                {"reg": 4, "val": 0b00000001},  # pulse disabled
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._simplify_pcm(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("reg", result.columns)
        # When pulse enabled, an extra PCM row is inserted before ctrl row
        self.assertGreater(len(result), len(test_df))

    def test_filter_irq(self):
        loader = RegLogParser(FakeArgs(min_irq=1000, max_irq=50000))

        # No irq column -> KeyError -> False
        test_df = pd.DataFrame([{"reg": 0, "val": 1}], dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter_irq(test_df, "test"))

        # irq below min_irq -> False
        test_df = pd.DataFrame([{"reg": 0, "val": 1, "irq": 100}], dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter_irq(test_df, "test"))

        # irq above max_irq -> False
        test_df = pd.DataFrame([{"reg": 0, "val": 1, "irq": 99999}], dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter_irq(test_df, "test"))

        # irq in range -> True
        test_df = pd.DataFrame([{"reg": 0, "val": 1, "irq": 19000}], dtype=MODEL_PDTYPE)
        self.assertTrue(loader._filter_irq(test_df, "test"))

    def test_filter(self):
        loader = RegLogParser(FakeArgs(seq_len=2))

        # No FRAME_REG -> False
        test_df = pd.DataFrame(
            [{"reg": 7, "val": i, "diff": 32} for i in range(10)], dtype=MODEL_PDTYPE
        )
        self.assertFalse(loader._filter(test_df, "test"))

        # Too short (< seq_len * 2 = 4) -> False
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        self.assertFalse(loader._filter(test_df, "test"))

        # Too many vol changes (>= 8 distinct values for reg=24) -> False
        rows = [{"reg": FRAME_REG, "val": 0, "diff": 19000}]
        for i in range(8):
            rows.append({"reg": 24, "val": i, "diff": 32})
        for _ in range(10):
            rows.append({"reg": FRAME_REG, "val": 0, "diff": 19000})
        test_df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter(test_df, "test"))

        # Passing filter
        rows = []
        for _ in range(5):
            rows.append({"reg": FRAME_REG, "val": 0, "diff": 19000})
            rows.append({"reg": 7, "val": 1, "diff": 32})
        test_df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        self.assertTrue(loader._filter(test_df, "test"))

    def test_combine_regs(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 8, "irq": 0, "reg": 0, "val": 1},
                {"clock": 16, "irq": 0, "reg": 1, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._combine_regs(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("reg", result.columns)

    def test_squeeze_frame_regs(self):
        loader = RegLogParser(FakeArgs())
        # Two occurrences of reg=0 in same frame; only last should remain
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 1, "diff": 32},
                {"reg": 0, "val": 2, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._squeeze_frame_regs(test_df)
        reg0_rows = result[result["reg"] == 0]
        self.assertEqual(len(reg0_rows), 1)
        self.assertEqual(reg0_rows["val"].iloc[0], 2)

        # reg >= VOICE_REG_SIZE: uses the non-voice else branch
        test_df2 = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FC_LO_REG, "val": 10, "diff": 32},
                {"reg": FC_LO_REG, "val": 20, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result2 = loader._squeeze_frame_regs(test_df2, regs=[FC_LO_REG])
        fc_rows = result2[result2["reg"] == FC_LO_REG]
        self.assertEqual(len(fc_rows), 1)
        self.assertEqual(fc_rows["val"].iloc[0], 20)

    def test_add_subreg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 4, "val": 0b11110101},
                {"reg": 7, "val": 0b11110101},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_subreg(test_df)
        self.assertIn("subreg", result.columns)
        # reg 4 split into subreg 0 (low nibble=5) and subreg 1 (high nibble=15)
        reg4_rows = result[result["reg"] == 4]
        self.assertEqual(len(reg4_rows), 2)
        self.assertIn(0, reg4_rows["subreg"].values)
        self.assertIn(1, reg4_rows["subreg"].values)
        # reg 7 not in the split list: subreg=-1
        reg7_rows = result[result["reg"] == 7]
        self.assertEqual(reg7_rows["subreg"].iloc[0], -1)

    def test_state_df(self):
        loader = RegLogParser(FakeArgs())
        # Build a minimal tokens DataFrame: n is the token index used for merging.
        # Include a normal reg, FRAME_REG, and a reg < -MAX_REG token.
        tokens = pd.DataFrame(
            [
                {"n": 0, "reg": 7, "val": 1, "op": SET_OP, "subreg": -1},
                {"n": 1, "reg": FRAME_REG, "val": 0, "op": SET_OP, "subreg": -1},
                {"n": 2, "reg": -MAX_REG - 1, "val": 0, "op": SET_OP, "subreg": -1},
            ],
            dtype=MODEL_PDTYPE,
        )

        class FakeTokenizer:
            pass

        class FakeDataset:
            pass

        tokenizer = FakeTokenizer()
        tokenizer.tokens = tokens
        dataset = FakeDataset()
        dataset.tokenizer = tokenizer

        irq = 19000
        result = loader._state_df([0, 1, 2], dataset, irq)

        # All three states are merged in
        self.assertEqual(len(result), 3)
        # Normal reg gets MIN_DIFF
        self.assertEqual(result.loc[result["reg"] == 7, "diff"].iloc[0], MIN_DIFF)
        # FRAME_REG gets irq
        self.assertEqual(result.loc[result["reg"] == FRAME_REG, "diff"].iloc[0], irq)
        # reg < -MAX_REG gets diff=0
        self.assertEqual(result.loc[result["reg"] == -MAX_REG - 1, "diff"].iloc[0], 0)

    def test_remove_voice_reg_is_inverse_of_add_voice_reg(self):
        loader = RegLogParser(FakeArgs())
        # Input must start with FRAME_REG so the voiceorder is available before the
        # first voice-register writes.  Each frame has one write per voice (regs 0, 7, 14
        # for VOICE_REG_SIZE=7, VOICES=3).
        orig_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
                {"reg": 0, "val": 100, "diff": 32, "op": SET_OP},
                {"reg": 7, "val": 200, "diff": 32, "op": SET_OP},
                {"reg": 14, "val": 150, "diff": 32, "op": SET_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
                {"reg": 0, "val": 110, "diff": 32, "op": SET_OP},
                {"reg": 7, "val": 210, "diff": 32, "op": SET_OP},
                {"reg": 14, "val": 160, "diff": 32, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )

        # Forward pass: collapse voice-specific regs and add VOICE_REG markers.
        voice_df = loader._add_voice_reg(orig_df)
        self.assertTrue((voice_df["reg"] == VOICE_REG).any())

        # Inverse pass: expand back using the voiceorder stored in FRAME_REG val.
        result_df, _ = loader._remove_voice_reg(voice_df, {})

        # No VOICE_REG markers remain.
        self.assertFalse((result_df["reg"] == VOICE_REG).any())

        # The row count is restored to the original.
        self.assertEqual(len(result_df), len(orig_df))

        # Every non-FRAME_REG row is recovered exactly: register, value, diff, and op.
        orig_regs = orig_df[orig_df["reg"] != FRAME_REG].reset_index(drop=True)
        result_regs = result_df[result_df["reg"] != FRAME_REG].reset_index(drop=True)
        self.assertTrue(orig_regs.equals(result_regs))

        # FRAME_REG rows: diff and op are preserved.  val changes from 0 to the
        # voiceorder encoding produced by _add_voice_reg — that is intentional and
        # is exactly what remove_voice_reg uses to decode voice assignments.
        orig_frame = orig_df[orig_df["reg"] == FRAME_REG][["diff", "op"]].reset_index(
            drop=True
        )
        result_frame = result_df[result_df["reg"] == FRAME_REG][
            ["diff", "op"]
        ].reset_index(drop=True)
        self.assertTrue(orig_frame.equals(result_frame))
