"""Coverage tests for ``preframr.regdataset`` helpers that don't need
a full corpus. Targets:
  * ``LowMemoryRandomSampler`` iter / len.
  * ``_get_loader`` / ``get_val_loader`` branches.
  * ``materialize_block_array`` with empty input.
"""

import argparse
import os
import unittest

import numpy as np
import pandas as pd
import torch

from preframr.train.regdataset import (
    LowMemoryRandomSampler,
    _get_loader,
    get_val_loader,
)
from preframr_tokens.blocks import iter_voiced_blocks, materialize_block_array


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


class TestLowMemoryRandomSampler(unittest.TestCase):
    def test_iter_yields_num_samples(self):
        ds = _ListDataset(10)
        sampler = LowMemoryRandomSampler(ds, num_samples=5)
        out = list(sampler)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(0 <= i < 10 for i in out))

    def test_len_matches_num_samples(self):
        ds = _ListDataset(3)
        sampler = LowMemoryRandomSampler(ds, num_samples=7)
        self.assertEqual(len(sampler), 7)


class TestGetLoader(unittest.TestCase):
    def test_shuffle_branch(self):
        args = argparse.Namespace(shuffle=2.0, batch_size=2, eval_reglogs="")
        ds = _ListDataset(5)
        loader = _get_loader(args, ds)
        self.assertEqual(len(loader.sampler), 10)
        self.assertEqual(loader.batch_size, 2)

    def test_no_shuffle_branch(self):
        args = argparse.Namespace(shuffle=0, batch_size=4, eval_reglogs="")
        ds = _ListDataset(8)
        loader = _get_loader(args, ds)
        self.assertIsInstance(loader.sampler, torch.utils.data.SequentialSampler)


class TestGetValLoader(unittest.TestCase):
    def test_returns_none_without_eval_reglogs(self):
        class _DS:
            val_block_mappers = {}

        args = argparse.Namespace(eval_reglogs="")
        loader, names = get_val_loader(args, _DS())
        self.assertIsNone(loader)
        self.assertEqual(names, [])

    def test_returns_none_when_mapper_empty(self):
        class _Mapper:
            def __len__(self):
                return 0

        class _DS:
            val_block_mappers = {"val": _Mapper()}

        args = argparse.Namespace(eval_reglogs="x")
        loader, names = get_val_loader(args, _DS())
        self.assertIsNone(loader)
        self.assertEqual(names, [])


class TestMaterializeBlockArray(unittest.TestCase):
    def test_empty_input_returns_zero_shape(self):
        class _Tokenizer:
            tokens = pd.DataFrame(columns=["op", "reg", "subreg", "val", "n"])

            def merge_token_df(self, *_):
                return pd.DataFrame()

            def encode(self, n, dtype=None):
                return np.array(n, dtype=dtype or np.int16)

        class _Parser:
            args = argparse.Namespace()

        empty_df = pd.DataFrame(columns=["reg", "val", "op", "subreg", "diff"])
        out = materialize_block_array(
            tokenizer=_Tokenizer(),
            raw_df=empty_df,
            seq_len=8,
            parser=_Parser(),
            reg_widths={},
        )
        self.assertEqual(out.shape, (0, 9))


class TestIterVoicedBlocks(unittest.TestCase):
    def test_empty_input_yields_nothing(self):
        class _Parser:
            args = argparse.Namespace()

        empty_df = pd.DataFrame(columns=["reg", "val", "op", "subreg", "diff"])
        out = list(iter_voiced_blocks(empty_df, 8, _Parser(), {}, stride=4))
        self.assertEqual(out, [])


class TestRegDatasetBasics(unittest.TestCase):
    def _args(self):
        return argparse.Namespace(
            seq_len=8,
            tkvocab=0,
            tokenizer="unigram",
            tkmodel=None,
            predict_set="train",
            eval_reglogs="",
            require_pq=False,
            max_files=8,
            shuffle=0,
            batch_size=2,
        )

    def test_init_sets_up_state(self):
        from preframr.train.regdataset import RegDataset

        ds = RegDataset(self._args())
        self.assertEqual(ds.n_vocab, 0)
        self.assertEqual(ds.reg_widths, {})
        self.assertIsNotNone(ds.block_mapper)
        self.assertIsNotNone(ds.val_block_mapper)
        self.assertIsNotNone(ds.tokenizer)
        self.assertEqual(len(ds), 0)

    def test_preload_with_tokens_short_circuits(self):
        from preframr.train.regdataset import RegDataset
        from preframr_tokens.stfconstants import MODEL_PDTYPE, SET_OP

        ds = RegDataset(self._args())
        tokens = pd.DataFrame(
            [
                {"op": SET_OP, "reg": -1, "subreg": -1, "val": 0, "n": 0},
                {"op": SET_OP, "reg": 0, "subreg": -1, "val": 5, "n": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        ds.preload(tokens=tokens, tkmodel=None)
        self.assertIsNotNone(ds.tokenizer.tokens)
        self.assertEqual(len(ds.tokenizer.tokens), 2)

    def test_load_dfs_requires_reglogs_or_dump_files(self):
        from preframr.train.regdataset import RegDataset

        ds = RegDataset(self._args())
        with self.assertRaises(ValueError):
            list(ds.corpus.load_dfs(reglogs="", dump_files=None))

    def test_predict_load_aliases_val_to_eval_a(self):
        import json
        import tempfile
        from preframr.train.regdataset import RegDataset
        from preframr_tokens.blocks import reg_widths_path as _reg_widths_path
        from preframr_tokens.stfconstants import DUMP_SUFFIX

        with tempfile.TemporaryDirectory() as tmp:
            train_dump = os.path.join(tmp, f"train_x{DUMP_SUFFIX}")
            eval_dump = os.path.join(tmp, f"eval_a_x{DUMP_SUFFIX}")
            blocks_path = eval_dump.replace(DUMP_SUFFIX, ".0.blocks.npy")
            np.save(blocks_path, np.zeros((1, 9), dtype=np.int16))
            df_map_csv = os.path.join(tmp, "df-map.csv")
            pd.DataFrame(
                [
                    {
                        "dump_file": train_dump,
                        "kind": "train",
                        "irq": 19656,
                        "n_rotations": 1,
                    },
                    {
                        "dump_file": eval_dump,
                        "kind": "eval_a",
                        "irq": 19656,
                        "n_rotations": 1,
                    },
                ]
            ).to_csv(df_map_csv, index=False)
            with open(_reg_widths_path(df_map_csv), "w") as f:
                json.dump({"0": 1}, f)

            args = self._args()
            args.df_map_csv = df_map_csv
            args.predict_set = "val"
            args.start_seq = 0
            args.tkvocab = 0
            ds = RegDataset(args)
            ds.tokenizer.tokens = {"n": [0]}
            ds.predict_load()
            self.assertEqual(args.predict_set, "eval_a")
            self.assertIn("eval_a", ds.val_block_mappers)

    def test_predict_load_raises_when_no_held_out_kinds(self):
        import tempfile
        from preframr.train.regdataset import RegDataset
        from preframr_tokens.stfconstants import DUMP_SUFFIX

        with tempfile.TemporaryDirectory() as tmp:
            train_dump = os.path.join(tmp, f"train_x{DUMP_SUFFIX}")
            df_map_csv = os.path.join(tmp, "df-map.csv")
            pd.DataFrame(
                [
                    {
                        "dump_file": train_dump,
                        "kind": "train",
                        "irq": 19656,
                        "n_rotations": 1,
                    },
                ]
            ).to_csv(df_map_csv, index=False)

            args = self._args()
            args.df_map_csv = df_map_csv
            args.predict_set = "val"
            args.start_seq = 0
            args.tkvocab = 0
            ds = RegDataset(args)
            ds.tokenizer.tokens = {"n": [0]}
            with self.assertRaisesRegex(ValueError, "no 'val' files"):
                ds.predict_load()

    def test_getseq_routes_via_predict_set(self):
        from preframr.train.regdataset import RegDataset
        from preframr_tokens.blocks import SeqMeta

        ds = RegDataset(self._args())
        block = np.zeros(9, dtype=np.int16)
        block[3] = 42
        meta = SeqMeta(irq=19656, df_file="synthetic", i=0)
        ds.block_mapper.block_metas = [("p", meta, 1)]
        ds.block_mapper.get_block = lambda rotation_i, block_j: block
        ds.block_mapper._n = 1
        out, out_meta = ds.getseq(0, block_j=0)
        self.assertEqual(out_meta.irq, 19656)
        self.assertEqual(out[3].item(), 42)


if __name__ == "__main__":
    unittest.main()
