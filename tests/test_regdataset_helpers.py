"""Coverage tests for ``preframr.regdataset`` helpers that don't need
a full corpus. Targets:
  * ``LowMemoryRandomSampler`` iter / len.
  * ``_get_loader`` / ``get_val_loader`` branches.
  * ``materialize_block_array`` with empty input.
  * ``iter_voiced_blocks`` skip path on empty blocks.
"""

import argparse
import unittest

import numpy as np
import pandas as pd
import torch

from preframr.regdataset import (
    LowMemoryRandomSampler,
    _get_loader,
    get_val_loader,
    iter_voiced_blocks,
    materialize_block_array,
)


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
        # All samples in valid range.
        self.assertTrue(all(0 <= i < 10 for i in out))

    def test_len_matches_num_samples(self):
        ds = _ListDataset(3)
        sampler = LowMemoryRandomSampler(ds, num_samples=7)
        self.assertEqual(len(sampler), 7)


class TestGetLoader(unittest.TestCase):
    def test_shuffle_branch(self):
        # shuffle > 0 -> RandomSampler with len = shuffle * len(dataset).
        args = argparse.Namespace(shuffle=2.0, batch_size=2, eval_reglogs="")
        ds = _ListDataset(5)
        loader = _get_loader(args, ds)
        # RandomSampler requested 10 samples (2.0 * 5).
        self.assertEqual(len(loader.sampler), 10)
        self.assertEqual(loader.batch_size, 2)

    def test_no_shuffle_branch(self):
        args = argparse.Namespace(shuffle=0, batch_size=4, eval_reglogs="")
        ds = _ListDataset(8)
        loader = _get_loader(args, ds)
        # Sequential sampler used.
        self.assertIsInstance(loader.sampler, torch.utils.data.SequentialSampler)


class TestGetValLoader(unittest.TestCase):
    def test_returns_none_without_eval_reglogs(self):
        class _DS:
            class _Mapper:
                def __len__(self):
                    return 0

            val_block_mapper = _Mapper()

        args = argparse.Namespace(eval_reglogs="")
        self.assertIsNone(get_val_loader(args, _DS()))

    def test_returns_none_when_mapper_empty(self):
        class _DS:
            class _Mapper:
                def __len__(self):
                    return 0

            val_block_mapper = _Mapper()

        args = argparse.Namespace(eval_reglogs="x")
        self.assertIsNone(get_val_loader(args, _DS()))


class TestMaterializeBlockArray(unittest.TestCase):
    def test_empty_input_returns_zero_shape(self):
        # raw_df with no markers -> iter_voiced_blocks yields nothing.
        # materialize_block_array returns a zero-row array.
        class _Tokenizer:
            tokens = pd.DataFrame(columns=["op", "reg", "subreg", "val", "n"])

            def merge_token_df(self, *_):
                return pd.DataFrame()

            def encode(self, n, dtype=None):
                return np.array(n, dtype=dtype or np.int16)

        class _Parser:
            args = argparse.Namespace()

            def _remove_voice_reg(self, df, _w):
                return df, {}

        empty_df = pd.DataFrame(columns=["reg", "val", "op", "subreg", "diff"])
        out = materialize_block_array(
            tokenizer=_Tokenizer(),
            raw_df=empty_df,
            seq_len=8,
            parser=_Parser(),
            reg_widths={},
        )
        # Shape is (0, seq_len + 1).
        self.assertEqual(out.shape, (0, 9))


class TestIterVoicedBlocks(unittest.TestCase):
    def test_empty_input_yields_nothing(self):
        class _Parser:
            args = argparse.Namespace()

            def _remove_voice_reg(self, df, _w):
                return df, {}

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
        from preframr.regdataset import RegDataset

        ds = RegDataset(self._args())
        self.assertEqual(ds.n_vocab, 0)
        self.assertEqual(ds.reg_widths, {})
        self.assertIsNotNone(ds.block_mapper)
        self.assertIsNotNone(ds.val_block_mapper)
        self.assertIsNotNone(ds.tokenizer)
        self.assertEqual(len(ds), 0)

    def test_load_dfs_requires_reglogs_or_dump_files(self):
        from preframr.regdataset import RegDataset

        ds = RegDataset(self._args())
        with self.assertRaises(ValueError):
            list(ds.load_dfs(reglogs="", dump_files=None))

    def test_getseq_routes_via_predict_set(self):
        from preframr.regdataset import RegDataset
        from preframr.block_mapper import SeqMeta

        ds = RegDataset(self._args())
        # Inject a synthetic block in the train mapper.
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
