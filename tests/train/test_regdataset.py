"""Tests for the BACC ``RegDataset`` adapter (thin layer over ``preframr.corpus.Corpus`` + per-subset ``BlockMapper``s): the torch-Dataset protocol, sampler/loader helpers, ``getseq`` routing, and ``get_prompt`` slicing (corpus construction is covered in ``tests/test_corpus.py``)."""

import argparse
import logging
import unittest

import numpy as np
import torch
from preframr_tokens import VOCAB

from preframr.corpus import SeqMeta
from preframr.train.regdataset import (
    LowMemoryRandomSampler,
    RegDataset,
    _get_loader,
    get_prompt,
    get_val_loader,
)


def _args(**kw):
    defaults = dict(
        manifest_arg="",
        manifest="",
        eval_manifest="",
        sid_root="",
        songlengths="",
        seq_len=8,
        block_stride=None,
        max_files=8,
        predict_set="train",
        shuffle=0,
        batch_size=2,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestRegDataset(unittest.TestCase):
    def test_init_fixed_vocab(self):
        ds = RegDataset(_args(), logger=logging)
        self.assertEqual(ds.n_vocab, VOCAB + 1)
        self.assertEqual(ds.n_words, VOCAB + 1)
        self.assertEqual(ds.reg_widths, {})
        self.assertIsNotNone(ds.block_mapper)
        self.assertIsNotNone(ds.val_block_mapper)
        self.assertEqual(len(ds), 0)

    def test_getitem_empty_raises(self):
        ds = RegDataset(_args(), logger=logging)
        with self.assertRaises(IndexError):
            _ = ds[0]

    def test_getseq_returns_block_and_meta(self):
        ds = RegDataset(_args(), logger=logging)
        block = np.zeros(9, dtype=np.int16)
        block[3] = 7
        meta = SeqMeta(df_file="x.dump.parquet", irq=19656.0, subtune=0)
        ds.block_mapper.block_metas = [("p", meta, 1)]
        ds.block_mapper.get_block = lambda rotation_i, block_j: block
        seq, out_meta = ds.getseq(0, block_j=0)
        self.assertEqual(out_meta.irq, 19656.0)
        self.assertEqual(int(seq[3]), 7)


class TestSampler(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(LowMemoryRandomSampler(list(range(10)), 5)), 5)

    def test_iter_in_range(self):
        out = list(LowMemoryRandomSampler(list(range(10)), 20))
        self.assertEqual(len(out), 20)
        self.assertTrue(all(0 <= i < 10 for i in out))


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


class TestLoaders(unittest.TestCase):
    def test_sequential_when_no_shuffle(self):
        loader = _get_loader(_args(shuffle=0), _ListDataset(8))
        self.assertIsInstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_random_when_shuffle(self):
        loader = _get_loader(_args(shuffle=2.0), _ListDataset(5))
        self.assertIsInstance(loader.sampler, torch.utils.data.RandomSampler)
        self.assertEqual(loader.batch_size, 2)

    def test_get_val_loader_none_without_eval(self):
        loader, names = get_val_loader(_args(eval_manifest=""), RegDataset(_args()))
        self.assertIsNone(loader)
        self.assertEqual(names, [])


class TestGetPrompt(unittest.TestCase):
    def _dataset(self, seq, meta):
        class FakeDataset:
            def getseq(self, i, block_j=0):  # pylint: disable=unused-argument
                return torch.from_numpy(seq.astype(np.int64)), meta

        return FakeDataset()

    def test_slices_prompt_and_returns_meta(self):
        seq = np.arange(1, 11, dtype=np.int16)
        meta = SeqMeta(df_file="t.dump.parquet", irq=19000.0, subtune=0)
        ds = self._dataset(seq, meta)
        prompt_args = argparse.Namespace(
            start_seq=0, start_block=0, max_seq_len=6, prompt_seq_len=4
        )
        irq, n, prompt, prompt_compare, out_meta = get_prompt(prompt_args, ds, logging)
        self.assertEqual(irq, 19000.0)
        self.assertEqual(n, 2)
        self.assertEqual(tuple(prompt.shape), (1, 4))
        self.assertEqual(len(prompt_compare), 6)
        self.assertIs(out_meta, meta)

    def test_max_seq_len_too_short_raises(self):
        seq = np.arange(1, 5, dtype=np.int16)
        meta = SeqMeta(df_file="t.dump.parquet", irq=19000.0, subtune=0)
        ds = self._dataset(seq, meta)
        prompt_args = argparse.Namespace(
            start_seq=0, start_block=0, max_seq_len=4, prompt_seq_len=4
        )
        with self.assertRaises(ValueError):
            get_prompt(prompt_args, ds, logging)


if __name__ == "__main__":
    unittest.main()
