"""BACC corpus builder: (.sid, .dump) pairing, windowing, and skip behaviour."""

import argparse
import logging
import os
import shutil
import unittest

import numpy as np

from preframr.corpus import (
    BLOCKS_SUFFIX,
    Corpus,
    _resolve_paths,
    _windows,
    parse_corpus,
)
from preframr.tokenizer import PAD_ID


def _args(reglogs, **kw):
    defaults = dict(
        reglogs=reglogs,
        reglog="",
        eval_reglogs="",
        seq_len=512,
        block_stride=None,
        max_files=0,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestResolvePaths(unittest.TestCase):
    def test_subtune_is_zero_indexed(self):
        sid, subtune, base = _resolve_paths("/a/Tune.1.dump.parquet")
        self.assertEqual(sid, "/a/Tune.sid")
        self.assertEqual(subtune, 0)
        self.assertEqual(base, "/a/Tune.1")

    def test_second_subtune(self):
        sid, subtune, _ = _resolve_paths("/a/Tune.2.dump.parquet")
        self.assertEqual(sid, "/a/Tune.sid")
        self.assertEqual(subtune, 1)

    def test_no_subtune_component(self):
        sid, subtune, _ = _resolve_paths("/a/Tune.dump.parquet")
        self.assertEqual(sid, "/a/Tune.sid")
        self.assertEqual(subtune, 0)


class TestWindows(unittest.TestCase):
    def test_short_stream_one_padded_block(self):
        blocks = _windows(list(range(1, 6)), seq_len=8, stride=8)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 9)
        self.assertEqual(blocks[0][-1], PAD_ID)

    def test_long_stream_multiple_windows(self):
        ids = list(range(1, 30))
        blocks = _windows(ids, seq_len=8, stride=8)
        self.assertGreater(len(blocks), 1)
        self.assertTrue(all(len(b) == 9 for b in blocks))

    def test_too_short_yields_nothing(self):
        self.assertEqual(_windows([1], seq_len=8, stride=8), [])


class TestCorpusSkip(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), f"_tmp_corpus_{os.getpid()}")
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dump_without_sid_is_skipped(self):
        dump = os.path.join(self.tmp, "Orphan.1.dump.parquet")
        with open(dump, "wb") as fh:
            fh.write(b"not really a parquet")
        corpus = Corpus(_args(os.path.join(self.tmp, "*dump.parquet")), logging)
        self.assertEqual(list(corpus.iter_block_seqs()), [])

    def test_n_vocab_is_fixed(self):
        corpus = Corpus(_args(""), logging)
        self.assertEqual(corpus.n_vocab, 34)


def test_parse_corpus_empty_glob_counts_zero(tmp_path):
    pattern = os.path.join(str(tmp_path), "*dump.parquet")
    assert parse_corpus(_args(pattern), logging) == 0


def test_builds_blocks_from_pair(monty_pair, tmp_path):
    sid, dump = monty_pair
    shutil.copy(sid, os.path.join(str(tmp_path), "Monty_on_the_Run.sid"))
    shutil.copy(dump, os.path.join(str(tmp_path), "Monty_on_the_Run.1.dump.parquet"))
    corpus = Corpus(
        _args(os.path.join(str(tmp_path), "*dump.parquet"), seq_len=1024), logging
    )
    got = list(corpus.iter_block_seqs())
    assert len(got) == 1
    kind, path, meta = got[0]
    assert kind == "train"
    assert path.endswith(BLOCKS_SUFFIX)
    arr = np.load(path)
    assert arr.shape[1] == 1025
    assert ((arr >= 0) & (arr <= 33)).all()
    assert meta.subtune == 0


if __name__ == "__main__":
    unittest.main()
