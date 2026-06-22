"""BACC corpus builder: manifest parsing, windowing, blocks paths, skip behaviour."""

import argparse
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np

from preframr.corpus import (
    BLOCKS_SUFFIX,
    Corpus,
    _blocks_path,
    _load_cached_blocks,
    _windows,
    parse_corpus,
    parse_eval_manifests,
    read_manifest,
)
from preframr.tokenizer import PAD_ID


def _args(manifest="", **kw):
    defaults = dict(
        manifest=manifest,
        manifest_arg="",
        eval_manifest="",
        sid_root="",
        songlengths="",
        seq_len=512,
        block_stride=None,
        max_files=0,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestReadManifest(unittest.TestCase):
    def test_tab_and_default_subtune(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.list")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("A/Tune.sid\t2\n# comment\n\nB/Other.sid\n")
            self.assertEqual(
                read_manifest(p), [("A/Tune.sid", 2), ("B/Other.sid", 1)]
            )


class TestParseEvalManifests(unittest.TestCase):
    def test_named_subsets(self):
        self.assertEqual(
            parse_eval_manifests("eval-A=/a.list;eval-B-x=/b.list"),
            [("eval-A", "/a.list"), ("eval-B-x", "/b.list")],
        )

    def test_empty(self):
        self.assertEqual(parse_eval_manifests(""), [])


class TestBlocksPath(unittest.TestCase):
    def test_subtune_distinct(self):
        self.assertEqual(_blocks_path("/a/Tune.sid", 1), "/a/Tune.1" + BLOCKS_SUFFIX)
        self.assertEqual(_blocks_path("/a/Tune.sid", 2), "/a/Tune.2" + BLOCKS_SUFFIX)


class TestWindows(unittest.TestCase):
    def test_short_stream_one_padded_block(self):
        blocks = _windows(list(range(1, 6)), seq_len=8, stride=8)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 9)
        self.assertEqual(blocks[0][-1], PAD_ID)

    def test_long_stream_multiple_windows(self):
        blocks = _windows(list(range(1, 30)), seq_len=8, stride=8)
        self.assertGreater(len(blocks), 1)
        self.assertTrue(all(len(b) == 9 for b in blocks))

    def test_too_short_yields_nothing(self):
        self.assertEqual(_windows([1], seq_len=8, stride=8), [])


class TestLoadCachedBlocks(unittest.TestCase):
    def test_missing_returns_none(self):
        self.assertIsNone(_load_cached_blocks("/nope/x.blocks.npy", 8))

    def test_right_width_reused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.blocks.npy")
            np.save(p, np.zeros((3, 9), dtype=np.int16))
            got = _load_cached_blocks(p, 8)
            self.assertIsNotNone(got)
            self.assertEqual(got.shape, (3, 9))

    def test_wrong_width_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.blocks.npy")
            np.save(p, np.zeros((3, 5), dtype=np.int16))
            self.assertIsNone(_load_cached_blocks(p, 8))

    def test_empty_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.blocks.npy")
            np.save(p, np.zeros((0, 9), dtype=np.int16))
            self.assertIsNone(_load_cached_blocks(p, 8))


class TestCorpusSkip(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), f"_tmp_corpus_{os.getpid()}")
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_sid_is_skipped(self):
        man = os.path.join(self.tmp, "train.list")
        with open(man, "w", encoding="utf-8") as fh:
            fh.write("Gone/Orphan.sid\t1\n")
        corpus = Corpus(_args(man, sid_root=self.tmp), logging)
        self.assertEqual(list(corpus.iter_block_seqs()), [])

    def test_n_vocab_is_fixed(self):
        corpus = Corpus(_args(""), logging)
        self.assertEqual(corpus.n_vocab, 35)  # VOCAB 34 + PAD


def test_parse_corpus_empty_manifest_counts_zero(tmp_path):
    man = tmp_path / "train.list"
    man.write_text("", encoding="utf-8")
    assert parse_corpus(_args(str(man)), logging) == 0


def test_builds_blocks_from_manifest(tmp_path, monkeypatch):
    """End-to-end manifest -> blocks with the codec mocked (the real sid-only
    recovery is exercised by the codec repo's gate; here we pin the framework
    plumbing: manifest read, +1 model-space shift, subtune-distinct blocks path)."""
    import preframr.corpus as corpus_mod

    sid_rel = "X/Tune.sid"
    sid_path = tmp_path / "X" / "Tune.sid"
    sid_path.parent.mkdir(parents=True)
    sid_path.write_bytes(b"PSID-stub")
    man = tmp_path / "train.list"
    man.write_text(f"{sid_rel}\t1\n", encoding="utf-8")

    monkeypatch.setattr(corpus_mod, "subtune_frames", lambda *a, **k: 100)
    monkeypatch.setattr(corpus_mod, "recover_from_sid", lambda *a, **k: ("PROG", {}, None))
    monkeypatch.setattr(corpus_mod, "program_to_ids", lambda prog: list(range(34)) * 40)

    corpus = Corpus(_args(str(man), sid_root=str(tmp_path), seq_len=64), logging)
    got = list(corpus.iter_block_seqs())
    assert len(got) == 1
    kind, path, meta = got[0]
    assert kind == "train"
    assert path.endswith(".1" + BLOCKS_SUFFIX)
    arr = np.load(path)
    assert arr.shape[1] == 65
    assert int(arr.min()) >= 0 and int(arr.max()) <= 34  # 0..33 shifted to 1..34, PAD=0
    assert meta.subtune == 1


if __name__ == "__main__":
    unittest.main()
