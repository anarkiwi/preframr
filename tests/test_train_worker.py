"""Coverage tests for ``train_worker.get_tk`` + ``train_worker``."""

import os
import string
import tempfile
import unittest

import zstandard as zstd

from preframr.train_worker import get_tk, train_worker


class TestGetTk(unittest.TestCase):
    def test_unigram(self):
        tk, trainer = get_tk(2048, tokenizer="unigram", initial_alphabet=["a", "b"])
        self.assertIsNotNone(tk)
        self.assertIsNotNone(trainer)

    def test_bpe(self):
        tk, trainer = get_tk(2048, tokenizer="bpe", initial_alphabet=["a", "b"])
        self.assertIsNotNone(tk)
        self.assertIsNotNone(trainer)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_tk(2048, tokenizer="lzw")


class TestTrainWorker(unittest.TestCase):
    def _write_uni(self, path, text):
        with zstd.open(path, "w") as f:
            f.write(text)

    def test_unigram_end_to_end(self):
        # Synthesise a tiny uni file (zstd-compressed text), then train
        # a unigram tokenizer over it. Covers the train_worker body
        # (read_uni / reader / tkmodel.save).
        with tempfile.TemporaryDirectory() as tmpdir:
            uni = os.path.join(tmpdir, "tiny.uni.zst")
            # 200+ chars across letters/digits/punctuation gives the
            # Unigram trainer enough coverage to fit a 64-vocab model.
            text = "".join((string.ascii_letters + string.digits + ".,;: ") * 6)
            self._write_uni(uni, text)
            tkmodel_path = os.path.join(tmpdir, "tk.json")
            train_worker(
                tokenizer="unigram",
                tkvocab=256,
                args_tkmodel=tkmodel_path,
                uni_files=[uni],
                initial_alphabet=list(string.ascii_letters + string.digits),
            )
            self.assertTrue(os.path.exists(tkmodel_path))
            self.assertGreater(os.path.getsize(tkmodel_path), 0)


if __name__ == "__main__":
    unittest.main()
