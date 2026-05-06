"""Coverage tests for ``train_worker.get_tk``."""

import unittest

from preframr.train_worker import get_tk


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


if __name__ == "__main__":
    unittest.main()
