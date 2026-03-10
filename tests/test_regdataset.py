import logging
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from preframr.regdataset import (
    LowMemoryRandomSampler,
    RegDataset,
    _get_loader,
    get_prompt,
    glob_dumps,
    parser_worker,
)
from preframr.seq_mapper import SeqMeta
from preframr.stfconstants import (
    DUMP_SUFFIX,
    FRAME_REG,
    MODEL_PDTYPE,
    SET_OP,
    UNICODE_BASE,
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
        shuffle=False,
        batch_size=4,
    ):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.tkvocab = tkvocab
        self.tkmodel = tkmodel
        self.max_files = 1
        self.diffq = diffq
        self.token_csv = None
        self.cents = cents
        self.min_irq = min_irq
        self.max_irq = max_irq
        self.require_pq = False
        self.min_dump_size = 0
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_perm = 1
        self.dataset_csv = None
        self.df_map_csv = None


class TestGlobDumps(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def _make_file(self, name, size=100):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        return path

    def test_no_match(self):
        result = glob_dumps(os.path.join(self.tmpdir, "*.dump.parquet"), 10, 0, False)
        self.assertEqual(result, [])

    def test_basic_match(self):
        f = self._make_file("test.dump.parquet")
        result = glob_dumps(f, 10, 0, False)
        self.assertEqual(result, [f])

    def test_max_files(self):
        for i in range(3):
            self._make_file(f"test{i}.dump.parquet")
        result = glob_dumps(os.path.join(self.tmpdir, "*.dump.parquet"), 2, 0, False)
        self.assertEqual(len(result), 2)

    def test_min_dump_size_exclude(self):
        f = self._make_file("small.dump.parquet", size=10)
        result = glob_dumps(f, 10, 100, False)
        self.assertEqual(result, [])

    def test_min_dump_size_include(self):
        f = self._make_file("big.dump.parquet", size=200)
        result = glob_dumps(f, 10, 100, False)
        self.assertEqual(result, [f])

    def test_require_pq_excluded(self):
        # No matching parsed parquet -> excluded when require_pq=True
        f = self._make_file("test.dump.parquet")
        result = glob_dumps(f, 10, 0, True)
        self.assertEqual(result, [])

    def test_require_pq_included(self):
        # Matching parsed file (test.0.parquet matches PARSED_SUFFIX .[0-9]*.parquet)
        f = self._make_file("test.dump.parquet")
        self._make_file("test.0.parquet")
        result = glob_dumps(f, 10, 0, True)
        self.assertEqual(result, [f])

    def test_comma_separated(self):
        f1 = self._make_file("a.dump.parquet")
        f2 = self._make_file("b.dump.parquet")
        result = glob_dumps(f"{f1},{f2}", 10, 0, False)
        self.assertIn(f1, result)
        self.assertIn(f2, result)

    def test_comma_separated_max_files(self):
        # max_files=1 stops after the first pattern fills the quota
        f1 = self._make_file("c.dump.parquet")
        f2 = self._make_file("d.dump.parquet")
        result = glob_dumps(f"{f1},{f2}", 1, 0, False)
        self.assertEqual(len(result), 1)


class TestParserWorker(unittest.TestCase):
    def test_require_pq_no_match(self):
        # With require_pq=True and no pre-parsed parquet, parse() returns early -> dfs=[]
        args = FakeArgs()
        args.require_pq = True
        dump_file = "/nonexistent/file.dump.parquet"
        result_file, dfs = parser_worker(args, logging, dump_file, 1)
        self.assertEqual(result_file, dump_file)
        self.assertEqual(dfs, [])


class TestRegDataset(unittest.TestCase):
    def test_init(self):
        dataset = RegDataset(FakeArgs())
        self.assertEqual(dataset.n_vocab, 0)
        self.assertEqual(dataset.n_words, 0)
        self.assertEqual(dataset.reg_widths, {})

    def test_len_empty(self):
        dataset = RegDataset(FakeArgs())
        self.assertEqual(len(dataset), 0)

    def test_getitem_empty_raises(self):
        dataset = RegDataset(FakeArgs())
        with self.assertRaises(IndexError):
            _ = dataset[0]

    def test_load_dfs_no_args_raises(self):
        dataset = RegDataset(FakeArgs())
        with self.assertRaises(ValueError):
            list(dataset.load_dfs())

    def test_load_dfs_empty_dump_files_raises(self):
        # dump_files=[] is falsy so falls through to the reglogs check
        dataset = RegDataset(FakeArgs())
        with self.assertRaises(ValueError):
            list(dataset.load_dfs(dump_files=[]))


class TestLowMemoryRandomSampler(unittest.TestCase):
    def test_len(self):
        sampler = LowMemoryRandomSampler(list(range(10)), 5)
        self.assertEqual(len(sampler), 5)

    def test_iter_count(self):
        sampler = LowMemoryRandomSampler(list(range(10)), 7)
        items = list(sampler)
        self.assertEqual(len(items), 7)

    def test_iter_values_in_range(self):
        sampler = LowMemoryRandomSampler(list(range(10)), 20)
        for item in sampler:
            self.assertGreaterEqual(item, 0)
            self.assertLess(item, 10)


class TestGetLoader(unittest.TestCase):
    def _make_dataset(self, n=10):
        class FakeDataset(torch.utils.data.Dataset):
            def __len__(self):
                return n

            def __getitem__(self, i):
                return torch.zeros(4), torch.zeros(4)

        return FakeDataset()

    def test_sequential_sampler(self):
        loader = _get_loader(FakeArgs(shuffle=False), self._make_dataset())
        self.assertIsInstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_random_sampler(self):
        loader = _get_loader(FakeArgs(shuffle=1), self._make_dataset())
        self.assertIsInstance(loader.sampler, torch.utils.data.RandomSampler)

    def test_batch_size(self):
        loader = _get_loader(FakeArgs(batch_size=3), self._make_dataset())
        self.assertEqual(loader.batch_size, 3)


class TestGetPrompt(unittest.TestCase):
    def _make_dataset(self, seq, seq_meta):
        tokens = pd.DataFrame(
            [
                {"n": 0, "reg": FRAME_REG, "val": 0, "op": SET_OP, "subreg": -1},
                {"n": 1, "reg": 7, "val": 1, "op": SET_OP, "subreg": -1},
            ],
            dtype=MODEL_PDTYPE,
        )

        class FakeTokenizer:
            def __init__(self, t):
                self.tokens = t
                self.tkmodel = None

            def decode(self, encoded):
                return encoded

        class FakeDataset:
            def __init__(self):
                self.reg_widths = {}
                self.tokenizer = FakeTokenizer(tokens)

            def getseq(self, i):
                return torch.from_numpy(seq.astype(np.int64)), seq_meta

        return FakeDataset()

    def test_basic(self):
        seq = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int16)
        seq_meta = SeqMeta(irq=19000, df_file="test.dump.parquet", i=0)
        dataset = self._make_dataset(seq, seq_meta)

        class PromptArgs:
            start_seq = 0
            start_n = 0
            max_seq_len = 6
            prompt_seq_len = 4

        irq, n, prompt, prompt_compare, reg_start = get_prompt(
            PromptArgs(), dataset, logging
        )
        self.assertEqual(irq, 19000)
        # n = max_seq_len - prompt_seq_len
        self.assertEqual(n, 2)
        # prompt has batch dimension from unsqueeze(0)
        self.assertEqual(prompt.shape[0], 1)
        self.assertEqual(prompt.shape[1], 4)
        # start_n=0 -> empty preamble -> no reg_start entries
        self.assertEqual(reg_start, {})

    def test_max_seq_len_too_short_raises(self):
        seq = np.array([0, 1, 0, 1], dtype=np.int16)
        seq_meta = SeqMeta(irq=19000, df_file="test.dump.parquet", i=0)
        dataset = self._make_dataset(seq, seq_meta)

        class PromptArgs:
            start_seq = 0
            start_n = 0
            max_seq_len = 4
            prompt_seq_len = 4  # n = 0 -> ValueError

        with self.assertRaises(ValueError):
            get_prompt(PromptArgs(), dataset, logging)

    def test_start_n_none_uses_random(self):
        # start_n=None triggers randint path; result is still a valid prompt
        seq = np.zeros(50, dtype=np.int16)
        seq_meta = SeqMeta(irq=19000, df_file="test.dump.parquet", i=0)
        dataset = self._make_dataset(seq, seq_meta)

        class PromptArgs:
            start_seq = 0
            start_n = None
            max_seq_len = 10
            prompt_seq_len = 6

        irq, n, prompt, prompt_compare, reg_start = get_prompt(
            PromptArgs(), dataset, logging
        )
        self.assertEqual(irq, 19000)
        self.assertEqual(n, 4)
        self.assertEqual(prompt.shape[0], 1)


class TestPreload(unittest.TestCase):
    def _minimal_tokens(self):
        return pd.DataFrame(
            [{"n": 0, "reg": FRAME_REG, "val": 0, "op": SET_OP, "subreg": -1}],
            dtype=MODEL_PDTYPE,
        )

    def _patch_make_tokens(self, dataset, df_files=None):
        """Monkey-patch make_tokens to avoid needing real parquet files."""
        tokens = self._minimal_tokens()
        if df_files is None:
            df_files = []

        def mock_make_tokens(reglogs):
            dataset.tokenizer.tokens = tokens
            return df_files

        dataset.make_tokens = mock_make_tokens

    def test_tokens_provided_sets_tokenizer(self):
        # When tokens is passed directly, tokenizer.load is called and preload returns.
        dataset = RegDataset(FakeArgs())
        tokens = self._minimal_tokens()
        dataset.preload(tokens=tokens)
        self.assertTrue(dataset.tokenizer.tokens.equals(tokens))

    def test_tokens_provided_tkmodel_none(self):
        # tkmodel=None leaves tokenizer.tkmodel as None.
        dataset = RegDataset(FakeArgs())
        tokens = self._minimal_tokens()
        dataset.preload(tokens=tokens, tkmodel=None)
        self.assertIsNone(dataset.tokenizer.tkmodel)

    def test_early_return_no_csv(self):
        # no tkvocab, no dataset_csv, no df_map_csv -> returns after make_tokens
        dataset = RegDataset(FakeArgs())
        self._patch_make_tokens(dataset)
        dataset.preload()
        # Verify tokenizer.tokens was set by mock_make_tokens
        self.assertIsNotNone(dataset.tokenizer.tokens)

    def test_writes_token_csv(self):
        # token_csv is set -> tokens DataFrame is written to CSV after make_tokens
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            fname = f.name
        try:
            args = FakeArgs()
            args.token_csv = fname
            dataset = RegDataset(args)
            self._patch_make_tokens(dataset)
            dataset.preload()
            result = pd.read_csv(fname)
            self.assertIn("reg", result.columns)
            self.assertEqual(result["reg"].iloc[0], FRAME_REG)
        finally:
            os.unlink(fname)

    def test_writes_df_map_csv(self):
        # df_map_csv is set, no tkvocab, no dataset_csv -> df_map CSV is written then returns
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            fname = f.name
        try:
            args = FakeArgs()
            args.df_map_csv = fname
            dataset = RegDataset(args)
            self._patch_make_tokens(dataset, df_files=["a.dump.parquet"])
            dataset.preload()
            result = pd.read_csv(fname)
            self.assertIn("dump_file", result.columns)
            self.assertEqual(result["dump_file"].iloc[0], "a.dump.parquet")
        finally:
            os.unlink(fname)


class TestRegDatasetLoader(unittest.TestCase):
    pass
